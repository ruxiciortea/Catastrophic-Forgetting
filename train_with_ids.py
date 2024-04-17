import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Adafactor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

class RecipeDataset(Dataset):
    def __init__(self, tokenizer, data_file):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = (f"Does the following recipe have oven settings? "
                      f"If yes, extract the oven settings as well as the protein, "
                      f"browning and drying numerical values.\n"
                      f"Recipe title: {self.data['title'][idx]}.\n"
                      f"Recipe instructions: {self.data['instructions'][idx]}")

        if self.data['oven'][idx] == 'yes':
            output_text = (f"Oven settings: Yes; Cooking parameters: {self.data['baking_steps'][idx]}\n"
                           f"Classification values: {self.data['target_sentence'][idx]}")
        else:
            output_text = "Oven settings: No"

        recipe_id = self.data['recipe_id'][idx]

        source_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            output_text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']

        return {
            'recipe_id': recipe_id,
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

class RecipeDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, tokenizer, batch_size):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = RecipeDataset(self.tokenizer, self.train_file)
        self.val_dataset = RecipeDataset(self.tokenizer, self.val_file)
        self.test_dataset = RecipeDataset(self.tokenizer, self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model_name, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.predictions = []
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        recipe_ids = batch['recipe_id']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=2, early_stopping=True)

        preds = [self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for gen_id in generated_ids]
        self.predictions.extend(zip(recipe_ids.tolist(), preds))

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        current_epoch = self.current_epoch
        prediction_df = pd.DataFrame(self.predictions, columns=['recipe_id', 'predicted_baking_steps'])
        prediction_df.to_csv(f'/out/predictions/test_predictions_with_ids_batch8_epoch{current_epoch}.csv', index=False)
        self.predictions = []

    def configure_optimizers(self):
        return Adafactor(
            self.parameters(),
            lr=self.learning_rate,
            scale_parameter=False,
            relative_step=False
        )

def train_model():
    model_name = 'google/mt5-small'
    batch_size = 8
    number_of_epochs = 4
    learning_rate = 3e-4

    # model = MT5FineTuner.load_from_checkpoint(checkpoint_path=f"{location}/{ckpt_name}", model_path=MODEL_PATH, lr=LR, train_len=0, epochs=EPOCHS)
    #     trainer = pl.Trainer(
    #         devices=1,
    #         accelerator='gpu',
    #         default_root_dir='/VOITA/out/t5/',
    #         inference_mode=False
    #     )
    #     outputs = trainer.predict(model, test_dataloader)

    # tokenizer = MT5Tokenizer.from_pretrained(model_name)
    tokenizer = MT5Tokenizer.from_pretrained(
        model_name,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    data_module = RecipeDataModule(
        train_file='/dataset/train_set.csv',
        val_file='/dataset/val_set.csv',
        test_file='/dataset/test_set.csv',
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    model = MT5FineTuner(
        model_name=model_name,
        tokenizer=tokenizer,
        learning_rate=learning_rate
    )

    trainer = Trainer(
        max_epochs=number_of_epochs,
        devices='4',
        accelerator='gpu',
        strategy='ddp',
        logger=True,
        callbacks=[
            ModelCheckpoint(
                dirpath='/out/models',
                save_top_k=1,
                verbose=True,
                monitor='val_loss',
                mode='min'
            )
        ],
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == '__main__':
    train_model()