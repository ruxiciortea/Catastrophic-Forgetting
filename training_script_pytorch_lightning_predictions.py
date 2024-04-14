from torch.utils.data import DataLoader, Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Adafactor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import pytorch_lightning as pl

class RecipeDataset(Dataset):
    def __init__(self, tokenizer, data_file):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = (f"Does the following recipe have oven settings? "
                      f"If yes, extract the oven settings as well as"
                      f"the protein, browning and drying numerical values.\n"
                      f"Recipe title: {self.data['title'][idx]}.\n"
                      f"Recipe instructions: {self.data['instructions'][idx]}")

        # if self.data['oven'][idx] == 'yes':
        #     output_text = (f"Oven settings: Yes\nCooking parameters: {self.data['baking_steps'][idx]}\n"
        #                    f"Classification values: {self.data['target_sentence'][idx]}")
        # else:
        #     output_text = "Oven settings: No"
        if self.data['oven'][idx] == 'yes':
            output_text = (f"Oven settings: Yes\nCooking parameters: {self.data['baking_steps'][idx]}\n"
                   f"Protein value: {self.data['protein'][idx]}\n"
                   f"Browning value: {self.data['browning'][idx]}\n"
                   f"Drying value: {self.data['drying'][idx]}\n"
                   f"Classification values: {self.data['target_sentence'][idx]}")
        else:
            output_text = "Oven settings: No"

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
        labels[labels == 0] = -100

        return {
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15, persistent_workers=True)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model_name, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.predictions = []

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
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)

        generated_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=512, num_beams=2, early_stopping=True)
        preds = [self.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for i in generated_ids]
        self.predictions.extend(preds)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        pd.DataFrame(self.predictions, columns=['predictions']).to_csv('/out/test_predictions.csv',
                                                                       index=False)
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
    batch_size = 4
    number_of_epochs = 2
    learning_rate = 3e-4

    tokenizer = MT5Tokenizer.from_pretrained(model_name)

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
        devices="auto",
        accelerator="auto",
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
