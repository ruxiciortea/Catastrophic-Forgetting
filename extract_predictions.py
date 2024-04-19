import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer

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

        return {
            'recipe_id': self.data['recipe_id'][idx],
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class RecipeDataModule(pl.LightningDataModule):
    def __init__(self, test_file, tokenizer, batch_size):
        super().__init__()
        self.test_dataset = None
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage):
        self.test_dataset = RecipeDataset(self.tokenizer, self.test_file)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model_path, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.predictions = []
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def test_step(self, batch):
        recipe_ids = batch['recipe_id']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        generated_items = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=2, early_stopping=True)

        preds = [self.tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for item in generated_items]
        self.predictions.extend(zip(recipe_ids.tolist(), preds))
        prediction_df = pd.DataFrame(self.predictions, columns=['recipe_id', 'predicted_baking_steps'])
        prediction_df.to_csv('/out/predictions/test_predictions_with_ids_small.csv', index=False)

        self.log('test_loss', outputs.loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'test_loss': outputs.loss}  
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)      

def train_model():
    model_name = 'google/mt5-base'
    checkpoint_path = '/out/models/epoch=5-step=15570-v1.ckpt'
    batch_size = 8
    number_of_epochs = 6
    learning_rate = 3e-4

    tokenizer = T5Tokenizer.from_pretrained(model_name, max_length=512, padding="max_length", truncation=True)
    model = MT5FineTuner.load_from_checkpoint(checkpoint_path=checkpoint_path, model_path=model_name, lr=learning_rate, train_len=0, epochs=number_of_epochs)
    trainer = Trainer(max_epochs=number_of_epochs, devices='auto', accelerator='gpu')
    data_module = RecipeDataModule(test_file='/dataset/test_set.csv', tokenizer=tokenizer, batch_size=batch_size)

    trainer.test(model, data_module)

if __name__ == '__main__':
    train_model()