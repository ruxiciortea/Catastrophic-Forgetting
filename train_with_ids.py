import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW
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

        source_encoding = self.tokenizer(input_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(output_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

        return {
            'recipe_id': self.data['recipe_id'][idx],
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class RecipeDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, tokenizer, batch_size):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = RecipeDataset(self.tokenizer, self.train_file)
        self.val_dataset = RecipeDataset(self.tokenizer, self.val_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model_name, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask, labels)
        self.log('train_loss', output.loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask, labels)
        self.log('val_loss', output.loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

def train_model():
    model_name = 'google/mt5-base'
    batch_size = 4
    epochs = 8
    learning_rate = 3e-4

    tokenizer = T5Tokenizer.from_pretrained(model_name, max_length=512, padding="max_length", truncation=True)
    model = MT5FineTuner(model_name=model_name, tokenizer=tokenizer, learning_rate=learning_rate)

    trainer = Trainer(
        max_epochs=epochs,
        devices='auto',
        accelerator='gpu',
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

    data_module = RecipeDataModule(
        train_file='/dataset/train_set.csv',
        val_file='/dataset/val_set.csv',
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    train_model()