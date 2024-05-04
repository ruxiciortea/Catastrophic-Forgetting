import pytorch_lightning as pl

from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data_load import load_data

MODEL_PATH = 'google/mt5-small'
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 3e-4
CHECKPOINT_PATH = '/out/models'

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model_path, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        # calls the __call__ method on self.model (instance of MT5ForConditionalGeneration), 
        # which calls the forward method of MT5ForConditionalGeneration
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # calls the __call__ method on self, which then calls the forward method
        output_loss = self(input_ids, attention_mask, labels)
        self.log('train_loss', output_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output_loss

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        output_loss = self(input_ids, attention_mask, labels)
        self.log('val_loss', output_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output_loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

def fine_tune():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, max_length=512, padding="max_length", truncation=True)
    model = MT5FineTuner(model_path=MODEL_PATH, tokenizer=tokenizer, learning_rate=LEARNING_RATE)
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath=CHECKPOINT_PATH, verbose=True)
    trainer = Trainer(accelerator='gpu', devices='auto', max_epochs=EPOCHS, callbacks=[model_checkpoint], logger=True)

    train_path = '/dataset/train_set.csv'
    val_path = '/dataset/val_set.csv'
    train_dataloader = load_data(tokenizer=tokenizer, data_path=train_path, batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = load_data(tokenizer=tokenizer, data_path=val_path, batch_size=BATCH_SIZE, shuffle=False)

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    fine_tune()