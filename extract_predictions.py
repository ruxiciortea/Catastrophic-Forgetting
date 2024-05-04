import pandas as pd
import pytorch_lightning as pl

from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer 
from data_load import load_data

MODEL_PATH = 'google/mt5-base'
CHECKPOINT_PATH = '/out/models/base_all_epoch_6_batch_4_lr_1e.ckpt'
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4
MAX_SEQUENCE_LENGTH = 512

class MT5Classifier(pl.LightningModule):
    def __init__(self, model_path, tokenizer, learning_rate):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.predictions = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss

    def test_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log('test_loss', output.loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return output.loss
    
    def predict_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        generated_items = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_SEQUENCE_LENGTH, num_beams=2)
        generated_predicitons = [self.tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in generated_items]
        self.predictions.extend(generated_predicitons)

        return self.predictions  
    
    def on_predict_epoch_end(self):
        prediction_df = pd.DataFrame(self.predictions, columns=['predicted_browning_value'])
        prediction_df.to_csv('/out/predictions/predictions_all_tasks.csv', index=False)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)   

def extract_predictions():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, max_length=MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)
    model = MT5Classifier.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, model_path=MODEL_PATH, lr=LEARNING_RATE, train_len=0, epochs=EPOCHS)
    trainer = Trainer(accelerator='gpu', devices='auto', max_epochs=EPOCHS, logger=True)

    data_path = '/dataset/predict_set.csv'
    dataloader = load_data(tokenizer=tokenizer, data_path=data_path, batch_size=BATCH_SIZE, shuffle=False)

    trainer.test(model, dataloader)
    trainer.predict(model, dataloader)

if __name__ == '__main__':
    extract_predictions()