import pandas as pd
from torch.utils.data import DataLoader, Dataset
# from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW
    
class RecipeDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_path, dtype=object)

    def __len__(self):
        return len(self.data)
    
    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    def get_encoded_input(self, idx):
        input_text = (f"Extract the browning, drying and protein numerical values.\n"
                      f"Recipe instructions: {self.data['instructions'][idx]}")
        # input_text = (f"Extract the protein value.\n"
        #               f"Recipe instructions: {self.data['instructions'][idx]}")
        input_encoding = self.tokenize(input_text)
        return input_encoding

    def get_encoded_output(self, idx):
        target_sentence = self.data['target_sentence'][idx]
        drying_value = target_sentence.split('The drying value is ')[1].split('.')[0]
        browning_value = target_sentence.split('The browning value is ')[1].split('.')[0]
        protein_value = target_sentence.split('The protein value is ')[1].split('.')[0]

        output_text = (f"Drying value: {drying_value}\n"
                       f"Browning value: {browning_value}\n"
                       f"Protein value: {protein_value}\n")
        
        # output_text = f"Protein value: {protein_value}\n"
        
        output_encoding = self.tokenize(output_text)
        return output_encoding

    def __getitem__(self, idx):
        source_encoding = self.get_encoded_input(idx)
        target_encoding = self.get_encoded_output(idx)

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def load_data(tokenizer, data_path, batch_size, shuffle):
    dataset = RecipeDataset(tokenizer, data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return dataloader

# if __name__ == '__main__':
#     tokenizer = T5Tokenizer.from_pretrained('google/mt5-small', max_length=512, padding="max_length", truncation=True)
#     dataset = RecipeDataset(tokenizer, 'test_train_set.csv')

#     for i in range(len(dataset)):
#         sample = dataset[i]
#         input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
#         target_text = tokenizer.decode(sample['labels'], skip_special_tokens=True)
#         print(f"Sample {i}:")
#         print(f"  Input text: {input_text}")
#         print(f"  Target text: {target_text}")