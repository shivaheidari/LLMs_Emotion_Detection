from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import config

#my own dataset class
class Twitter_emo(Dataset):

    def __init__(self,data_dir, tokenizer, max_len_token: int=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len_token = max_len_token
        self._prepare_data()

    def _prepare_data(self):
        data = pd.read_csv(self.data_dir)
        data = data.dropna()
        sentiment_map = {"empty": 0, "sadness": 1, "enthusiasm": 2, "neutral": 3, "worry": 4, "surprise": 5, "love": 6, "fun": 7, "hate": 8, "happiness": 9, "boredom": 10, "relief": 11, "anger": 12}
        data['sentiment'] = data['sentiment'].map(sentiment_map)
        data = data[["content", "sentiment"]]
        data.columns = ["text", "labels"]
        data["labels"] = data["labels"].astype(int)
        self.data = data

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        content = item['text']
        labels = item['labels']
        tokens = self.tokenizer.encode_plus(content, add_special_tokens = True, max_length=self.max_len_token, padding='max_length', truncation=True, return_tensors='pt')
        return dict(
            input_ids = tokens['input_ids'].flatten(),
            attention_mask = tokens['attention_mask'].flatten(),
            labels = torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.data)


#pl datamodule
class TwitterDataModule(pl.LightningDataModule):
   
   def __init__(self, data_dir, tokenizer, max_len_token, batch_size, num_workers,train_split, val_split, test_split):
       super().__init__()
       self.data_dir = data_dir
       self. tokenizer = tokenizer
       self.max_len_token = max_len_token
       self.batch_size = batch_size
       self.num_workers = num_workers
       self.train_split = train_split
       self.val_split = val_split
       self.test_split = test_split
       self.dataset = Twitter_emo(data_dir=self.data_dir, tokenizer=self.tokenizer, max_len_token=self.max_len_token)
  
   def setup(self, stage=None):
       
        
        num_samples = len(self.dataset)
        train_size = int(self.train_split * num_samples)
        val_size = int(self.val_split * num_samples)                
        test_size = num_samples - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
       
   def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
   def val_dataloader(self):
       return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
   def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
