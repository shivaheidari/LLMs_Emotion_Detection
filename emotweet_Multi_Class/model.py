
import torch
import pytorch_lightning as pl
from transformers import AutoModel
import config
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric

class EmoTweetClassifier(pl.LightningModule):
   
   def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.pretrained_model = AutoModel.from_pretrained(config.model_name, return_dict=True)
        self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, num_classes)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(0.1)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.num_classes = num_classes

   def forward(self, input_ids, attention_mask, labels = None):
    #input 32*128
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask ,output_hidden_states = True, return_dict=True) 
    #print(output.keys())
    #size last hidden_state 32*128*768
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # 32, 768
    pooled_output = self.dropout(pooled_output)
    # 32, 768
    pooled_output = self.hidden(pooled_output)
    # 32, 768
    pooled_output = F.relu(pooled_output)
    # 32, 768
    pooled_output = self.dropout(pooled_output)
    # 32, 768
    logits = self.classifier(pooled_output)
    # 32, 13
    print(logits)
    loss = 0
    if labels is not None:
           loss = self.loss_func(logits, labels)
           return loss, logits
    else:
        return logits

   
   def training_step(self, batch, batch_idx):
       loss, output = self(**batch)
       self.log('train_loss', loss, prog_bar=True, logger=True)
       return {'loss': loss, 'score': output, 'y': batch['labels']}
   
   def validation_step(self, batch, batch_idx):
       loss, output = self(**batch)
       self.log('val_loss', loss, prog_bar=True, logger=True)
       return {"val_loss": loss, "score": output, "y": batch["labels"]}

   def test_step(self, batch, batch_idx):
         loss, output = self(**batch)
         self.log('test_loss', loss, prog_bar=True, logger=True)
         return {"test_loss": loss, "score": output, "y": batch["labels"]}

   def configure_optimizers(self):
       return optim.Adam(self.parameters(), lr=self.lr)
   
obj = EmoTweetClassifier(input_size=config.input_size, learning_rate = config.learning_rate, num_classes=config.num_classes)

