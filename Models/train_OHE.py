
import torch
import pandas as pd

from torch.utils.data import Dataset

datapath = "/home/antoniodeblasi/Scaricati/data.csv"
savepath = "/home/antoniodeblasi/Scaricati/Dataset_1_hot"
DataTrainpath="/home/antoniodeblasi/Scaricati/Dataset_1_hot"
DataTestpath="/home/antoniodeblasi/Scaricati/Dataset_1_hot_validation"
DataValpath="/home/antoniodeblasi/Scaricati/Dataset_1_hot_testing"

class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sequence = self.df["sequenza"][index]
        label = self.df[" id"][index]
        return sequence, label

# Create custom dataset object
train_data_object = CustomDataSet(datapath)

train_loader = torch.utils.data.DataLoader(train_data_object,
        batch_size=32, shuffle = False)



for i,item in enumerate(train_loader):
  dna,label = item

# [32]--> [32,599,4]
  my_tensor = torch.zeros(label.shape[0],599,4)
  for j,elem in enumerate(dna):
      for k,c in enumerate(elem):
          if c == "A":
              my_tensor[j,k,0] = 1
          if c == "T":
              my_tensor[j,k,1] = 1
          if c == "G":
              my_tensor[j,k,2] = 1
          if c == "C":
              my_tensor[j,k,3] = 1




  torch.save(my_tensor, savepath + "/embeddings/%d.pt" % i)
  torch.save(label, savepath + "/labels/%d.pt" % i )

import numpy as np
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl

import os

class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.df_sequences = os.listdir(path+'/embeddings')
        self.df_labels = os.listdir(path+'/labels')

    def __len__(self):
        return len(self.df_sequences)

    def __getitem__(self, index):
        sequence = torch.load(self.path+'/embeddings/'+self.df_sequences[index])
        sequence = sequence.swapaxes(1,2)

        label = torch.load(self.path+'/labels/'+self.df_labels[index]).float()


        return sequence, label




# Create custom dataset object
train_data_object = MyDataSet(DataTrainpath)
test_data_object = MyDataSet(DataTestpath)
val_data_object = MyDataSet(DataValpath)

def collate(batch):
  (a, b) = batch[0]
  return (a,b)

class MyDataModule(pl.LightningDataModule):

  def setup(self, stage):
    self.dataset = ""#MyDataSet("")


  def train_dataloader(self):
    return torch.utils.data.DataLoader(train_data_object,
        batch_size=1, shuffle = False, collate_fn=collate)
  def val_dataloader(self):
    return torch.utils.data.DataLoader(val_data_object,
        batch_size=1, shuffle = False, collate_fn=collate)
  def test_dataloader(self):
    return torch.utils.data.DataLoader(test_data_object,
       batch_size=1, shuffle = False, collate_fn=collate)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=30, kernel_size=19, padding="same")
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=5, padding="same")

        self.pool = nn.MaxPool1d(10,stride=10)


        self.fc1 = nn.Linear(640, 513)
        self.fc2 = nn.Linear(513, 1)
        self.sigmoid = nn.Sigmoid()

        # for validation/testing
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1=torchmetrics.classification.BinaryF1Score()
        self.precision=torchmetrics.classification.BinaryPrecision()
        self.recall=torchmetrics.classification.BinaryRecall()

    def forward(self, x):
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = torch.flatten(x,start_dim=1)
        # print(x.shape)

        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def cross_entropy_loss(self, logits, labels):
      return F.binary_cross_entropy(logits, labels)




    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x).squeeze()
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x).squeeze()
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        f1_val=self.f1(logits,y)
        precision_val=self.precision(logits,y)
        recall_val=self.recall(logits,y)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_f1', f1_val)
        self.log('val_precision', precision_val)
        self.log('val_recall', recall_val)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x).squeeze()
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        f1_test=self.f1(logits,y)
        precision_test=self.precision(logits,y)
        recall_test=self.recall(logits,y)

        self.log('test_loss', loss)
        self.log('test_f1', f1_test)
        self.log('test_accuracy', acc)
        self.log('test_precision', precision_test)
        self.log('test_recall', recall_test)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer

# Creazione del modello
model = CNN()

torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True
num_epoch=""
trainer = pl.Trainer(max_epochs =num_epoch)

data_module = MyDataModule()
p = trainer.test(model, data_module)

trainer.fit(model, data_module)
# Valutazione del modello
p = trainer.test(model, data_module)
print("Loss sul set di validazione:", p)