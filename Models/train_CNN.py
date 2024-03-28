# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_g4DZEvT3Fd7BFVBBa9d8nnXoF5DhVJN

Rete per la gestione di dei dati con shape [32,seq_len, 768] (fatti diventare [32,768, seq_len] per uniformare la dim lungo la quale fare la conv) tramite convoluzione 1D
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
Dataset_path=""


class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.df_sequences = os.listdir(os.path.join(path,'embeddings'))
        self.df_labels = os.listdir(os.path.join(path,'labels'))

    def __len__(self):
        return len(self.df_sequences)

    def __getitem__(self, index):
        sequence = torch.load(os.path.join(self.path,'embeddings',self.df_sequences[index])).swapaxes(1,2)
        #sequence = sequence.unsqueeze(dim=1)
        label = torch.load(os.path.join(self.path,"labels",self.df_labels[index])).float()
        return sequence, label





def collate(batch):
  return batch[0]




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



class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.pool = nn.MaxPool1d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pooling globale per ridurre le dimensioni

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        # for validation/testing
        self.accuracy = torchmetrics.Accuracy(task="binary")


    def forward(self, x):
        
        
        x = torch.relu(self.conv1(x))
        
        x = self.pool(x)
        
        x = torch.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = torch.relu(self.conv3(x))
        
        x = self.global_avg_pool(x)
       
        x = torch.relu(self.fc1(x.squeeze()))
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
        acc = self.accuracy(logits,y)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)




    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x).squeeze()
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits,y)

        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
      optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
      return optimizer


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parser for create Dataset')
    parser.add_argument('-d', '--dataset', help='Dataset name')
    args=parser.parse_args()
    Dataset_path=args.dataset
    if Dataset_path==None:
        parser.print_help()
        parser.error("Parameters error")
    DataTrainpath=os.path.join(Dataset_path,"training_data")
    DataTestpath=os.path.join(Dataset_path,"testing_data")
    DataValpath=os.path.join(Dataset_path,"validation_data")
    # Create custom dataset object
    train_data_object = MyDataSet(DataTrainpath)
    test_data_object = MyDataSet(DataTestpath)
    val_data_object = MyDataSet(DataValpath)
    for seq, _ in torch.utils.data.DataLoader(train_data_object,
        batch_size=1, shuffle = False, collate_fn=collate):
        print("sequenza in ingresso")
        print(seq.shape)
        break
    
    # Creazione del modello
    model = CNN()
    #creazione del trainer
    trainer = pl.Trainer(max_epochs =40)
    data_module = MyDataModule()
    #controllare le performance del modello da non trainato
    p = trainer.test(model, data_module)

    #fit del modello 
    trainer.fit(model, data_module)
    # Valutazione del modello
    p = trainer.test(model, data_module)
    print("Loss sul set di validazione:", p)


    """
    mod = CNN.load_from_checkpoint("lightning_logs/version_42/checkpoints/epoch=39-step=30600.ckpt")




    trainer = pl.Trainer(max_epochs = 40)

    data_module = MyDataModule()

    p = trainer.test(mod, data_module)

    for i, el in enumerate(test_data_object):

        print(el[0][mod(el[0]).squeeze() < 0.5])"""
