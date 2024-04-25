
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import os
import shutil
import random
import argparse

Dataset_path=""

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
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=30, kernel_size=19, padding="same")
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=5, padding="same")

        self.pool = nn.MaxPool1d(7,stride=7)


        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Sigmoid()

        # for validation/testing
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1=torchmetrics.classification.BinaryF1Score()
        self.precision=torchmetrics.classification.BinaryPrecision()
        self.recall=torchmetrics.classification.BinaryRecall()
        self.MCC=torchmetrics.MatthewsCorrCoef(task="binary")
        self.AUC=torchmetrics.AUROC(task="binary")
        
    def forward(self, x):
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.flatten(x,start_dim=1)
        print(x.shape)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        print(x.shape)
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
        mcc = self.MCC(logits, y)
        auroc = self.AUC(logits, y)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_f1', f1_val)
        self.log('val_precision', precision_val)
        self.log('val_recall', recall_val)
        self.log("val_MCC",mcc)
        self.log("val_AUROC",auroc)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x).squeeze()
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        f1_test=self.f1(logits,y)
        precision_test=self.precision(logits,y)
        recall_test=self.recall(logits,y)
        mcc = self.MCC(logits, y)
        auroc = self.AUC(logits, y)

        self.log('test_loss', loss)
        self.log('test_f1', f1_test)
        self.log('test_accuracy', acc)
        self.log('test_precision', precision_test)
        self.log('test_recall', recall_test)
        self.log("val_MCC",mcc)
        self.log("val_AUROC",auroc)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer

def reset_dataset(train,test,validation):
    
    test_emb=os.listdir(os.path.join(test,"embeddings"))
    validation_emb=os.listdir(os.path.join(validation,"embeddings"))
    test_lab=os.listdir(os.path.join(test,"labels"))
    validation_lab=os.listdir(os.path.join(validation,"labels"))
    
    if len(test_emb) !=0 or len(validation_emb) !=0 or len(test_lab) !=0 or len(validation_lab) !=0:
      
        for el in validation_emb:
            shutil.move(os.path.join(validation,"embeddings",el), os.path.join(train,"embeddings"))
            
        for el in test_emb:
            shutil.move(os.path.join(test,"embeddings",el), os.path.join(train,"embeddings"))
    
        for el in validation_lab:
            shutil.move(os.path.join(validation,"labels",el), os.path.join(train,"labels"))
        
        for el in test_lab:
            shutil.move(os.path.join(test,"labels",el), os.path.join(train,"labels")) 


def collate(batch):
  (a, b) = batch[0]
  return (a,b)

def split_dataset(train,test,validation):
  
  num_elem = len(os.listdir(os.path.join(train,"embeddings")))
  emb=os.listdir(os.path.join(DataTrainpath,"embeddings"))
  lab=os.listdir(os.path.join(DataTrainpath,"labels"))
  for el in emb:
    i = random.randint(0, num_elem -1)
    if i % 10 == 0:
      shutil.move(os.path.join(train,"embeddings",el),os.path.join(validation,"embeddings"))
      shutil.move(os.path.join(train,"labels",el),os.path.join(validation,"labels"))
    if i % 10 == 1:
      shutil.move(os.path.join(train,"embeddings",el),os.path.join(test,"embeddings"))
      shutil.move(os.path.join(train,"labels",el),os.path.join(test,"labels")) 
   



if __name__ == "__main__": 
  
  parser = argparse.ArgumentParser(description="Parser for OHE training")
  parser.add_argument('-c',"--checkpoint",help="checkpoint path (.ckpt files)")
  parser.add_argument("-d", "--dataset_path", help="dataset complete path")
  
  args = parser.parse_args()
  checkpoint=args.checkpoint
  Dataset_path = args.dataset_path
  
  if Dataset_path is None and checkpoint is None: 
    parser.print_help()
    parser.error("Parameter Dataset_Path error")
  
  DataTrainpath=os.path.join(Dataset_path,"training")
  DataTestpath=os.path.join(Dataset_path,"testing")
  DataValpath=os.path.join(Dataset_path,"validation")
  reset_dataset(DataTrainpath, DataTestpath, DataValpath)
  split_dataset(DataTrainpath, DataTestpath, DataValpath)

  # Create custom dataset object
  train_data_object = MyDataSet(DataTrainpath)
  test_data_object = MyDataSet(DataTestpath)
  val_data_object = MyDataSet(DataValpath)

  # Creazione del modello
  model = CNN.load_from_checkpoint(checkpoint)

  torch.backends.cuda.matmul.allow_tf32 = True

  torch.backends.cudnn.allow_tf32 = True
  num_epoch=40
  trainer = pl.Trainer(max_epochs = num_epoch)

  data_module = MyDataModule()
  p = trainer.test(model, data_module)

  # Test del modello
  print("Test:", p)