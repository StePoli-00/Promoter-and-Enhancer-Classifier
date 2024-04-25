from torch.utils.data import Dataset
import pandas as pd
import os 
import torch
from Transformer import Transformer
import pytorch_lightning as pl
import argparse


DataTrainpath = ""
DataTestpath = ""
DataValpath = ""

class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.df_ids = os.listdir(path+'/ids')
        self.df_att_mak = os.listdir(path+'/att_mask')
        self.df_labels = os.listdir(path+'/labels')

    def __len__(self):
        return len(self.df_ids)

    def __getitem__(self, index):
        ids = torch.load(self.path+'/ids/'+self.df_ids[index])
        att_mask = torch.load(self.path+'/att_mask/'+self.df_att_mak[index])
        label = torch.load(self.path+'/labels/'+self.df_labels[index]).float()
        print("loading file"+self.path+'/ids/'+self.df_ids[index])

        return ids,label,att_mask
    

def collate(batch): #
  (a, b, c) = batch[0]
  return (a,b,c)

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



if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Script to train the model.')
    parser.add_argument('-train',"--training_path", help = 'path to the training Dataset folder')
    parser.add_argument('-test', '--testing_path', help = 'path to the testing Dataset folder')
    parser.add_argument('-validation',"--validation_path", help = "path to the validation Dataset folder")
    args=parser.parse_args()
    
    DataTrainpath=args.training_path
    DataTestpath=args.testing_path
    DataValpath=args.validation_path

    if DataTrainpath is None or DataTrainpath is None or DataTrainpath is None:
       parser.print_help()
       parser.error("Parameters error")

    # Create custom dataset object
    train_data_object = MyDataSet(DataTrainpath)
    test_data_object = MyDataSet(DataTestpath)
    val_data_object = MyDataSet(DataValpath)


    src_vocab_size = 35000
    tgt_vocab_size = 1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 500
    dropout = 0.1


    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)



    trainer = pl.Trainer(max_epochs=4)
    data_module = MyDataModule()


    trainer.fit(transformer, data_module)
    # Valutazione del modello
    p = trainer.test(transformer, data_module)
    print("Loss sul set di testing:", p)

