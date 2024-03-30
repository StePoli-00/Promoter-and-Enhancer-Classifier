import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import pytorch_lightning as pl
import os
from torch.utils.data import Dataset
import argparse

csv_path = ""
seq_length=""
dataset_path=""
training_embeddings_path=""
validation_embeddings_path=""
testing_embeddings_path=""
dataset_folder=""
class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sequence = self.df["sequence"][index]
        label = self.df["label"][index]
        return sequence, label

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
    
def create_OHE(train_loader):
    for i,item in enumerate(train_loader):
            dna,label = item
            # [32]--> [32,seq_length,4]
            my_tensor = torch.zeros(label.shape[0],seq_length,4)
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

            torch.save(my_tensor, training_path + "/embeddings/%d.pt" % i)
            torch.save(label, training_path + "/labels/%d.pt" % i )
          
    return
def split_dataset():
    for i, el in enumerate(os.listdir(training_embeddings_path)):
                if i % 10 == 0:
                    shutil.move(os.path.join(training_embeddings_path, el), validation_embeddings_path)
                    shutil.move(os.path.join(training_labels_path, el), validation_labels_path)
                if i % 10 == 1:
                    shutil.move(os.path.join(training_embeddings_path, el), testing_embeddings_path)
                    shutil.move(os.path.join(training_labels_path, el), testing_labels_path)
    return 

def check_folder(dataset_folder):
    if os.path.exists(dataset_folder):
        print("Folder already exists: Please choose another path/remove folder and run again")
        exit()
    else:
        create_folder()

def create_folder():
    os.makedirs(dataset_folder,exist_ok=True)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Parser for create Dataset')
    parser.add_argument("-s","--source_path",help="dataset_path (.csv)")
    parser.add_argument("-d","--destination_path",help="destination folder")
    parser.add_argument("-l","--seq_length", help="sequence length")
    
    args=parser.parse_args()
    csv_path=args.source_path
    dataset_path=args.destination_path
    seq_length=args.seq_length

    if csv_path is None or  dataset_path is None or seq_length is None:
        parser.print_help()
        parser.error("Parameters error")


    dataset_folder=os.path.join(dataset_path,"Dataset"+"_"+seq_length)
    check_folder(dataset_folder)

    seq_length=int(seq_length)
    training_path=os.path.join(dataset_folder,"training")
    testing_path=os.path.join(dataset_folder,"testing")
    validation_path=os.path.join(dataset_folder,"validation")
    
    training_embeddings_path = os.path.join(training_path, "embeddings")
    training_labels_path = os.path.join(training_path, "labels")

    validation_embeddings_path = os.path.join(validation_path, "embeddings")
    validation_labels_path = os.path.join(validation_path, "labels")

    testing_embeddings_path = os.path.join(testing_path, "embeddings")
    testing_labels_path = os.path.join(testing_path, "labels")

    os.makedirs(training_path,exist_ok=True)
    os.makedirs(os.path.join(training_path,"embeddings"), exist_ok=True)
    os.makedirs(os.path.join(training_path,"labels"),exist_ok=True)
    
    # Create custom dataset object


    if len(os.listdir(training_embeddings_path))==0:
        train_data_object = CustomDataSet(csv_path)
        train_loader = torch.utils.data.DataLoader(train_data_object,
                batch_size=32, shuffle = False)
        create_OHE(train_loader)
        

    os.makedirs(testing_path,exist_ok=True)
    os.makedirs(os.path.join(testing_path,"embeddings"), exist_ok=True)
    os.makedirs(os.path.join(testing_path,"labels"),exist_ok=True)
    os.makedirs(validation_path,exist_ok=True)
    os.makedirs(os.path.join(validation_path,"embeddings"),exist_ok=True)
    os.makedirs(os.path.join(validation_path,"labels"),exist_ok=True)


    if (len(os.listdir(testing_embeddings_path))==0 and len(os.listdir(validation_embeddings_path))==0):

        split_dataset()
        list_path=[training_path,testing_path,validation_path] 
        for folder in list_path:
            num=len(os.listdir(os.path.join(folder,"embeddings")))
            print(f"num of data in {folder} folder: {num}")
    else:
        print("Folders already exist and not empty. Please choose another destination path or remove the folder")
        exit()


