import numpy as np
from sklearn.model_selection import train_test_split
import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModel, BertConfig
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from torch.utils.data import Dataset
datapath = ""
savepath = ""
embedding_folder=""


class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sequence = self.df["sequence"][index]
        label = self.df["label"][index]
        return sequence, label

def check_folder():
    if os.path.exists(embedding_folder):
        print("Folder already exists,choose another path/remove folder and run again")
        exit()
    else:
        create_folder()


def create_folder():
    os.makedirs(embedding_folder,exist_ok=True)
    os.makedirs(os.path.join(embedding_folder,"embeddings"), exist_ok=True)
    os.makedirs(os.path.join(embedding_folder,"labels"), exist_ok=True)

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("-s","--source_path",help="dataset_path (.csv)")
    parser.add_argument("-d","--destination_path",help="destination folder")
    args=parser.parse_args()
    datapath=args.source_path
    savepath=args.destination_path

    if datapath is None or savepath is None:
        parser.print_help()
        parser.error("Parameters error")
    
    
    embedding_folder=os.path.join(savepath,"Embedding")
    check_folder()
    
    # Create custom dataset object
    train_data_object = CustomDataSet(datapath)
    #To simply and have faster computation we work with a batch of 32 insted of processing each single sequence
    train_loader = torch.utils.data.DataLoader(train_data_object,batch_size=32, shuffle = False)



    """The sequence formed by basis couple (ATCG) is fed into Tokenizer, in particular what we need to represent data is the input_ids which is an integer array
    that represent token ID generated by the Tokenizer. Each work or text piece is mapped into a specific ID.
    Those ID are specifi for BERT model and used as input for the model.
    After that the input_ids are fed into the model which generare embedding of shape [32,sequence_length,768] (the first shape is due to batch size).
    As input to the model we use both data generated as described before and also data with shape [32,768] built with a mean on the first shape."""

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_config(config, trust_remote_code=True)

    for i,item in enumerate(train_loader):
        dna,label = item
        inputs = tokenizer(dna, return_tensors = 'pt', padding=True)
        print(inputs)
        #inputs = tokenizer(dna, return_tensors='pt', padding=True, truncation=True, max_length=max_sequence_length)
        ids = inputs["input_ids"]
        hidden_states = model(ids)[0] # [32, sequence_length, 768]
        print(hidden_states)
        #embedding_mean = torch.mean(hidden_states, dim=1) if you want something with shape [768]

        torch.save(hidden_states, embedding_folder + "/embeddings/%d.pt" % i)
        torch.save(label, embedding_folder + "/labels/%d.pt" % i )
    
        