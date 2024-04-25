from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer
import argparse

datapath=""
savepath=""
tokenizer_type=""

class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sequence = self.df["sequence"][index]
        sequence = "[token]" + sequence #adding the cls_token for classification purpose
        label = self.df["label"][index]
        return sequence, label
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Script to split Dataset into training, validation and testing.')
    parser.add_argument('-csv',"--csv_path", help='path to the csv file')
    parser.add_argument('-dataset',"--dataset_path",help="path to the Dataset folder")
    parser.add_argument('-tokenizer',"--tokenizer_type", help = "Insert Bert for bert tokenizer, DnaBert for tokenizer of DnaBert2 or Deep for tokenizer of InstaDeep")
    args=parser.parse_args()
    
    datapath = args.csv_path
    savepath = args.dataset_path
    tokenizer_type = args.tokenizer_type


    if(tokenizer_type == "DnaBert"):
       tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", cls_token="[token]",  trust_remote_code=True)      #tokenizer of DNABERT2
    elif(tokenizer_type == "Bert"):
       tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cls_token="[token]")  
    elif(tokenizer_type == "Deep"):
       tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", cls_token="[token]")   #tokenizer of InstaDeep
    else: 
       parser.print_help()
       parser.error("Parameters error")


    # Create custom dataset object
    train_data_object = CustomDataSet(datapath)

    train_loader = torch.utils.data.DataLoader(train_data_object,batch_size=32, shuffle = False)


    for i,item in enumerate(train_loader):
        dna,label = item
        inputs = tokenizer(dna, return_tensors = 'pt', add_special_tokens=False, padding=True)
        
        ids = inputs["input_ids"]
        att_mask = inputs["attention_mask"]
        
        torch.save(ids, savepath + "/ids/%d.pt" % i)
        torch.save(att_mask, savepath + "/att_mask/%d.pt" % i)
        torch.save(label, savepath + "/labels/%d.pt" % i )
    
    print("Data processed and saved in folder : " + savepath)