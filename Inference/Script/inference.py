import pandas as pd
from Transformer import Transformer
from transformers import AutoTokenizer
import random
import torch
import argparse

data=""

def genera_numeri_unici(n):
    numeri_generati = set()
    
    while len(numeri_generati) < 100:
        numero = random.randint(1, n)  
        numeri_generati.add(numero)
            
    return numeri_generati



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Script to split Dataset into training, validation and testing.')
    parser.add_argument('-data',"--data_path", help='path to the introns csv')
    parser.add_argument('-weights',"--weights_path", help='path to the checkpoints of the model')
    args=parser.parse_args()
    
        
    data=args.data_path
    checkpoints=args.weights_path

    if data is None or checkpoints is None:
        parser.print_help()
        parser.error("Parameters error")


    df = pd.read_csv(data)
    X = df["Seq"]

    transformer = Transformer.load_from_checkpoint(checkpoints)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", cls_token="[token]",  trust_remote_code=True)      #tokenizer of DNABERT2

    numeri_casuali = genera_numeri_unici(32000)
    numeri_casuali = set(numeri_casuali)
   

    emb_list = []
    for l in numeri_casuali:
        emb_list.append("[token]" + X[l])

    emb = tokenizer(emb_list, return_tensors = 'pt', add_special_tokens=False, padding=True)
    results = transformer(emb['input_ids'].cuda(), None, emb['attention_mask'].cuda())

    enhancer = results < 0.2
    promoter = results > 0.8

    print("Considering " + str(len(numeri_casuali)) + " introns : \n" )

    n = torch.zeros(100)
    n[enhancer == True] = 1
    enhancer_tot = int(n.sum())
    print("percentage of enhancer : " + str(enhancer_tot)+'%')
    n = torch.zeros(100)
    n[promoter == True] = 1
    promoter_tot = int(n.sum())
    print("percentage of promoter : " + str(promoter_tot)+'%')