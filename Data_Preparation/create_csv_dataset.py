import re
import random
from Bio import SeqIO
import csv
import random
import argparse


promoter_path =""
seq_length=""
enhancer_path = ""
csv_path=""



class Promoter:
    def __init__(self, nome, sequence):
        self.nome = nome
        self.sequence = sequence
class Enhancer:
    def __init__(self, gene, sequence):
        self.gene = gene
        self.sequence = sequence
    def __str__(self):
        return "Gene: "+ str(self.gene) +" sequence: "+ str(self.sequence)+"\n"

class Elem:
    def __init__(self, sequence, classe):
        self.sequence = sequence
        self.classe = classe

def create_dataset(data,label):
  dataset=[]
  for i in data:
    tmp = Elem(i.sequence, label)
    dataset.append(tmp)
  return dataset

def load_enhancers(enhancer_path):
  enhancer_list = []
  with open(enhancer_path,"r") as file_enhancer:
      
      for line in file_enhancer:
        gene, sequence = line.split('\t')
        tmp = Enhancer(gene, sequence)
        enhancer_list.append(tmp)
      
  #remove last character of enhancer sequence"\n"
  for item in enhancer_list:
    item.sequence = item.sequence[:-1]

  i = 0
  #Cut ehancer of seq_lenght
  while 1:
    if(i >= len(enhancer_list)) :
      break
    if (len(enhancer_list[i].sequence)> seq_length):
      copy = enhancer_list[i].sequence
      tmp = Enhancer(enhancer_list[i].gene, copy[seq_length:])
      enhancer_list.append(tmp)
      enhancer_list[i].sequence = copy[0:seq_length]

    if(len(enhancer_list[i].sequence)< seq_length):
      copy = enhancer_list[i].sequence
      while (len(copy)< seq_length):
        copy += 'N'
      enhancer_list[i].sequence = copy[0:seq_length]
    i += 1
  return enhancer_list

def load_promoters():
  promoter_list=[]
  # open fasta file to read promoter sequence
  with open(promoter_path, "r") as handle:
      for record in SeqIO.parse(handle, "fasta"):
        id = record.id
        sequence = record.seq
        tmp = Promoter(id, sequence)
        promoter_list.append(tmp)

  return promoter_list



def balance_dataset(promoter_dataset,enhancer_dataset):
      
    dim_en=len(enhancer_dataset)
    dim_prom=len(promoter_dataset)
    if dim_en>dim_prom:
        diff=dim_en-dim_prom
        for i in range(diff):
          item = random.choice(promoter_dataset)
          promoter_dataset.append(item)

    elif dim_prom>dim_en:
      diff=dim_prom-dim_en
      for i in range(diff):
          item = random.choice(enhancer_dataset)
          enhancer_dataset.append(item)
  

    dataset=promoter_dataset+enhancer_dataset
    print(f"promoter dim: {dim_prom}")
    print(f"enhancer_dim: {dim_en}")
    print(f"lenght of the dataset {len(dataset)}")
    
    return dataset

def save_data_into_csv(dataset, csv_path):
  #shuffle the data to have unordered data
  random.shuffle(dataset)
  data = [(Elem.sequence, Elem.classe) for Elem in dataset]

  with open(csv_path, "w", newline="") as csvfile:
    csvfile.write("sequence,label\n")
    writer = csv.writer(csvfile)
    writer.writerows(data)
  csvfile.close()  

if __name__=="__main__":

  parser=argparse.ArgumentParser
  parser.add_argument("-p","-promoter", help="promoter file (.fa)")
  parser.add_argument("-e","-enhancer",help="enhancer file (.txt)")
  parser.add_argument("-o","-output", help="output filename (.csv)")
  parser.add_argument("-l","-length", help="length of promoters")
  args=parser.parse_args()
  
  promoter_path=parser.promoter
  enhancer_path=parser.enhancer
  seq_length=parser.length
  csv_path=parser.output

  if promoter_path is None or enhancer_path is None or seq_length is None or csv_path is None:
    parser.print_help()
    parser.error("Parameters error")
  seq_length=int(seq_length)
  enhancer_list=load_enhancers(enhancer_path)
  promoter_list =load_promoters(promoter_path)
  enhancer_dataset=create_dataset(enhancer_list,0)
  promoter_dataset=create_dataset(promoter_list,1)
  dataset=balance_dataset(promoter_dataset,enhancer_dataset)
  save_data_into_csv(dataset,csv_path)