import json
from Bio.Seq import Seq
import csv
import pandas as pd
introns_pos={}
genomefile="GRCh38.p14.genome.fa"
with open("intron_position_1.json","r") as f:
    introns_pos=json.load(f)

sequence=""
with open("chr1.txt","r") as f:
    sequence=Seq(f.read().strip())

# print(sequence)
# print(f"len of sequence: {len(sequence)} ")
j=0
introns=[]
print(type(sequence))
for k,v in introns_pos.items():
    #print(f"id{k} num of exon {len(v)}")
    for i in range(len(v)-1):
        _,begin,strand=v[i]
        end,_,strand=v[i+1]
       
        if strand=="-":
            #print(f"id:{k} begin:{end} end:{begin} strand {strand}")
            subs=sequence[int(end)-1:int(begin)-1]
            subs=Seq.reverse_complement(subs)
        else:
            #print(f"id:{k} begin:{begin} end:{end} strand {strand}")
            subs=sequence[int(begin)-1:int(end)-1]
       

        if len(subs)>=1000:
            subs=subs[0:1000]
        introns.append(subs)

print(len(introns))    
with open("intron_1.csv","w") as f:
    f.write("Seq,\n")
    for intr in introns:
        f.write(f"{intr},\n")

