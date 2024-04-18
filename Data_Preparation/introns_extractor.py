import json
from Bio import SeqIO
from Bio.Seq import Seq
import csv
import pandas as pd
import argparse


exons={}
genomefile=""
exon_file=""
intron_file=""


def save_introns(introns):   
    with open(intron_file+".csv","w") as f:
        f.write("Seq,\n")
        for intr in introns:
            f.write(f"{intr},\n")

def extract_introns():

    with open(exon_file,"r") as f:
        exons=json.load(f)

    chromosome=list(exons.keys())

    for record in SeqIO.parse(genomefile,"fasta"):
        
        if record.id not in chromosome:
            break
        sequence=record.seq
        
        introns=[]
        exon_list=exons[record.id]
        for i in range(len(exon_list)-1):
            _,begin,strand=exon_list[i]
            end,_,strand=exon_list[i+1]
        
            if strand=="-":
                #print(f"id:{k} begin:{end} end:{begin} strand {strand}")
                subs=sequence[int(end)-1:int(begin)-1]
                subs=Seq.reverse_complement(subs)
            else:
                #print(f"id:{k} begin:{begin} end:{end} strand {strand}")
                subs=sequence[int(begin)-1:int(end)-1]

            """taking all the exon of one chromosome and extract introns in this pattern
            may happend that exon very far are treated as closer, since they are inside the same list.
            luckily the slicing produce no output because the begin is and higher pos then end"""
            if len(subs)!=0:
                if len(subs)>=1000:
                    subs=subs[0:1000]
                
                introns.append(subs)
    
    print(f"{len(introns)} introns extracted")
    return introns

if __name__=="__main__":
                
    parser=argparse.ArgumentParser()
    parser.add_argument("-gr","--genome_reference",help="genome fasta file path (.fa)")
    parser.add_argument("-ex","--exon",help="exon file (.json)")
    parser.add_argument("-out","--output_file",help="output introns file basename")
   
    args=parser.parse_args()
    genomefile=args.genome_reference
    exon_file=args.exon
    intron_file=args.output_file
    
    if args==None or genomefile==None or exon_file==None or intron_file==None:
        parser.print_help()
        parser.error("Parameter Error")
    
    introns=extract_introns()
    save_introns(introns)

