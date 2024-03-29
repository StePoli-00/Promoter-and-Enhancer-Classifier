from math import ceil, floor
import os
from Bio import SeqUtils, SeqIO, Seq, SeqRecord
import random
import json
random
import argparse
def copy_list_on_file(list, filename):
    try:
        with open(filename, 'w') as file:
            for elemento in list:
                file.write(str(elemento) + '\n')
        print("List copied on: ", filename)
    except IOError:
        print("Error during copy on file")

def copy_dict_on_file(dict, filename):
    with open(filename,"w") as f:
        json.dump(dict,f)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parser for generate sequence of N length')
    parser.add_argument('-b',"--bed", help='Bed file for Promoter')
    parser.add_argument('-g', '--genome', help='Genome fasta file')
    parser.add_argument('-l',"--length",help="Promoter length sequence")
    parser.add_argument('-o',"--output",help="dest path for the output file")
    args=parser.parse_args()
   
    bedfilename=args.bed
    genomefile=args.genome
    length=args.length
    outputdir=args.output
    outfilename=""
    cut_list=[]
    if bedfilename is None or genomefile is None:
        parser.print_help()
        parser.error("Parameters error")
    if outputdir is None:
            outputdir="."
    if length is None:
        cut_list=[(5,5),(10,10),(50,50),(150,150),(500,500),(1000,1000)]
        outfilename=os.path.join(outputdir,"promoters_"+"mixed"+".fa")
    else :
        
        outfilename=os.path.join(outputdir,"promoters_"+length+".fa")
        length=int(length)
        if length%2!=0:
            cut_list=[(length-ceil(length//2),floor(length//2))]    
        else: 
            cut_list=[(length//2,length//2)]
   
    promoters={}
    item=[]
    prev=""
   
    with open(bedfilename,"r") as f:
        for line in f: 
            
            name, start, stop, id, uno, strand = line.split()
            tmp_tuple = (name, start, stop, strand)
            cur=name
            if prev=="":
                prev=name
                item.append(tmp_tuple)
            elif cur!=prev:
                promoters[prev]=item.copy()
                item.clear()
                item.append(tmp_tuple)
                prev=cur
            else:
                item.append(tmp_tuple)

            
            
    #copy_list_on_file(promoters, 'prova.txt')
    #copy_dict_on_file(promoters,"promoters.json")


   
    final_promoter_list = []

    for record in SeqIO.parse(genomefile, "fasta"): 
        print(f"processing promoters of {record.name} chromosome")
        sequence = record.seq
        if record.name in promoters.keys():
            pos=promoters[record.name]
            for p in pos:
                _, real_start, real_stop, strand=p
                #print(random.choice(cut_list))
                start,stop=random.choice(cut_list)
                #print((int(real_start) - start,int(start)-real_stop))
                subseq=sequence[int(real_start) - start :int(real_start) + stop]
                if strand == '-': 
                    subseq = subseq.reverse_complement()
                if '\x00' not in str(subseq):
                    #final_promoter_list.append((str(subseq), real_start, real_stop))
                    final_promoter_list.append(str(subseq))
    

    random.shuffle(final_promoter_list) 
    print(f"final dim of promoter list: {len(final_promoter_list)}")  
    i=0
    with open(outfilename,"w")as f:
        
        for item in final_promoter_list:
            line=">"+str(i)+"\n"+item+"\n"
            f.write(line)
            i+=1




