import re
import argparse
from Bio import SeqIO

bed_hg38_file = "" # path per gene + indici
fasta_file = ""
enhancer_file = ""

class gene:
    def __init__(self, nome, ind_start, ind_end):
        self.nome = nome
        self.ind_start = ind_start
        self.ind_end = ind_end
    def __str__(self):
        return "Gene: "+ str(self.nome) +" start index: "+ str(self.ind_start) +" end index: "+ str(self.ind_end)+"\n"

class fasta_elem:
    def __init__(self, nome, sequence):
        self.nome = nome
        self.sequence = sequence
    def __str__(self):
        return "Gene: "+ str(self.nome) +" sequence: "+ str(self.sequence)+"\n"


def extract_gene_hg38_list(bed_hg38_file):
    """
    Recuperiamo dal file .bed (formato chrX:start-end) i nomi dei geni e gli indici di inzio e fine degli enhancer da recuperare
    """
    my_list = []
    with open(bed_hg38_file,"r") as file_bed:
        for line in file_bed:
            match = re.match(r'^(\w+):(\d+)-(\d+)$', line)
            if match:
                sequence_id = match.group(1)
                sequence_start = int(match.group(2))
                sequence_end = int(match.group(3))
            tmp = gene(sequence_id, sequence_start, sequence_end)
            my_list.append(tmp)
        
    return my_list
    

def extract_fasta_gene_list(fasta_file):
    fasta_list = []
    # Apri il file .fa e leggi le sequenze
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            id = record.id
            sequence = record.seq
            tmp = fasta_elem(id, sequence)
            fasta_list.append(tmp)

    """a noi servono esclusivamento i geni fino al 22"""
    fasta_list = fasta_list[0:22]
    
    return fasta_list


def obtain_correct_gene(enhancer_file, hg38_list, fasta_list): 
    """
    recupero delle sequence degli enhancer e le salvo nel file enhancer.txt
    """
    with open(enhancer_file, "w") as file_enhancer:
        for item in hg38_list:
            i = item.nome
            start_index = item.ind_start
            end_index = item.ind_end
            diff = end_index - start_index
            for elem in fasta_list:
                if(elem.nome == i):
                    seq = elem.sequence[start_index:end_index]
                    file_enhancer.write(elem.nome + "\t"+ str(seq) +"\n")
                    print(str(diff) +"\t"+ str(len(seq)) + "\n")
                    
                
if __name__=="__main__":
    parser=argparse.ArgumentParser
    parser.add_argument("-b","-bed", help="bed file hg38(.bed)")
    parser.add_argument("-f","-fasta",help="fasta file (.fa)")
    parser.add_argument("-e","-enhancer", help="enhancer OUTPUT file (.txt)")
    args=parser.parse_args()
    
    bed_hg38_file = parser.bed
    fasta_file = parser.fasta
    enhancer_output_file = parser.enhancer

    if bed_hg38_file is None or fasta_file is None or enhancer_file is None:
        parser.print_help()
        parser.error("Parameters error")
        
    enhancer_hg38_list = extract_gene_hg38_list(bed_hg38_file)
    fasta_gene_list = extract_fasta_gene_list(fasta_file)
    obtain_correct_gene(enhancer_output_file, enhancer_hg38_list, fasta_gene_list)
    