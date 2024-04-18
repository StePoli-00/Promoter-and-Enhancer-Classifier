import json
import argparse
gtf_path="gencode.v45.annotation.gtf"
fasta_path="GRCh38.p14.genome.fa"
output_file=""


i=0
chromosomes=[]
exons_position={}



def save_exons():        
    with open(output_file+".json","w") as f:
        json.dump(exons_position,f,indent=1)


def extract_exons():
    with open(gtf_path,"r") as f:
        for k in range(5):
            line=f.readline()
        while(True):
            init_pos=""
            end_pos=""
            line=f.readline()

            data=line.split("\t")
            if data[0] not in chromosomes:
                break

            if data[2]=="transcript":
                exons=[]
                while(True):
                    line=f.readline()
                    ex_data=line.split("\t")
                    if ex_data[2]=="exon":
                        exons.append((ex_data[3],ex_data[4],ex_data[6]))
                    
                    else:
                        break
                exons_position[i]=exons
                i+=1
                
def parse_chromosome_list(chrs):
    return [ ch for ch in chrs.split(",")]
            
if __name__=="__main__":
                
    parser=argparse.ArgumentParser()
    parser.add_argument("-gr","--genome_reference",help="genome fasta file path (.fa)")
    parser.add_argument("-gtf","--genome_annotation",help="genome annotation file reference (.gtf)")
    parser.add_argument("-out","--output_file",help="output file basename")
    parser.add_argument("-ch","--chromosomes",type=parse_chromosome_list,help="list of chomosome to extract")
    
    args=parser.parse_args()
    fasta_path=args.genome_reference
    gtf_path=args.genome_annotation
    output_file=args.output_file
    chromosomes=args.chromosomes
    
    if args==None or gtf_path==None or output_file==None or chromosomes==None:
        parser.print_help()
        parser.error("Parameter Error")
    
    print(gtf_path)
    print(fasta_path)
    print(output_file)
    print(chromosomes)

    
    
    



