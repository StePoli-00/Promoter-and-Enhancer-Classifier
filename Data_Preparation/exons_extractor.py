import json
import argparse
gtf_path=""
output_file=""


i=0
chromosomes=[]



def save_exons(data):        
    with open(output_file+".json","w") as f:
        json.dump(data,f,indent=2)


def extract_exons():
    exons=[]
    exons_position={}
    prev=""
    curr=""
    with open(gtf_path,"r") as f:
        #skip header
        for k in range(5):
            line=f.readline()
        while(True):
           
            line=f.readline()
            data=line.split("\t")
            if data[0] not in chromosomes:
                exons_position[prev]=exons
                break

            curr=data[0]
            if prev=="":
                prev=curr
            elif prev!=curr:
                exons_position[prev]=exons
                exons.clear()

            if data[2]=="transcript":
                
                while(True):
                    line=f.readline()
                    ex_data=line.split("\t")
                    if ex_data[2]=="exon":
                        exons.append((ex_data[3],ex_data[4],ex_data[6]))
                    
                    else:
                        break
            prev=curr
    return exons_position
                
def parse_chromosome_list(chrs):
    return [ ch for ch in chrs.split(",")]
            
if __name__=="__main__":
                
    parser=argparse.ArgumentParser()
    parser.add_argument("-gtf","--genome_annotation",help="genome annotation file reference (.gtf)")
    parser.add_argument("-out","--output_file",help="output file basename")
    parser.add_argument("-ch","--chromosomes",type=parse_chromosome_list,help="list of chomosome to extract")
    
    args=parser.parse_args()
    gtf_path=args.genome_annotation
    output_file=args.output_file
    chromosomes=args.chromosomes
    
    if args==None or gtf_path==None or output_file==None or chromosomes==None:
        parser.print_help()
        parser.error("Parameter Error")
    
    data=extract_exons()
    save_exons(data)

    
    
    



