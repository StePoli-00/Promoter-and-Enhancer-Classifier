import re
import re
import argparse


def execute_first_preprocessing(enhancer_file, list_hg19_file, list_mm10_file): 
    my_list = []
    final = []
    final_mm10 = []
    
    with open(enhancer_file, 'r') as file:
        # Leggi il contenuto del file
        # Scorri le linee del file
        for line in file:
            # Rimuovi eventuali spazi bianchi o caratteri di nuova linea
            line = line.strip()

            # Controlla se la linea inizia con 'E_'
            if line.startswith('E_'):
                my_list.append(line)
            else:
                pass


    for item in my_list:
        index = item.find('hg19')
        if index != -1:
                    # Taglia la parte della stringa fino a 'chr'
                    substr_chr = item[index:]
                    parts = substr_chr.split('\t')
                    if len(parts) >= 3:
                        # Costruisci la sottostringa finale
                        final_substring = parts[1] + ':' + parts[2] + '-' + parts[3]
                        final.append(final_substring)


        with open(list_hg19_file, 'w') as file:
            # Scrivi ogni elemento della lista nel file
            for item in final:
                file.write(item + '\n')
            

    for item in my_list:
        index = item.find('mm10')
        if index != -1:
                    # Taglia la parte della stringa fino a 'chr'
                    substr_chr = item[index:]
                    parts = substr_chr.split('\t')
                    if len(parts) >= 3:
                        # Costruisci la sottostringa finale
                        final_substring = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' + parts[3]
                        final_mm10.append(final_substring)


        with open(list_mm10_file, 'w') as file:
            # Scrivi ogni elemento della lista nel file
            for item in final_mm10:
                file.write(item + '\n')
            
if __name__=="__main__":
    parser = argparse.ArgumentParser
    parser.add_argument("-e","-enhancer_file", help="name of enhancer file (<name>.txt)")
    parser.add_argument("-i19","-hg19_list",help="name of lista_indici_hg19 (<name>.txt)")
    parser.add_argument("-imm10","-mm10_list", help="name of lista_indici_mm10 (<name>.txt)")
    args=parser.parse_args()
    
    enhancer_file = parser.enhancer_file
    list_hg19_file = parser.hg19_list
    list_mm10_file = parser.mm10_list

    if enhancer_file is None or list_hg19_file is None or list_mm10_file is None:
        parser.print_help()
        parser.error("Parameters error")
        
    execute_first_preprocessing(enhancer_file, list_hg19_file, list_mm10_file)
    