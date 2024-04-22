<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<h1 align="center">Promoter & Enhancer Classifier</h1>

<!-- PROJECT LOGO -->

<div align="center">
  <a href="https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier">
    <img src="images/200.gif" alt="Logo">
  </a>
  <p align="center">This Repository contains all the information, <br>code of  Promoter and Enhancer Classifier project</p>
  <a href="https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier"><strong>Explore the docs »</strong></a>
    
  
</div>

## Abstract
Promoters and enhancers are both gene regulation elements and are often involved in activating specific genes. They are crucial elements in the regulation of gene expression, playing key roles in controlling when, where, and how much of a gene is transcribed into RNA and eventually translated into protein.
 Their similarity can make it difficult to distinguish them for several reasons. Promoters and enhancers may share some similar or identical DNA sequences. <br> Both contain binding sequences for regulatory proteins, such as transcription factors, which can bind and influence gene activity. These sequences can be present in different combinations and contexts in both promoters and enhancers, making it challenging to distinguish the two elements based solely on DNA sequence. Furthermore they perform different but related functions. Promoters are generally positioned near the gene they regulate and are responsible for initiating transcription, while enhancers can be located at variable distances from the target gene and act by enhancing transcription efficiency. Some enhancers can also act as alternative promoters or vice versa.

# Table of Contents
<!-- TABLE OF CONTENTS -->

<ol style="font-size:18px;">

  <li><a href="#installation">Installation</a></li>
    
  
  <li><a href="#data-preparation">Data Preparation</a>
    <ul>
    <li>
    <a href="#common-steps">Common Steps</a>
    <ul>
    <li><a href="#promoter-preprocessing">Promoter Preprocessing</a></li>
    <li><a href="#enhancer-preprocessing">Enhancer Preprocessing</a></li>
    </ul>
    </li>
    <li><a href="#embedding-tokenizer">Embedding Tokenizer</a></li>
    <li><a href="#one-hot-encoding">One Hot Encoding</a></li>
    <li><a href="#transformer">Transfomer</a></li>
    </ul>
  </li>
  
  <li><a href="#training">Training</a></li>
  <li><a href="#inference">Inference</a></li>
  <li><a href="#view-result">View Result</a></li>
  <li><a href="#performance">Performance</a></li>
</ol>




<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- ABOUT THE PROJECT 
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<p align="right">(<a href="#readme-top">back to top</a>)</p>-->

## Installation
1. Clone the repository
   ```sh
    git clone https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier.git
   ```
2. After cloning move into the folder: 
   ```sh
    cd Promoter-and-Enhancer-Classifier/
   ```
3. Create a conda environment with python 3.11
   ```sh
   conda create -n <env> python=3.11
   ```
4. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Preparation 
Before using our mode first we must create dataset.
Input Data are:
- Token Embedding: are generated by DNABERT Tokenizer
- One Hot Encoding: simple encoding of DNA basis: A = {1,0,0,0}, T = {0,1,0,0}, C = {0,0,1,0} e G = {0,0,0,1}

Depending on which model you want to test you must process data in two distinguish manner.
In this section you will learn how to process both. 
## Common Steps
As first thing to do we must extract our data: promoters and enhancer 
#### Promoter Preprocessing
1. Download the following files: 
- genome file: [GRCh38.p14.genome.fa](https://www.gencodegenes.org/human/)
- bed file: [human_epdnew_xxxxxx.bed](https://epd.expasy.org/epd/get_promoters.php) 
2. Place where you prefer, our suggest is to use `Data` folder to store used for the model 
3. From ` Data_Preparation` run **extract_promoters.py** 

```sh
cd Data_Preparation 
python extract_promoters.py -g <genome_file> -b <bed_file> [-l <promoters length> -o <output_path> ]

where: 
-o is the desired position of the output file
example: 
python extract_promoters.py -g GRCh38.p14.genome.fa -b human_epdnew_VgGtt.bed -l 100 -o Desktop/folder
```
> [!NOTE] 
> By omitting `-l` parameter will cut promoter sequence of  variable length: 5,10,20,100,200,1000,2000.<br>
Except for the Transfomer, all the other model take an input sequence of fixed length. <br>
In your case this `-l` parameter is **mandatory**
####  Enhancer Preprocessing
**First Enhancers Preprocessing**

For what concern the first enhancers preprocessing is required: 
1. Go to https://bio.liclab.net/ENdb/Download.php
2. Download file signed by "All the experimentally confirmed enhancers"


Now you need the execution of *first_enhancer_preprocessing.py*. 

This file take `enhancer_file`, remove the **mm10 genes** (that is genes of mouse) and generate a file in a hg19 format (necessary for the next step of preprocessing). 

Into terminal run: 
```sh
 python3 first_enhancer_preprocessing.py -e <enhancer_file.txt> -hg19_list <name_of_file_for_hg19_list>.txt -mm10_list <name_of_mm10_file>.txt
```
> [!NOTE]
> This command generates a file called `name_of_file_for_hg19_list.txt` that you will use for the second step of enhancers preprocessing. 

**Second Enhancers Preprocessing**

Using https://genome.ucsc.edu/cgi-bin/hgLiftOver, you upload on this web tool the 
`name_of_file_for_hg19_list.txt` (obtained in the previous step) and obtain a .BED file that is a list of all enhancers converted into hg38 format.
This file contains enhancers under the format "chr_name:<start_position> - <end_position>".

1. Into terminal run: 
```sh
 pip3 install biopython
```

The **enhancer_preprocessing.py** takes different mandatories arguments from command line: 
1) option `-b`: path bed file (hg38) 
2) option `-f`: path fasta file (GRCh38.FASTA 3gb) 
3) option `-e`: name of output file (.txt) for the final sequences.


Initially, the script open the .BED file and, using a Gene class (name, start, end) generate an iterative list of Gene.<br>
Moreover, the script open `GRCh38.FASTA` file and with FastaElem class (name, sequence) generate a list of FastaElem. <br> 
With Gene list and FastaElem list, using BioPython library, the script for each Gene into FastaElem (using the specific start,end position), takes the relative sequence of nucleotides and save it on `enhancer.txt` output file.
The enhancer.txt file, at the end, contains all the enhancer with the lenght specify in the initial .BED file in hg38 version. 

```sh
python enhancer_preprocessing.py -b <path_bed_file> -f <path_fasta_file> -e <output_name_enhancer_file>
```


#### Create csv dataset
It contains promoter and enhancer sequence. In the previous folder run **create_csv_dataset.py**
```sh
python create_csv_dataset.py -p <promoter_file> -e <enhancer_file> -o <output_file> -l <promoter_length>

example:
python create_csv_dataset.py -p promoters_100.fa -e enhancers.txt -o dataset.csv -l 100
```
### Embedding Tokenizer

> [!WARNING] 
> The following file require to be execute with GPU
1. make sure you have installed einops package, otherwise install it:
   ```sh
   pip install einops
   ```
2. Fed csv file into **create_embedding.py**
   ```sh
   python -s <dataset.csv> -d <destination>

   example:
   python -s data.csv -d Project_Folder/
   ```
>[!NOTE] 
>- This script  will create `Embedding` folder in the destination path provided  which contains `embeddings` and `labels` of our data converted from sequence to embeddings using Tokenizer of DNABert. 
> - Before to run the script make sure that destination folder doesn't have any folder named `Embedding` because the script will stop. This behaviour is implemented in order to avoid overriding data from previous execution of the code. 
> - if you have encounterd this type of error: `assert q.is_cuda and k.is_cuda and v.is_cuda` try to unistall triton package




3. run **split_dataset.py** to  create Dataset folder, run the following command: 
   ```sh
   python split_dataset.py -e <embedding_folder_path> -d <dataset_folder_path> [ -z <embedding_zip_path>]  
   example:
   python split_dataset.py -e Project/Embedding -d Dataset   
   ```
> [!NOTE]
> - If you want to use -z option <embedding_zip_path> must have the same name for  embedding_folder_path
> - -z: can be used only if the embedding zip folder is only one.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### One Hot Encoding
After the common steps run this command to generate OHE Dataset
```sh
python create_datasetOHE.py -s <source_path> -d <destionation_path> -l <sequence_length>

where:
- s: is the csv file obtained at the end of common steps
- d: destionation of Dataset folder
- l: length of the sequence

example: python create_datasetOHE.py -s ..\Data\dataset100.csv -d . -l 100
```
This script will create dataset folder named `Dataset_x` where x is the length passed by -l option.

### Transformer 
In order to use the Transfomer model the steps are:
1. run **extract_promoters.py** without  `-l` option:
```sh
example: 
python extract_promoters.py -g GRCh38.p14.genome.fa -b human_epdnew_VgGtt.bed -o Desktop/folder
```
the output file will be named as `promoters_mixed.fa` <br>

2. run **enhancer_preprocessing.py** will cut enhancer to `maximum sequence length` equal to 1000 because in literature enhancer and promoters have a variabile length of 100-1000 bp.
```sh
python enhancer_preprocessing.py
```
3. run **create_csv_dataset.py** with `-l` equal to 1000 (e.g. `maximum sequence length`) 
```sh
example: 
python create_csv_dataset.py -p promoters_mixed.fa -e enhancers.txt -o dataset.csv -l 1000
```




<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Inference
To test Transformer model on inference mode:
1. extract exons from reference genome annotation file (.gtf)
   ```sh
   python exons_extractor.py -gtf <genome_annotation_file> -out <output_file> -ch <list of crhomosome to extract >
   example:
   python exons_extractor.py gencode.v45.annotation.gtf -out exons.json  -ch chr1,chr2,chr3
   ```
2. extract introns from reference genome (.fa)
   ```sh
   
   ```
4. pass the introns sequence to the model 
    ```sh
   
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## View Result
Each training produce an output folder that contains checkpoint and "events.out" files. 
1. Into the terminal run: 
   ```sh
    pip3 install tensorboard
   ```
2. After the installation move into result folder
   ```sh 
   cd lightning_logs
   ```
3. Now, run command:
   ```sh
    tensorboard --logdir <name_of_version_folder> 
   ```
4. This command show graphics about training, test and valiadation accuracy and loss.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Performance
The following table report interval of confidence of our models


|<p align="center"> Models   </p> |<p align="center"> Accuracy  </p>   | <p align="center"> Loss  </p>  | <p align="center">  Specificity </p>  | <p align="center"> Sensitivity </p> |<p align="center"> F1 score </p> | <p align="center"> MCC</p>  | <p align="center"> AUC</p> |
| --- | -------- | ---   | -----    |  ---    | -----|           --- |  -----    | 
| CNN 1D with Embedding| 0.905 ± 0.007|0.454± 0.043|0.907± 0.027|0.905± 0.030|0.902± 0.006|0.827± 0.029|0.968± 0.013| 
|CNN 1D with One Hot Encoding| 0.888 ± 0.01|0.884 ± 0.029|0.905± 0.024|0.890± 0.009|0.815 ± 0.273 |0.772 ± 0.029|0.947 ± 0.011| 
|Transformer BERT|0.668 ± 0.133|0.562 ± 0.019|0.346 ± 0.070|0.982 ± 0.078|0.501 ± 0.095|0.448 ± 0.068|0.674 ± 0.035| 
|Transformer DNABERT2|0.961 ± 0.005|0.133 ± 0.009|0.944 ± 0.006|0.979 ± 0.001 |0.960 ± 0.001| 0.923 ± 0.012|0.987 ± 0.012| 
|Transformer InstaDeep| 0.958 ± 0.005|0.158 ± 0.007|0.946 ± 0.016|0.968 ± 0.007|0.956 ± 0.023|0.916  ± 0.005|0.985 ± 0.001|





<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contacts
* Antonio De Blasi - [AntonioDeBlasi-Git](https://github.com/AntonioDeBlasi-Git)
* Francesco Zampirollo - [zampifre](https://github.com/zampifre) 
* Stefano Politanò - [StePoli-00](https://github.com/StePoli-00) 



<!-- ## Project Link
[https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier](https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
