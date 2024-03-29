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

<h1 align="center">Promoter & Enhancer Classsifier</h1>

<!-- PROJECT LOGO -->

<div align="center">
  <a href="https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier">
    <img src="images/dna.svg" alt="Logo" width="200" height="200">
  </a>
  <p align="center">This Repository contains code for classify Promoteres and Enhancer</p>
  <a href="https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier"><strong>Explore the docs »</strong></a>
    
  
</div>


# Table of Contents
<!-- TABLE OF CONTENTS -->

<ol style="font-size:18px;">
  <li>
    <a href="#abstract">Abstract</a>
  <li><a href="#installation">Installation</a></li>
    
  
  <li><a href="#data-preparation">Data Preparation</a>
    <ul>
    <li><a href="#embedding-tokenizer">Embedding Tokenizer</a></li>
    <li><a href="#one-hot-encoding">One Hot Encoding</a></li>
    </ul>
  </li>
  
  <li><a href="#training">Training</a></li>
  <li><a href="#testing">Testing</a></li>
  <li><a href="#resut">Results</a></li>
</ol>


## Abstract
<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- ABOUT THE PROJECT 
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>-->

## Installation
1. Clone the repository
   ```sh
   $ git clone https://github.com/StePoli-00/Promoter-and-Enhancer-Classifier.git
   ```
2. After cloning the repository run: 
   ```sh
   $ cd Promoter-and-Enhancer-Classifier/
   ```
3. Create a conda environment by requirements.txt 
   ```sh
   $ conda create --name <env> --file requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Preparation 
Before using our mode first we must create dataset.
Input Data are:
- Token Embedding: are generated by DNABert2 Tokenizer
- One Hot Encoding: simple encoding of DNA basis: A = {1,0,0,0}, T = {0,1,0,0}, C = {0,0,1,0} e G = {0,0,0,1}

Depending on which model you want to test you must process data in two distinguish manner.
In this section you will learn how to process both.





<!-- USAGE EXAMPLES -->
### Embedding Tokenizer
1. Download the following files: 
- genome file: [GRCh38.p14.genome.fa](https://www.gencodegenes.org/human/)
- bed file: [human_epdnew_xxxxxx.bed](https://epd.expasy.org/epd/get_promoters.php) 
2. Place where you prefer, our suggest is to use `Data` folder to store used for the model 
3. From `cd Data_Preparation 
` run extract_promoters.py 
```sh
cd Data_Preparation 
$ python extract_promoters.py -g <genome_file> -b <bed_file> [-l <promoters length> -o <output_path> ]

where: 
-o is the desired position of the output file
example: 
python extract_promoters.py -g GRCh38.p14.genome.fa -b human_epdnew_xxxxxx.bed -l 100 -o Desktop/folder
```
> Since this script is used to generate dataset with sequence of variabile length for the Transfomer, by omitting -l parameter will cut promoter sequence of length: 5,10,20,100,200,1000,2000.<br>
In your case this paramter is **mandatory**


In `cd Data_Preparation 
` folder run `create_dataset.py` with the following command 
```sh
$ python create_dataset.py -e <embedding_folder_path> -d <dataset_folder-path> [ -z <embedding_zip_path>]  

Where 
-z: can be used only if the embedding zip folder is only one.

```
> the zip folder must have the same name of embedding_folder_path
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### One Hot Encoding
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Testing
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contacts
* Antonio De Blasi - [AntonioDeBlasi-Git](https://github.com/AntonioDeBlasi-Git)
* Francesco Zampirollo - [zampifre](https://github.com/zampifre) 
* Stefano Politanò - [StePoli-00](https://github.com/StePoli-00) 

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





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
