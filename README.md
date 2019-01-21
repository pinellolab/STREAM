[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/stream/README.html)

[![Build Status](https://travis-ci.org/pinellolab/STREAM.svg?branch=master)](https://travis-ci.org/pinellolab/STREAM)

# STREAM

STREAM (**S**ingle-cell **T**rajectories **R**econstruction, **E**xploration **A**nd **M**apping) is an interactive pipeline capable of disentangling and visualizing complex branching trajectories from both single-cell transcriptomic and epigenomic data.

<img src="https://github.com/pinellolab/STREAM/blob/stream_python2/STREAM/static/images/Figure1.png">

STREAM is available as user-friendly open source software and can be used interactively as a web-application at [stream.pinellolab.org](http://stream.pinellolab.org/), as a bioconda package [https://bioconda.github.io/recipes/stream/README.html](https://bioconda.github.io/recipes/stream/README.html) and as a standalone command-line tool with Docker [https://github.com/pinellolab/STREAM](https://github.com/pinellolab/STREAM)

Installation with Bioconda
--------------------------

1)	If Anaconda (or miniconda) is already installed with **Python 3**, skip to 2) otherwise please download and install Python3 Anaconda from here: https://www.anaconda.com/download/

2)	Open a terminal and add the Bioconda channel with the following commands:

```sh
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

3)	Recommended: Create an environment named ‘myenv’ and activate it with the following commands:

```sh
$ conda create -n myenv python=3.6
$ conda activate myenv
```

4)	Install the bioconda STREAM package within the environment ‘myenv’ with the following command:

```sh
$ conda install stream
```

Tutorial
--------

* Example for scRNA-seq: [1.STREAM_scRNA-seq.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.STREAM_scRNA-seq.ipynb)

* Example for *mapping* feature: [2.STREAM_mapping.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.STREAM_mapping.ipynb)

* Example for complex trajectories: [3.STREAM_complex_trajectories.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/3.STREAM_complex_trajectories.ipynb)

* Example for scATAC-seq: [4.STREAM_scATAC-seq.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/4.STREAM_scATAC-seq.ipynb)


Installation with Docker
------------------------

With Docker no installation is required, the only dependence is Docker itself. Users will completely get rid of all the installation and configuration issues. Docker will do all the dirty work for you!

Docker can be downloaded freely from here: [https://store.docker.com/search?offering=community&type=edition](https://store.docker.com/search?offering=community&type=edition)

To get an image of STREAM, simply execute the following command:

```sh
$ docker pull pinellolab/stream
```


Basic usage of *docker run* 

```sh
$ docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

OPTIONS:  
```
--publish , -p	Publish a container’s port(s) to the host  
--volume , -v	Bind mount a volume  
--workdir , -w	Working directory inside the container  
```

STREAM interactive website
--------------------------

In order to make STREAM user friendly and accessible to non-bioinformatician, we have created an interactive website: [http://stream.pinellolab.org](http://stream.pinellolab.org) The website implements all the features of the command line version and in addition provides interactive and exploratory panels to zoom and visualize single-cells on any given branch.

The website offers two functions: 1) To run STREAM on single-cell transcriptomic or epigenomic data provided by the users. 2) The first interactive database of precomputed trajectories with results for seven published datasets. The users can visualize and explore cells’ developmental trajectories, subpopulations and their gene expression patterns at single-cell level. 

The website can also run on a local machine using the provided Docker image we have created. To run the website in a local machine after the Docker installation, from the command line execute the following command:
```sh
$ docker run -p 10001:10001 pinellolab/stream STREAM_webapp
```

After the execution of the command the user will have a local instance of the website accessible at the URL: 
[http://localhost:10001](http://localhost:10001)


STREAM command line interface
-----------------------------

To run STREAM at the command-line interface:

* start a terminal session;

For **Mac OS**:
* enter ```docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM --help [options]```

For **Windows**:
* enter ```docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM --help [options]```

Users can specify the following options:
```
-m, --matrix  
input file name. Matrix is in .tsv or tsv.gz format in which each row represents a unique gene and each column is one cell. (default: None)
-l, --cell_labels  
file name of cell labels (default: None)
-c, --cell_labels_colors  
file name of cell label colors (default: None)
-s, --select_features  
LOESS, PCA, all: Select variable genes using LOESS or top principal components using PCA or keep all the gene (default: LOESS)
-f, --feature_genes  
specified feature genes (default: None)
-t, --detect_TG_genes  
detect transition genes automatically
-d, --detect_DE_genes  
detect DE genes automatically
-g, --gene_list  
genes to visualize, it can either be filename which contains all the genes in one column or a set of gene names separated by comma (default: None)
-p, --use_precomputed  
use precomputed data files without re-computing structure learning part
--log2  
perform log2 transformation
--norm  
normalize data based on library size
--atac
indicate scATAC-seq data
--atac_counts
scATAC-seq counts file name in .tsv or .tsv.gz format. Counts file is a compressed sparse matrix that contains three columns including region indices, sample indices and the number of reads(default: None)
--atac_regions
scATAC-seq regions file name in .tsv or .tsv.gz format. Regions file contains three columns including chromosome names, start and end positions of regions (default: None)
--atac_samples
scATAC-seq samples file name in .tsv or tsv.gz. Samples file contains one column of cell names  (default: None)
--atac_k
specify k-mers length for scATAC-seq analysis (default: 7)
--n_processes  
Specify the number of processes to use. (default, all the available cores).
--loess_frac  
The fraction of the data used in LOESS regression (default: 0.1)
--pca_max_PC  
Maximal principal components in PCA (default: 100)
--pca_first_PC  
keep first PC
--pca_n_PC  
The number of selected PCs (default: 15)
--n_processes  
Specify the number of processes to use. The default uses all the cores available
--lle_neighbours  
LLE neighbour percent (default: 0.1)
--lle_components  
number of components for LLE space (default: 3)
--AP_damping_factor  
Affinity Propagation: damping factor (default: 0.75)
--EPG_n_nodes
Number of nodes for elastic principal graph (default: 50)
--EPG_lambda
lambda parameter used to compute the elastic energy (default: 0.02)
--EPG_mu
mu parameter used to compute the elastic energy (default: 0.1)
--EPG_trimmingradius
maximal distance of point from a node to affect its embedment (default: Inf)
--EPG_finalenergy
indicating the final elastic energy associated with the configuration. It can be 'Base' or 'Penalized' (default: 'Penalized')
--EPG_alpha
positive numeric, alpha parameter of the penalized elastic energy (default: 0.02)
--disable_EPG_collapse
disable collapsing small branches
--EPG_collapse_mode
the mode used to collapse branches. It can be 'PointNumber','PointNumber_Extrema', 'PointNumber_Leaves','EdgesNumber' or 'EdgesLength' (default:'PointNumber')
--EPG_collapse_par
the control parameter used for collapsing small branches
--EPG_shift
shift branching point 
--EPG_shift_mode
the mode to use to shift the branching points 'NodePoints' or 'NodeDensity' (default: NodeDensity)
--EPG_shift_DR
positive numeric, the radius used when computing point density if EPG_shift_mode is 'NodeDensity' (default:0.05)
--EPG_shift_maxshift
positive integer, the maximum distance (number of edges) to consider when exploring the branching point neighborhood (default:5)
--disable_EPG_ext
disable extending leaves with additional nodes
--EPG_ext_mode
the mode used to extend the graph. It can be 'QuantDists', 'QuantCentroid' or 'WeigthedCentroid'. (default: QuantDists)
--EPG_ext_par
the control parameter used for contribution of the different data points when extending leaves with nodes (default: 0.5)
--DE_z_score_cutoff  
Differentially Expressed Genes Z-score cutoff (default: 2)
--DE_diff_cutoff  
Differentially Expressed Genes difference cutoff (default: 0.2)
--TG_spearman_cutoff  
Transition Genes Spearman correlation cutoff (default: 0.4)
--TG_diff_cutoff  
Transition Genes difference cutoff (default: 0.2)
--stream_log_view
use log2 scale for y axis of stream_plot 
--for_web
Output files for website
-o, --output_folder  
Output folder (default: None)
--new  
file name of data to be mapped (default: None)
--new_l  
filename of new cell labels (default: None)
--new_c  
filename of new cell label colors (default: None)
```



Input file format
-----------------
#### **Transcriptomic data**

The main and required input file is a tab-separated gene expression matrix (raw counts or normalized values) in tsv file format. Each row represents a unique gene and each column is one cell.


For example, in python
```R
>import pandas as pd
>input_data = pd.read_csv('data_Guo.tsv.gz',sep='\t',header=0,index_col=0)
>input_data.iloc[0:5,0:5]
```

|        | HSC1      | HSC1.1    | HSC1.2    | HSC1.3    | HSC1.4   |
|--------|-----------|-----------|-----------|-----------|----------|
| CD52   | 6.479620  | 0.000000  | 0.000000  | 5.550051  | 0.000000 |
| Ifitm1 | 11.688533 | 11.390682 | 10.561844 | 11.874295 | 8.976571 |
| Cdkn3  | 0.000000  | 0.000000  | 0.000000  | 0.000000  | 8.293616 |
| Ly6a   | 10.417026 | 11.452145 | 0.000000  | 8.158840  | 8.945882 |
| Bax    | 6.911608  | 10.201157 | 0.000000  | 9.396073  | 0.000000 |

In addition, it is possible to provide these optional files in .tsv or .tsv.gz format: 

**cell_labels** file: .tsv or .tsv.gz format. Each item can be a putative cell type or sampling time point obtained from experiments. Cell labels are helpful for visually validating the inferred trajectory. The order of labels should be consistent with cell order in the gene expression matrix file. No header is necessary:

|        |
|--------|
| HSC    | 
| HSC    | 
| GMP    |
| MEP    |
| MEP    |
| GMP    |

**cell_label_color** file: .tsv or .tsv.gz format. Customized colors to use for the different cell labels. The first column specifies cell labels and the second column specifies the color in the format of hex. No header is necessary:

|       |         | 
|-------|---------|
| HSC   | #7DD2D9 | 
| MPP   | #FFA500 |
| CMP   | #e55b54 |
| GMP   | #5dab5a |
| MEP   | #166FD5 |
| CLP   | #989797 |

**gene_list** file: .tsv or .tsv.gz format. It contains genes that users may be interested in visualizing in subway map and stream plot. Genes are listed in one column. No header is necessary: 

|        |
|--------|
| Ifitm1 | 
| Cdkn3  | 
| Ly6a   |
| CD52   |
| Foxo1  |
| GMP    |

**feature_genes** file: .tsv or .tsv.gz format. It contains genes that the user can specify and that are used as features to infer trajectories. instead of using the automatic feature selection of STREAM. No header is necessary:

|        |
|--------|
| Gata1  | 
| Pax5   | 
| CD63   |
| Klf1   |
| Lmo2   |
| GMP    |


#### **Epigenomic data**
To perform scATAC-seq trajectory inference analysis, the main input can be:   

1)a set of files including **count file**, **region file** and **sample file**. 

**count file**, .tsv or .tsv.gz format. A tab-delimited compressed matrix in sparse format (column-oriented). It contains three columns. The first column specifies the rows indices (the regions) for non-zero entry. The second column specifies the columns indices (the sample) for non-zero entry. The last column contains the number of reads in a given region for a particular cell. No header is necessary:

|        |     |  |
|--------|-----|--|
| 3735   | 96  | 1|
| 432739 | 171 | 2|
| 133126 | 292 | 1|
| 219297 | 359 | 1|
| 284936 | 1222| 1|
| 442588 | 1580| 2|

**region file**, .bed or .bed.gz format. A tab-delimited .bed file with three columns. The first column specifies chromosome names. The second column specifies the start position of the region. The third column specifies the end position of the region. The order of regions should be consistent with the regions indices in the count file. No header is necessary:

|      |       |      |
|------|-------|------|
| chr1 | 10279 | 10779|
| chr1 | 13252 | 13752|
| chr1 | 16019 | 16519|
| chr1 | 29026 | 29526|
| chr1 | 96364 | 96864|

**sample file**, .tsv or .tsv.gz format. It has one column. Each row is a cell name.  The order of the cells should be consistent with the sample indices in count file. No header is necessary:

|                                    |
|------------------------------------|
| singles-BM0828-HSC-fresh-151027-1  | 
| singles-BM0828-HSC-fresh-151027-2  | 
| singles-BM0828-HSC-fresh-151027-3  |
| singles-BM0828-HSC-fresh-151027-4  |
| singles-BM0828-HSC-fresh-151027-5  |


2)a precomputed scaled z-score file by STREAM. Each row represents a k-mer DNA sequence. Each column represents one cell. Each entry is a scaled z-score of the accessibility of each k-mer across cells

|        | singles-BM0828-HSC-fresh-151027-1 | singles-BM0828-HSC-fresh-151027-2 | singles-BM0828-HSC-fresh-151027-3 |
|--------|-----------------------------------|-----------------------------------|-----------------------------------|
| AAAAAAA|-0.15973157637808505               | 0.18950966450007853               | 0.07713107176524692               | 
| AAAAAAG|-1.3630723054479532                | -0.04770034004421244              | 0.6387323857481045                |
| AAACACG|-0.2065161126378667                | -1.3375384076872765               | 0.2660278729402342                |
| AGCGTTA|-0.496859947462221                 | 0.7181918229050274                | 0.19603357892921522               |
| ATACTCA|-1.2127919166377426                | 0.7938414496478844                | -1.2665513250104594               |



Example
--------

All the datasets used in the following examples can be found under the directory **./Datasets**

To download the datasets, 

```
$ git clone https://github.com/pinellolab/STREAM.git
$ cd STREAM/Datasets/
```

Or they can also be downloaded using the following Dropbox link: 
[https://www.dropbox.com/sh/xnw9ro22bgrz2pa/AADQWmyCjUekg3hudvhsrAWka?dl=0](https://www.dropbox.com/sh/xnw9ro22bgrz2pa/AADQWmyCjUekg3hudvhsrAWka?dl=0)


Please note that for large dataset analysis it'll be necessary to increase the default allocated memory of container. Especially, for scACTA-seq analysis based on **counts_file**,

<img src="https://github.com/pinellolab/STREAM/blob/stream_python2/STREAM/static/images/docker.png" width="50%">

### **Transcriptomic data**

Using the example data provided: data_Guo.tsv.gz, ell_label.tsv.gz and cell_label_color.tsv.gz, and assuming that **they are in the current folder**, to perform trajectories analysis, users can simply run a single command (By default, LOESS is used to select most variable gene. For qPCR data, the number of genes is relatively small and often preselected, it this case it may be necessary to keep all the genes as features by setting the flag -s all):

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all
```

To visualize genes of interest, user can provide a gene list file, for example: gene_list.tsv and add the flag  -p to use the precomputed file obtained from the first running (in this way, the analysis can will not restart from the beginning and other existing figures will not be re-generated):

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -g gene_list.tsv.gz -p
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -g gene_list.tsv.gz -p
```

Users can also provide a set of gene names separated by comma:

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -g Gata1,Pax5 -p
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -g Gata1,Pax5 -p
```

To explore potential marker genes, it is possible to add the flags -d or -t to detect DE (differentially expressed) genes and transition gens respectively. The top 10 DE (any pair of branches) and transition genes (any branch) are automatically plotted:

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -d -t
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Guo.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz -s all -d -t
```

### **Mapping**

To explore the feature **mapping**, users need to provide two dataset, one is used for inferring trajectories. The other is the dataset that is going to be mapped to the inferred trajectories. Here we take data_Moore_qPCR_WT.tsv.gz, data_mapping.tsv (Moore, F.E. et al.2016) as an example. We assume that **all the datasets are in the current folder**.

Users first need to run the following command to get initial inferred trajetories:

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Moore_qPCR_WT.tsv.gz -s all --EPG_shift --EPG_trimmingradius 0.1 -o STREAM_result
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Moore_qPCR_WT.tsv.gz -s all --EPG_shift --EPG_trimmingradius 0.1 -o STREAM_result
```

To map the labelled cells to the inferred trajectories, users need to specify the same output direcotry by executing the following command:

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -o STREAM_result --new data_mapping.tsv.gz --new_l cell_labels_mapping.tsv.gz --new_c cell_labels_mapping_color.tsv.gz 
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -o STREAM_result --new data_mapping.tsv.gz --new_l cell_labels_mapping.tsv.gz --new_c cell_labels_mapping_color.tsv.gz 
```
After running this command,  a folder named **'Mapping_Result'** will be created under **'/users_path/STREAM_result'** along with all the mapping analysis results.


### **scATAC-seq data**

To perform scATAC-seq trajectory inference analysis, three files are necessary, a .tsv file of counts in compressed sparse format, a sample file in .tsv format and a region file in .bed format. We assume that **they are in the current folder**.

Using these three files, users can run STREAM with the following command (note the flag **--atac** ):

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM --atac -s PCA --atac_counts count_file.tsv.gz --atac_samples sample_file.tsv.gz --atac_regions region_file.bed.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM --atac -s PCA --atac_counts count_file.tsv.gz --atac_samples sample_file.tsv.gz --atac_regions region_file.bed.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz
```

This command will generate a file named **df_zscores_scaled.tsv**. It’s a tab-delimited z-score matrix with k-mers in row and cells in column. Each entry is a scaled z-score of the accessibility of each k-mer across cells. This operation is time consuming and it may take a couple of hours with a modest machine. STREAM also provides the option to take as input a precomputed z-score file from the previous step, for example to recover trajectories when increasing the dimensionality of the manifold. Using a precomputed z-score file, users can run STREAM with the following command:

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m df_zscores_scaled.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz --atac -s PCA
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m df_zscores_scaled.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz --atac -s PCA
```
or 

For **Mac OS**:
```sh
$ docker run  -v $PWD:/data -w /data  pinellolab/stream STREAM -m data_Buenrostro_7mer_scaled.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz --atac -s PCA
```
For **Windows**:
```sh
$ docker run  -v ${pwd}:/data -w /data  pinellolab/stream STREAM -m data_Buenrostro_7mer_scaled.tsv.gz -l cell_label.tsv.gz -c cell_label_color.tsv.gz --atac -s PCA
```

Here the file **'data_Buenrostro_7mer_scaled.tsv.gz'** is the same as **'df_zscores_scaled.tsv'**.


Output description
------------------

STREAM write all the results by default in the folder STREAM_results, unless a different directory is specified by the user with the flag -o. This folder contains the following files and directories:

*   **LLE.pdf**: projected cells in the MLLE 3D space.
*   **EPG.pdf**: elastic principal graph fitted by ElPiGraph in 3D space
*   **flat_tree.pdf**: 2D single-cell level flat tree plot 
*   **nodes.tsv**: positions of nodes (or states) in the flat_tree plot
*   **edges.tsv**: edges information in the flat_tree plot
*   **cell_info.tsv**: Cell information file. Column 'CELL_ID', the cell names in the input file. Column 'Branch', the branch id a cell is assigned to. The branch id is encoded by the two cell states. Column 'lam',  the location on a branch, which is the arc length from the first cell state of branch id to the projection of the cell on that branch. Column 'dist', the euclidian distance between the cell and its projection on the branch.
*   sub-folder **'Transition_Genes'** contains several files, one for each branch id, for example for (S1,S2):
    - **Transition_Genes_S1_S2.png**: Detected transition genes plot for branch S1_S2. Orange bars are genes whose expression values increase from state S1 to S2 and green bars are genes whose expression values decrease from S1 to S2
    - **Transition_Genes_S1_S2.tsv**: Table that stores information of detected transition genes for branch S1_S2.
*   sub-folder **'DE_Genes'** contains several files, one for each pair of branches, for example for (S1,S2) and (S3,S2):
    - **DE_genes_S1_S2 and S3_S2.png**: Detected differentially expressed top 15 genes plot. Red bars are genes that have higher gene expression in branch S1_S2, blue bars are genes that have higher gene expression in branch S3_S2
    - **DE_up_genes_S1_S2 and S3_S2.tsv**: Table that stores information of DE genes that have higher expression in branch S1_S2.
    - **DE_down_genes_S1_S2 and S3_S2.tsv**: Table that stores information of DE genes that have higher expression in branch S3_S2.
    sub-folder **'S0'**: Set of linearized plots (subway and stream plots) for each of the cell states, for example, choosing S0 state as root state:   
    - **subway_map.pdf**: single-cell level cellular branches plot
    - **stream_plot.pdf**: density level cellular branches plot
    - **subway_map_gene.pdf**: gene expression pattern on subway map plot
    - **stream_plot_gene.pdf**: gene expression pattern on stream plot
*   sub-folder **'Precomputed'**:
    - It contains files that store computed variables used when the flag -p is enabled.
