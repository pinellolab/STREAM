# Ariadne

STREAM is a powerful computational pipeline implemented in python for reconstructing complex celluar developmental trajectories from scPCR, scRNA-seq or scATAC-seq data.

Ariade provides two different informative and intuitive ways of visulizing celluar trajectories, including flat tree at single-cell level and rainbow plot at density level. It also provides a set of analytic tools that help automatically detect gene markers, which is important in defining cell subpopulations or deciding cell fates.

For given gene, Ariade can present its expression pattern in subway map plot, which is a re-layout of flat tree by choosing a start state, and rainbow plot.


Installation
------------

Make sure that all the input files are in the same directory

Input file format
----------------

The input file is a log2-transformed tab-separated gene expression matrix in tsv file format. Each row represents an unique gene and each column is one cell.  

For example, in python
```R
>import pandas as pd
>input_data = pd.read_csv('data_guoji.tsv',sep='\t',header=0,index_col=0)
>input_data.iloc[0:5,0:5]
```

|        | HSC1      | HSC1.1    | HSC1.2    | HSC1.3    | HSC1.4   |
|--------|-----------|-----------|-----------|-----------|----------|
| CD52   | 6.479620  | 0.000000  | 0.000000  | 5.550051  | 0.000000 |
| Ifitm1 | 11.688533 | 11.390682 | 10.561844 | 11.874295 | 8.976571 |
| Cdkn3  | 0.000000  | 0.000000  | 0.000000  | 0.000000  | 8.293616 |
| Ly6a   | 10.417026 | 11.452145 | 0.000000  | 8.158840  | 8.945882 |
| Bax    | 6.911608  | 10.201157 | 0.000000  | 9.396073  | 0.000000 |

Other optionally provided files format:

**cell_labels** file: tsv format. Cell labels are listed in one column. The order of labels should be consistent with cell order in gene expression matrix. No index names or column names are included.

**cell_label_color** file: tsv format. The first column is cell labels, the second column is color. The order of labels should be consistent with cell order in gene expression matrix. No index names or column names are included

**gene_list**, **feature_genes** file: tsv format. Genes are listed in one column. No index names or column names are included

**precomputed_DR** file: tsv format. Each row represents one component and each column is one cell. The columns should be the same with gene expression matrix.


Usage
-----

To run Ariadne python script at the command-line interface:
* start a terminal session;
* enter ```$ python Ariadne.py [options] ```.

Users can specify the following options:
```
-m, --matrix  
input file name (default: None)
-l, --cell_labels  
filename of cell labels (default: None)
-c, --cell_labels_colors  
filename of cell label colors (default: None)
-s, --select_features  
LOESS or PCA: Select variable genes using LOESS or principal components using PCA (default: None)
-f, --feature_genes  
prepared feature genes (default: None)
-d, --detect_genes  
whether to detect Transition and DE genes automatically
-g, --gene_list  
genes to be visualized (default: None)
-p, --use_precomputed  
whether to use precomputed data files store.pckl (default: False)
--new  
file name of data to be mapped (default: None)
--new_l  
filename of new cell labels
--new_c  
filename of new cell label colors
--log2  
whether to do log2 transformation (default: False)
--norm  
whether to normalize data based on libary size (default: False)
--atac
whether it is atac-seq data (default: No)
--n_processes  
Specify the number of processes to use. The default is cores available.
--loess_frac  
The fraction of the data used in LOESS regression (default: 0.1)
--loess_z_score_cutoff  
Z-score cutoff in gene selection based on LOESS regression (default: 1)
--pca_max_PC  
Maximal principal components in PCA (default: 100)
--pca_n_PC  
The number of selected PCs, it's determined automatically if it's not specified
--lle_neighbours  
LLE neighbour percent (default: 0.1)
--lle_components  
LLE dimension reduction (default: 3)
--AP_damping_factor  
Affinity Propagation: damping factor (default: 0.75)
--AP_min_percent  
Affinity Propagation: minimal percentage of cell number in each cluster (default: 0.005)
--AP_alpha_factor  
Affinity Propagation: minimal percentage of cell number in each cluster (default: 0.5)
--sp_size_cutoff  
The percentile cutoff for the size of sparse cluster(0~100) (default: 50)
--sp_density_cutoff
The percentile cutoff for the density of sparse cluster(0~100) (default: 50)
--DE_z_score_cutoff  
Differentially Expressed Genes Z-score cutoff (default: 2)
--DE_diff_cutoff  
Differentially Expressed Genes difference cutoff (default: 0.2)
--TG_spearman_cutoff  
Transition Genes Spearman correlation cutoff (default: 0.4)
--TG_diff_cutoff  
Transition Genes difference cutoff (default: 0.2)
--RB_nbins  
Number of bins on one branch used for plotting Rainbow_Plot (default: 3, suggesting range: 3~6)
--RB_log_scale  
whether to use log2 scale for y axis of rainbow_plot (default: No)
--refit  
whether to re-fit Principal Curve (default: No)
--save_figures_for_web
Output format the figures for the web interface
-o, --output_folder  
Output folder (default: Ariadne_Result)
```

Example
--------

Using the data in the example folder of this repo: data_guoji.tsv, cell_label.tsv and cell_label_color.tsv, and assuming that they are in the current folder, you can run the analysis with the following command:

```sh
$ python Ariadne.py -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result  
```
if you want to visualize genes of interest, you can provide gene list file 'gene_list.tsv' and add '-p' to call the precomputed file obtained from the first running (In this way, the other existing figures will not be re-generated any more):

```sh
$ python Ariadne.py -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result -g gene_list.tsv -p
```

if you want to explore potential DE genes and transition genes, you can add '-d' to detect marker genes automatically and top three genes from DE genes and transition genes of each branch pair or each branch will plotted automatically:

```sh
$ python Ariadne.py -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result -d
```



Using Ariadne with Docker
------------------

Assuming that the input files are in the current folder, the command line for docker is:

```
docker run -v $PWD/:/DATA -w /DATA pinellolab/ariadne ariadne-cli -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result --save_figures_for_web

## Fix two commands below
docker run -v $PWD/:/DATA -w /DATA lucapinello/ariadne python Ariadne.py -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result -g gene_list.tsv -p

docker run -v $PWD/:/DATA -w /DATA lucapinello/ariadne python Ariadne.py -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -o qPCR_result -d
```

Interactive web application
------------------

[Our web-hosted version is available here.](http://ariadne.pinellolab.org) However, you may be interested in running the application
on your own machine, in which case this can be executed using the following command--

```
docker run -p 3838:3838 -t pinellolab/ariadne ariadne-webapp
```


Results
-------

The output folder is specified by user in the command line (option -o). It contains the following files and directories:

* **LLE.png**: dimension reduction plot of LLE in 3D space
* **IAP.png**: clustering plot of Improved Affinity Propagation in 3D space
* **MST.png**: Minimum Spanning Tree plot in 3D space
* **MST_Branches.png**: Minimum spanning Tree with extracted branches in 3D space
* **Principal_Curve.png**: Principal Curve plot in 3D space.
* **flat_tree.png**: single-cell level flat tree plot in 2D plane
* **cell_info.tsv**: Cell information file. The index is the cell order in the original input file. Column 'CELL_ID' is cell names in input file. Column 'branch_id' is the branch id which cell is assigned to. The branch id consists of two cell states. Column 'lam' is the arc length from the first cell state of branch id to projection of cell on the branch. Column 'distance' is the euclidian distance between cell and its projection on the branch.  
* **df_sc.msg**,**store.pckl**: file that stores computed variables and can be called using -p in command line
* sub-folder **'Transition_Genes'**:
    - **Transition_Genes_S1_S2.png**: Detected transition genes plot for branch S1_S2. Orange bars are genes whose expression values increase from state S1 to S2 and green bars are genes whose expression values decrease from S1 to S2
    - **Transition_Genes_S1_S2.tsv**: Table that stores information of detected transition genes for branch S1_S2
* sub-folder **'DE_Genes'**:
    - **DE_genes_S1_S2 and S3_S2.png**: Detected differentially expressed top 15 genes plot. Red bars are genes that have higher gene expression in branch S1_S2, blue bars are genes that have higher gene expression in branch S3_S2
    - **DE_up_genes_S1_S2 and S3_S2.tsv**: Table that stores information of DE genes that have higher expression in branch S1_S2.
    - **DE_down_genes_S1_S2 and S3_S2.tsv**: Table that stores information of DE genes that have higher expression in branch S3_S2.
* sub-folder **'S0'**: Choosing S0 state as root state
    - **subway_map.png**: single-cell level cellular branches plot
    - **rainbow_branches.png**: density level cellular branches plot
    - **subway_map_gene.png**: gene expression pattern subway map plot
    - **subway_map_filtered_0.6_gene.png**: gene expression pattern subway map plot after filtering out cells whose expression level is less than 60 percent according to that's gene expression distribution
    - **rainbow_branches_gene.png**: gene expression pattern rainbow plot
