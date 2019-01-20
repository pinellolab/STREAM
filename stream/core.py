import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import anndata as ad
import networkx as nx
import re
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.stats.multicomp import multipletests
import seaborn as sns
import pylab as plt
import shapely.geometry as geom
import multiprocessing
import os
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from sklearn.manifold import LocallyLinearEmbedding,TSNE
from sklearn.cluster import SpectralClustering,AffinityPropagation,KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min,pairwise_distances,euclidean_distances
import matplotlib.patches as Patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.axes3d import Axes3D
import umap
from copy import deepcopy
import itertools
from scipy.spatial import distance,cKDTree,KDTree
import math
import matplotlib as mpl
# mpl.use('Agg')
mpl.rc('pdf', fonttype=42)
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats,interpolate
from scipy.stats import spearmanr,mannwhitneyu,gaussian_kde,kruskal
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline,UnivariateSpline
from scipy.signal import savgol_filter
from scipy.linalg import eigh, svd, qr, solve
from slugify import slugify
from decimal import *
import matplotlib.gridspec as gridspec
import pickle
import gzip


from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from .extra import *
#scikit_posthocs is currently not available in conda system. We will update it once it can be installed via conda.
#import scikit_posthocs as sp
from .scikit_posthocs import posthoc_conover

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read(file_name,file_name_sample=None,file_name_region=None,file_path='',file_format='tsv',delimiter='\t',experiment='rna-seq', workdir=None,**kwargs):
    """Read gene expression matrix into anndata object.
    
    Parameters
    ----------
    file_name: `str`
        File name. For atac-seq data, it's the count file name.
    file_name_sample: `str`
        Sample file name. Only valid when atac_seq = True.
    file_name_region: `str`
        Region file name. Only valid when atac_seq = True.
    file_path: `str`, optional (default: '')
        File path. By default it's empty
    file_format: `str`, optional (default: 'tsv')
        File format. currently supported file formats: 'tsv','txt','tab','data','csv','mtx','h5ad','pklz','pkl'
    delimiter: `str`, optional (default: '\t')
        Delimiter to use.
    experiment: `str`, optional (default: 'rna-seq')
        Choose from {{'rna-seq','atac-seq'}}       
    workdir: `float`, optional (default: None)
        Working directory. If it's not specified, a folder named 'stream_result' will be created under the current directory
    **kwargs: additional arguments to `Anndata` reading functions
   
    Returns
    -------
    AnnData object
    """

    _fp = lambda f:  os.path.join(file_path,f)
    if(experiment == 'rna-seq'):
        if(file_format in ['tsv','txt','tab','data']):
            adata = ad.read_text(_fp(file_name),delimiter=delimiter,**kwargs).T
            adata.raw = adata        
        elif(file_format == 'csv'):
            adata = ad.read_csv(_fp(file_name),delimiter=delimiter,**kwargs).T
            adata.raw = adata
        elif(file_format == 'mtx'):
            adata = ad.read_mtx(_fp(file_name),**kwargs).T 
            adata.raw = adata
        elif(file_format == 'h5ad'):
            adata = ad.read_h5ad(_fp(file_name),**kwargs)
        elif(file_format == 'pklz'):
            f = gzip.open(_fp(file_name), 'rb')
            adata = pickle.load(f)
            f.close()  
        elif(file_format == 'pkl'):
            f = open(_fp(file_name), 'rb')
            adata = pickle.load(f)
            f.close()            
        else:
            print('file format ' + file_format + ' is not supported')
            return
    elif(experiment == 'atac-seq'):
        if(file_format == 'pklz'):
            f = gzip.open(_fp(file_name), 'rb')
            adata = pickle.load(f)
            f.close()  
        elif(file_format == 'pkl'):
            f = open(_fp(file_name), 'rb')
            adata = pickle.load(f)
            f.close()
        else:            
            if(file_name_sample is None):
                print('sample file must be provided')
            if(file_name_region is None):
                print('region file must be provided')
            df_counts = pd.read_csv(file_name,sep='\t',header=None,names=['i','j','x'],compression= 'gzip' if file_name.split('.')[-1]=='gz' else None)
            df_regions = pd.read_csv(file_name_region,sep='\t',header=None,compression= 'gzip' if file_name_region.split('.')[-1]=='gz' else None)
            df_regions = df_regions.iloc[:,:3]
            df_regions.columns = ['seqnames','start','end']
            df_samples = pd.read_csv(file_name_sample,sep='\t',header=None,names=['cell_id'],compression= 'gzip' if file_name_sample.split('.')[-1]=='gz' else None)
            adata = ad.AnnData()
            adata.uns['atac-seq'] = dict()
            adata.uns['atac-seq']['count'] = df_counts
            adata.uns['atac-seq']['region'] = df_regions
            adata.uns['atac-seq']['sample'] = df_samples
    else:
        print('The experiment '+experiment +' is not supported')
        return        
    adata.uns['experiment'] = experiment
    if(workdir==None):
        if('workdir' in adata.uns_keys()):
            workdir = adata.uns['workdir']
            print('Working directory specified.')
        else:
            workdir = os.path.join(os.getcwd(), 'stream_result')
            print('Working directory not set.')
    if(not os.path.exists(workdir)):
        os.makedirs(workdir)
    adata.uns['workdir'] = workdir
    print('Saving results in: %s' % workdir)
    return adata


def counts_to_kmers(adata,k=7,n_jobs = multiprocessing.cpu_count()):
    """Covert counts files to kmer files.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    k: `int`, optional (default: 7)
        k mer.  
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run
        
    Returns
    -------
    updates `adata` with the following fields.
    
    X : `numpy.ndarray` (`adata.X`)
        A #observations × #k-mers scaled z-score matrix.
    z_score: `numpy.ndarray` (`adata.layers['z_score']`)
        A #observations × #k-mers z-score matrix.
    atac-seq: `dict` (`adata.uns['atac-seq']`)   
        A dictionary containing the following keys:
        'count': (`adata.uns['atac-seq']['count']`), dataframe in sparse format, 
                the first column specifies the rows indices (the regions) for non-zero entry. 
                the second column specifies the columns indices (the sample) for non-zero entry. 
                the last column contains the number of reads in a given region for a particular cell.
        'region': (`adata.uns['atac-seq']['region']`), dataframe
                the first column specifies chromosome names.
                the second column specifies the start position of the region.
                the third column specifies the end position of the region.
        'sample': (`adata.uns['atac-seq']['sample']`), dataframe, the name of samples
    """
    chromVAR = importr('chromVAR')
    GenomicRanges = importr('GenomicRanges')
    SummarizedExperiment = importr('SummarizedExperiment')
    BSgenome_Hsapiens_UCSC_hg19 = importr('BSgenome.Hsapiens.UCSC.hg19')
    r_Matrix = importr('Matrix')
    BiocParallel = importr('BiocParallel')
    BiocParallel.register(BiocParallel.MulticoreParam(n_jobs))
    pandas2ri.activate()
    df_regions = adata.uns['atac-seq']['region']
    r_regions_dataframe = pandas2ri.py2ri(df_regions)
    regions = GenomicRanges.makeGRangesFromDataFrame(r_regions_dataframe)
    
    df_counts = adata.uns['atac-seq']['count']
    counts = r_Matrix.sparseMatrix(i = df_counts['i'], j = df_counts['j'], x=df_counts['x'])
    
    df_samples = adata.uns['atac-seq']['sample']
    samples = pandas2ri.py2ri(df_samples)
    samples.rownames = df_samples['cell_id']
    
    SE = SummarizedExperiment.SummarizedExperiment(rowRanges = regions,colData = samples,assays = robjects.ListVector({'counts':counts}))
    SE = chromVAR.addGCBias(SE, genome = BSgenome_Hsapiens_UCSC_hg19.BSgenome_Hsapiens_UCSC_hg19)
    
    # compute kmer deviations
    KmerMatch = chromVAR.matchKmers(k, SE, BSgenome_Hsapiens_UCSC_hg19.BSgenome_Hsapiens_UCSC_hg19)
    BiocParallel.register(BiocParallel.SerialParam())
    Kmerdev = chromVAR.computeDeviations(SE, KmerMatch)
    KmerdevTable = SummarizedExperiment.assays(Kmerdev)
    cn = pandas2ri.ri2py((Kmerdev.do_slot('colData')).do_slot('listData').rx2('cell_id'))
    rn = pandas2ri.ri2py(Kmerdev.do_slot('NAMES'))
    scores = pandas2ri.ri2py(KmerdevTable.do_slot('listData').rx2('deviations'))    

    df_zscores = pd.DataFrame(scores,index=rn,columns=cn)
    df_zscores_scaled = preprocessing.scale(df_zscores,axis=1)
    df_zscores_scaled = pd.DataFrame(df_zscores_scaled,index=df_zscores.index,columns=df_zscores.columns)
    adata_new = ad.AnnData(X=df_zscores_scaled.values.T, obs={'obs_names':df_zscores_scaled.columns},var={'var_names':df_zscores_scaled.index})
    adata_new.raw = adata_new
    adata_new.uns['workdir'] = adata.uns['workdir']
    adata_new.uns['experiment'] = adata.uns['experiment']
    adata_new.uns['atac-seq'] = dict()
    adata_new.uns['atac-seq']['count'] = df_counts
    adata_new.uns['atac-seq']['region'] = df_regions
    adata_new.uns['atac-seq']['sample'] = df_samples
    adata_new.layers["z_score"] = df_zscores.values.T
    return adata_new


def write(adata,file_name=None,file_path='',file_format='pkl'):
    """Write Anndate object to file
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix. 
    file_name: `str`, optional (default: None)
        File name. If it's not specified, a file named 'stream_result' with the specified file format will be created 
        under the working directory
    file_path: `str`, optional (default: '')
        File path. If it's not specified, it's set to working directory
    file_format: `str`, optional (default: 'pkl')
        File format. By default it's compressed pickle file. Currently two file formats are supported:
        'pklz': compressed pickle file
        'pkl': pickle file
    """

    if(file_name is None):
        file_name = 'stream_result.'+file_format
    
    if(file_path is ''):
        file_path = adata.uns['workdir']
    
    if(file_format == 'pklz'):
        f = gzip.open(os.path.join(file_path,file_name), 'wb')
        pickle.dump(adata, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()  
    elif(file_format == 'pkl'):
        f = open(os.path.join(file_path,file_name), 'wb')
        pickle.dump(adata, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()            
    else:
        print('file format ' + file_format + ' is not supported')
        return


def add_cell_labels(adata,file_path='',file_name=None):
    """Add cell lables.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    fig_path: `str`, optional (default: '')
        The file path of cell label file.
    fig_name: `str`, optional (default: None)
        The file name of cell label file. If file_name is not specified, 'unknown' is added as the label for all cells.
        

    Returns
    -------
    updates `adata` with the following fields.
    label: `pandas.core.series.Series` (`adata.obs['label']`,dtype `str`)
        Array of #observations that stores the label of each cell.
    """
    _fp = lambda f:  os.path.join(file_path,f)
    if(file_name!=None):
        df_labels = pd.read_csv(_fp(file_name),sep='\t',header=None,index_col=None,names=['label'],
                                dtype=str,compression= 'gzip' if file_name.split('.')[-1]=='gz' else None)
        df_labels['label'] = df_labels['label'].str.replace('/','-')        
        df_labels.index = adata.obs_names
        adata.obs['label'] = df_labels
    else:
        print('No cell label file is provided, \'unknown\' is used as the default cell labels')
        adata.obs['label'] = 'unknown'
    return None


def add_cell_colors(adata,file_path='',file_name=None):
    """Add cell colors.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    fig_path: `str`, optional (default: '')
        The file path of cell label color file.
    fig_name: `str`, optional (default: None)
        The file name of cell label color file. If file_name is not specified, random color are generated for each cell label.
        
    Returns
    -------
    updates `adata` with the following fields.
    label_color: `pandas.core.series.Series` (`adata.obs['label_color']`,dtype `str`)
        Array of #observations that stores the color of each cell (hex color code).
    label_color: `dict` (`adata.uns['label_color']`,dtype `str`)
        Array of #observations that stores the color of each cell (hex color code).        
    """

    _fp = lambda f:  os.path.join(file_path,f)
    labels_unique = adata.obs['label'].unique()
    if(file_name!=None):
        df_colors = pd.read_csv(_fp(file_name),sep='\t',header=None,index_col=None,names=['label','color'],
                                dtype=str,compression= 'gzip' if file_name.split('.')[-1]=='gz' else None)
        df_colors['label'] = df_colors['label'].str.replace('/','-')   
        adata.uns['label_color'] = {df_colors.iloc[x,0]:df_colors.iloc[x,1] for x in range(df_colors.shape[0])}
    else:
        print('No cell color file is provided, random color is generated for each cell label')
        if(len(labels_unique)==1):
            adata.uns['label_color'] = {labels_unique[0]:'gray'}
        else:
            list_colors = sns.color_palette("hls",n_colors=len(labels_unique)).as_hex()
            adata.uns['label_color'] = {x:list_colors[i] for i,x in enumerate(labels_unique)}
    df_cell_colors = adata.obs.copy()
    df_cell_colors['label_color'] = ''
    for x in labels_unique:
        id_cells = np.where(adata.obs['label']==x)[0]
        df_cell_colors.loc[df_cell_colors.index[id_cells],'label_color'] = adata.uns['label_color'][x]
    adata.obs['label_color'] = df_cell_colors['label_color']
    return None

# def add_cell_colors(adata,file_path = None,file_name=None,key_label='label',key_color = 'label_color'):
#     labels_unique = adata.obs[key_label].unique()
#     if(file_name!=None):
#         df_colors = pd.read_csv(file_path+file_name,sep='\t',header=None,index_col=None,
#                                 compression= 'gzip' if file_name.split('.')[-1]=='gz' else None)

#         adata.uns[key_color] = {df_colors.iloc[x,0]:df_colors.iloc[x,1] for x in range(df_colors.shape[0])}
#     else:
#         list_colors = sns.color_palette("hls",n_colors=len(labels_unique)).as_hex()
#         adata.uns[key_color] = {x:list_colors[i] for i,x in enumerate(labels_unique)}
#     df_cell_colors = adata.obs.copy()
#     df_cell_colors[key_color] = ''
#     for x in labels_unique:
#         id_cells = np.where(adata.obs[key_label]==x)[0]
#         df_cell_colors.loc[df_cell_colors.index[id_cells],key_color] = adata.uns[key_color][x]
#     adata.obs[key_color] = df_cell_colors[key_color]
#     return None


def filter_genes(adata,min_num_cells = None,min_pct_cells = None,min_count = None, expr_cutoff = 1):
    """Filter out genes based on different metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_num_cells: `int`, optional (default: None)
        Minimum number of cells expressing one gene
    min_pct_cells: `float`, optional (default: None)
        Minimum percentage of cells expressing one gene
    min_count: `int`, optional (default: None)
        Minimum number of read count for one gene
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff. If greater than expr_cutoff,the gene is considered 'expressed'
    Returns
    -------
    updates `adata` with a subset of genes that pass the filtering.      
    """

    n_counts = np.sum(adata.X,axis=0)
    adata.var['n_counts'] = n_counts
    n_cells = np.sum(adata.X>expr_cutoff,axis=0)
    adata.var['n_cells'] = n_cells 
    if(sum(list(map(lambda x: x is None,[min_num_cells,min_pct_cells,min_count])))==3):
        print('No filtering')
    else:
        gene_subset = np.ones(len(adata.var_names),dtype=bool)
        if(min_num_cells!=None):
            print('Filter genes based on min_num_cells')
            gene_subset = (n_cells>min_num_cells) & gene_subset
        if(min_pct_cells!=None):
            print('Filter genes based on min_pct_cells')
            gene_subset = (n_cells>adata.shape[0]*min_pct_cells) & gene_subset
        if(min_count!=None):
            print('Filter genes based on min_count')
            gene_subset = (n_counts>min_count) & gene_subset 
        adata._inplace_subset_var(gene_subset)
        print('After filtering out low-expressed genes: ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' genes')
    return None


def filter_cells(adata,min_num_genes = None,min_pct_genes = None,min_count=None,expr_cutoff = 1):
    """Filter out cells based on different metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_num_genes: `int`, optional (default: None)
        Minimum number of genes expressed
    min_pct_genes: `float`, optional (default: None)
        Minimum percentage of genes expressed
    min_count: `int`, optional (default: None)
        Minimum number of read count for one cell
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff. If greater than expr_cutoff,the gene is considered 'expressed'
    Returns
    -------
    updates `adata` with a subset of cells that pass the filtering.      
    """

    n_counts = np.sum(adata.X,axis=1)
    adata.obs['n_counts'] = n_counts
    n_genes = np.sum(adata.X>=expr_cutoff,axis=1)
    adata.obs['n_genes'] = n_genes
    if(sum(list(map(lambda x: x is None,[min_num_genes,min_pct_genes,min_count])))==3):
        print('No filtering')    
    else:
        cell_subset = np.ones(len(adata.obs_names),dtype=bool)
        if(min_num_genes!=None):
            print('filter cells based on min_num_genes')
            cell_subset = (n_genes>=min_num_genes) & cell_subset
        if(min_pct_genes!=None):
            print('filter cells based on min_pct_genes')
            cell_subset = (n_genes>=adata.shape[0]*min_pct_genes) & cell_subset
        if(min_count!=None):
            print('filter cells based on min_count')
            cell_subset = (n_counts>=min_count) & cell_subset 
        adata._inplace_subset_obs(cell_subset)
        print('after filtering out low-expressed cells: ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' genes')
    return None


def log_transform(adata,base=2):
    """Logarithmize gene expression.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    base: `int`, optional (default: 2)
        The base used to calculate logarithm

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes logarithmized data matrix.
    """

    adata.X = np.log2(adata.X+1)/np.log2(base)
    return None


def normalize_per_cell(adata):
    """Normalize gene expression based on library size.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes normalized data matrix.
    """
    adata.X = (np.divide(adata.X.T,adata.X.sum(axis=1)).T)*1e6


def remove_mt_genes(adata):
    """remove mitochondrial genes.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with a subset of genes that excluded mitochondrial genes.
    """        

    r = re.compile("^MT-",flags=re.IGNORECASE)
    mt_genes = list(filter(r.match, adata.var_names))
    if(len(mt_genes)>0):
        print('remove mitochondrial genes:')
        print(mt_genes)
        gene_subset = ~adata.var_names.isin(mt_genes)
        adata._inplace_subset_var(gene_subset)


def select_variable_genes(adata,loess_frac=0.1,percentile=95,n_genes = None,n_jobs = multiprocessing.cpu_count(),
                          save_fig=False,fig_name='std_vs_means.pdf',fig_path=None,fig_size=(5,5)):

    """Select the most variable genes.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    loess_frac: `float`, optional (default: 0.1)
        Between 0 and 1. The fraction of the data used when estimating each y-value in LOWESS function.
    percentile: `int`, optional (default: 95)
        Between 0 and 100. Specify the percentile to select genes.Genes are ordered based on its distance from the fitted curve.
    n_genes: `int`, optional (default: None)
        Specify the number of selected genes. Genes are ordered based on its distance from the fitted curve.
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run when calculating the distance from each gene to the fitted curve
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (5,5))
        figure size.
    fig_path: `str`, optional (default: '')
        if empty, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'std_vs_means.pdf')
        if save_fig is True, specify figure name.

    Returns
    -------
    updates `adata` with the following fields.
    var_genes: `numpy.ndarray` (`adata.obsm['var_genes']`)
        Store #observations × #var_genes data matrix used for subsequent dimension reduction.
    var_genes: `pandas.core.indexes.base.Index` (`adata.uns['var_genes']`)
        The selected variable gene names.
    """

    if(fig_path is None):
        fig_path = adata.uns['workdir']  
    mean_genes = np.mean(adata.X,axis=0)
    std_genes = np.std(adata.X,ddof=1,axis=0)
    loess_fitted = lowess(std_genes,mean_genes,return_sorted=False,frac=loess_frac)
    residuals = std_genes - loess_fitted
    XP = np.column_stack((np.sort(mean_genes),loess_fitted[np.argsort(mean_genes)]))
    mat_p = np.column_stack((mean_genes,std_genes))
    with multiprocessing.Pool(processes=n_jobs) as pool:
        dist_point_to_curve = pool.starmap(project_point_to_curve_distance,[(XP,mat_p[i,]) for i in range(XP.shape[0])])
    mat_sign = np.ones(XP.shape[0])
    mat_sign[np.where(residuals<0)[0]] = -1
    dist_point_to_curve = np.array(dist_point_to_curve)*mat_sign
    if(n_genes is None):
        cutoff = np.percentile(dist_point_to_curve,percentile)
        id_var_genes = np.where(dist_point_to_curve>cutoff)[0]
        id_non_var_genes = np.where(residuals<=cutoff)[0]
    else:
        id_var_genes = np.argsort(dist_point_to_curve)[::-1][:n_genes]
        id_non_var_genes = np.argsort(dist_point_to_curve)[::-1][n_genes:]
 
    adata.obsm['var_genes'] = adata.X[:,id_var_genes].copy()
    adata.uns['var_genes'] = adata.var_names[id_var_genes]
    print(str(len(id_var_genes))+' variable genes are selected')
    ###plotting
    fig = plt.figure(figsize=fig_size)      
    plt.scatter(mean_genes[id_non_var_genes], std_genes[id_non_var_genes],s=5,alpha=0.2,zorder=None,c='#6baed6')
    plt.scatter(mean_genes[id_var_genes], std_genes[id_var_genes],s=5,alpha=0.9,zorder=1,c='#EC4E4E')
    plt.plot(np.sort(mean_genes), loess_fitted[np.argsort(mean_genes)],linewidth=3,zorder=2,c='#3182bd')
    plt.xlabel('mean value')
    plt.ylabel('standard deviation')
    if(save_fig):
        plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
        plt.close(fig)
    return None


def select_top_principal_components(adata,feature=None,n_pc = 15,max_pc = 100,first_pc = False,use_precomputed=True,
                                    save_fig=False,fig_name='top_pcs.pdf',fig_path=None,fig_size=(5,5)):
    """Select top principal components.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    feature: `str`, optional (default: None)
        Choose from {{'var_genes'}}
        Features used for pricipal component analysis
        If None, all the genes will be used.
        IF 'var_genes', the most variable genes obtained from select_variable_genes() will be used.
    n_pc: `int`, optional (default: 15)
        The number of selected principal components.
    max_pc: `int`, optional (default: 100)
        The maximum number of principal components used for variance Ratio plot.
    first_pc: `bool`, optional (default: False)
        If True, the first principal component will be included
    use_precomputed: `bool`, optional (default: True)
        If True, the PCA results from previous computing will be used
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (5,5))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'top_pcs.pdf')
        if save_fig is True, specify figure name.

    Returns
    -------
    updates `adata` with the following fields.
    pca: `numpy.ndarray` (`adata.obsm['pca']`)
        Store #observations × n_components data matrix after pca. Number of components to keep is min(#observations,#variables)
    top_pcs: `numpy.ndarray` (`adata.obsm['top_pcs']`)
        Store #observations × n_pc data matrix used for subsequent dimension reduction.
    pca_variance_ratio: `numpy.ndarray` (`adata.uns['pca_variance_ratio']`)
        Percentage of variance explained by each of the selected components.
    """
    
    if(fig_path is None):
        fig_path = adata.uns['workdir']    
    if(use_precomputed and ('pca' in adata.obsm_keys())):
        print('Importing precomputed principal components')
        X_pca = adata.obsm['pca']
        pca_variance_ratio = adata.uns['pca_variance_ratio']
    else:
        sklearn_pca = sklearnPCA(svd_solver='full')
        if(feature == 'var_genes'):
            print('using most variable genes ...')
            X_pca = sklearn_pca.fit_transform(adata.obsm['var_genes'])
            pca_variance_ratio = sklearn_pca.explained_variance_ratio_
            adata.obsm['pca'] = X_pca
            adata.uns['pca_variance_ratio'] = pca_variance_ratio                
        else:
            print('using all the genes ...')
            X_pca = sklearn_pca.fit_transform(adata.X)
            pca_variance_ratio = sklearn_pca.explained_variance_ratio_
            adata.obsm['pca'] = X_pca
            adata.uns['pca_variance_ratio'] = pca_variance_ratio            
    if(first_pc):
        adata.obsm['top_pcs'] = X_pca[:,0:(n_pc)]
    else:
        #discard the first Principal Component
        adata.obsm['top_pcs'] = X_pca[:,1:(n_pc+1)]
    print(str(n_pc) + ' PCs are selected')
    ##plotting
    fig = plt.figure(figsize=fig_size)
    plt.plot(range(max_pc),pca_variance_ratio[:max_pc])
    if(first_pc):
        plt.axvline(n_pc,c='red',ls = '--')
    else:
        plt.axvline(1,c='red',ls = '--')
        plt.axvline(n_pc+1,c='red',ls = '--')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    if(save_fig):
        plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
        plt.close(fig)
    return None


def dimension_reduction(adata,nb_pct = 0.1,n_components = 3,n_jobs = multiprocessing.cpu_count(),feature='var_genes',method = 'mlle'):

    """Perform dimension reduction.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    nb_pct: `float`, optional (default: 0.1)
        The percentage neighbor cells (only valid when 'mlle' or 'umap' is specified).
    n_components: `int`, optional (default: 3)
        Number of components to keep.
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run.
    feature: `str`, optional (default: 'var_genes')
        Choose from {{'var_genes','top_pcs','all'}}
        Feature used for dimension reduction.
        'var_genes': most variable genes
        'top_pcs': top principal components
        'all': all genes
    method: `str`, optional (default: 'mlle')
        Choose from {{'mlle','umap','pca'}}
        Method used for dimension reduction.
        'mlle': Modified locally linear embedding algorithm
        'umap': Uniform Manifold Approximation and Projection
        'pca': Principal component analysis
   
    Returns
    -------
    updates `adata` with the following fields.
    
    X_dr : `numpy.ndarray` (`adata.obsm['X_dr']`)
        A #observations × n_components data matrix after dimension reduction.
    X_mlle : `numpy.ndarray` (`adata.obsm['X_mlle']`)
        Store #observations × n_components data matrix after mlle.
    X_umap : `numpy.ndarray` (`adata.obsm['X_umap']`)
        Store #observations × n_components data matrix after umap.
    X_pca : `numpy.ndarray` (`adata.obsm['X_pca']`)
        Store #observations × n_components data matrix after pca.
    trans_mlle : `sklearn.manifold.locally_linear.LocallyLinearEmbedding` (`adata.uns['trans_mlle']`)
        Store mlle object
    trans_umap : `umap.UMAP` (`adata.uns['trans_umap']`)
        Store umap object
    trans_pca : `sklearn.decomposition.PCA` (`adata.uns['trans_pca']`)
        Store pca object 
    """

    if(feature == 'var_genes'):
        input_data = adata.obsm['var_genes']
    if(feature == 'top_pcs'):
        input_data = adata.obsm['top_pcs']
    if(feature == 'all'):
        input_data = adata.X
    print(str(n_jobs)+' cpus are being used ...')
    if(method == 'mlle'):
        np.random.seed(2)
        reducer = LocallyLinearEmbedding(n_neighbors=int(np.around(input_data.shape[0]*nb_pct)), 
                                             n_components=n_components,
                                             n_jobs = n_jobs,
                                             method = 'modified',eigen_solver = 'dense',random_state=10,
                                             neighbors_algorithm = 'kd_tree')
        trans = reducer.fit(input_data)
        adata.uns['trans_mlle'] = trans
        adata.obsm['X_mlle'] = trans.embedding_
        adata.obsm['X_dr'] = trans.embedding_
    if(method == 'umap'):
        reducer = umap.UMAP(n_neighbors=int(input_data.shape[0]*nb_pct),n_components=n_components,random_state=42)
        trans = reducer.fit(input_data)
        adata.uns['trans_umap'] = trans
        adata.obsm['X_umap'] = trans.embedding_
        adata.obsm['X_dr'] = trans.embedding_
    if(method == 'pca'):
        reducer = sklearnPCA(n_components=n_components,svd_solver='full')
        trans = reducer.fit(input_data)
        adata.uns['trans_pca'] = trans
        adata.obsm['X_pca'] = trans.transform(input_data) 
        adata.obsm['X_dr'] = adata.obsm['X_pca']
    return None


def plot_dimension_reduction(adata,n_components = 3,comp1=0,comp2=1,
                             save_fig=False,fig_name='dimension_reduction.pdf',fig_path=None,fig_size=(8,8),fig_legend_ncol=3):
    """Plot cells after dimension reduction.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    n_components: `int`, optional (default: 3)
        Number of components to be plotted.
    comp1: `int`, optional (default: 0)
        Component used for x axis.
    comp2: `int`, optional (default: 1)
        Component used for y axis.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'dimension_reduction.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.
        
    Returns
    -------
    None
    
    """
    if(fig_path is None):
        fig_path = adata.uns['workdir']
    df_sample = adata.obs[['label','label_color']].copy()
    df_coord = pd.DataFrame(adata.obsm['X_dr'],index=adata.obs_names)
    list_patches = []
    for x in adata.uns['label_color'].keys():
        list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
    color = df_sample.sample(frac=1,random_state=100)['label_color'] 
    coord = df_coord.sample(frac=1,random_state=100)
    if(n_components==3): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coord[0], coord[1],coord[2],c=color,s=50,linewidth=0,alpha=0.8) 
        max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min(), coord[2].max()-coord[2].min()]).max() / 1.9
        mid_x = (coord[0].max()+coord[0].min()) * 0.5
        mid_y = (coord[1].max()+coord[1].min()) * 0.5
        mid_z = (coord[2].max()+coord[2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('Component1',labelpad=20)
        ax.set_ylabel('Component2',labelpad=20)
        ax.set_zlabel('Component3',labelpad=20)
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)
    if(n_components==2): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(coord[comp1], coord[comp2],c=color,s=50,linewidth=0,alpha=0.8) 
        max_range = np.array([coord[comp1].max()-coord[comp1].min(), coord[comp2].max()-coord[comp2].min()]).max() / 1.9
        mid_x = (coord[comp1].max()+coord[comp1].min()) * 0.5
        mid_y = (coord[comp2].max()+coord[comp2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_xlabel('Component1',labelpad=20)
        ax.set_ylabel('Component2',labelpad=20)
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)


def plot_branches(adata,n_components = 3,comp1=0,comp2=1,key_graph='epg',save_fig=False,fig_name='branches.pdf',fig_path=None,fig_size=(8,8)):  
    """Plot branches skeleton with all nodes.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    n_components: `int`, optional (default: 3)
        Number of components to be plotted.
    comp1: `int`, optional (default: 0)
        Component used for x axis.
    comp2: `int`, optional (default: 1)
        Component used for y axis.
    key_graph: `str`, optional (default: None): 
        Choose from {{'epg','seed_epg','ori_epg'}}
        Specify gragh to be plotted.
        'epg' current elastic principal graph
        'seed_epg' seed structure used for elastic principal graph learning, which is obtained by running seed_elastic_principal_graph()
        'ori_epg' original elastic principal graph, which is obtained by running elastic_principal_graph()
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'branches.pdf')
        if save_fig is True, specify figure name.

    Returns
    -------
    None
    
    """
    if(fig_path is None):
        fig_path = adata.uns['workdir']
    
    if(key_graph=='epg'):
        epg = adata.uns['epg']
        flat_tree = adata.uns['flat_tree']
    elif(key_graph=='seed_epg'):
        epg = adata.uns['seed_epg']
        flat_tree = adata.uns['seed_flat_tree']        
    elif(key_graph=='ori_epg'):
        epg = adata.uns['ori_epg']
        flat_tree = adata.uns['ori_flat_tree'] 
    else:
        print("'"+key_graph+"'"+'is not supported')
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    nodes_pos = np.array(list(dict_nodes_pos.values()))
    coord = pd.DataFrame(adata.obsm['X_dr'])
    if(n_components>=3): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        for edge_i in flat_tree.edges():
            branch_i_nodes = flat_tree.edges[edge_i]['nodes']
            branch_i_color = flat_tree.edges[edge_i]['color']
            branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])
            ax.plot(branch_i_pos[:,0],branch_i_pos[:,1],branch_i_pos[:,2],c = branch_i_color,lw=5,zorder=None)
        ax.scatter(nodes_pos[:,0],nodes_pos[:,1],nodes_pos[:,2],color='gray',s=12,alpha=1,zorder=5)
        for i in dict_nodes_pos.keys():
            ax.text(dict_nodes_pos[i][0],dict_nodes_pos[i][1],dict_nodes_pos[i][2],i,color='black',fontsize = 10)
        max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min(), coord[2].max()-coord[2].min()]).max() / 1.9
        mid_x = (coord[0].max()+coord[0].min()) * 0.5
        mid_y = (coord[1].max()+coord[1].min()) * 0.5
        mid_z = (coord[2].max()+coord[2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('Component1',labelpad=20)
        ax.set_ylabel('Component2',labelpad=20)
        ax.set_zlabel('Component3',labelpad=20)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)
    if(n_components==2): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        for edge_i in flat_tree.edges():
            branch_i_nodes = flat_tree.edges[edge_i]['nodes']
            branch_i_color = flat_tree.edges[edge_i]['color']
            branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])
            ax.plot(branch_i_pos[:,comp1],branch_i_pos[:,comp2],c = branch_i_color,lw=5,zorder=None)
        ax.scatter(nodes_pos[:,comp1],nodes_pos[:,comp2],color='gray',s=12,alpha=1,zorder=5)
        for i in dict_nodes_pos.keys():
            ax.text(dict_nodes_pos[i][comp1],dict_nodes_pos[i][comp2],i,color='black',fontsize = 10)
        max_range = np.array([coord[comp1].max()-coord[comp1].min(), coord[comp2].max()-coord[comp2].min()]).max() / 1.9
        mid_x = (coord[comp1].max()+coord[comp1].min()) * 0.5
        mid_y = (coord[comp2].max()+coord[comp2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_xlabel('Component'+str(comp1+1),labelpad=20)
        ax.set_ylabel('Component'+str(comp2+1),labelpad=20)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)        


def plot_branches_with_cells(adata,adata_new=None,n_components = 3,comp1=0,comp2=1,key_graph='epg',show_all_cells=True,
                             save_fig=False,fig_name='branches_with_cells.pdf',fig_path=None,fig_size=(8,8),fig_legend_ncol=3):    
    """Plot branches along with cells. The branches only contain leaf nodes and branching nodes
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    n_components: `int`, optional (default: 3)
        Number of components to be plotted.
    comp1: `int`, optional (default: 0)
        Component used for x axis.
    comp2: `int`, optional (default: 1)
        Component used for y axis.
    key_graph: `str`, optional (default: None): 
        Choose from {{'epg','seed_epg','ori_epg'}}
        Specify gragh to be plotted.
        'epg' current elastic principal graph
        'seed_epg' seed structure used for elastic principal graph learning, which is obtained by running seed_elastic_principal_graph()
        'ori_epg' original elastic principal graph, which is obtained by running elastic_principal_graph()
    show_all_cells: `bool`, optional (default: False)
        if show_all_cells is True and adata_new is speicified, both original cells and mapped cells will be shown
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'branches_with_cells.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.

    Returns
    -------
    None

    """

    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']
    if(key_graph=='epg'):
        epg = adata.uns['epg']
        flat_tree = adata.uns['flat_tree']
    elif(key_graph=='seed_epg'):
        epg = adata.uns['seed_epg']
        flat_tree = adata.uns['seed_flat_tree']        
    elif(key_graph=='ori_epg'):
        epg = adata.uns['ori_epg']
        flat_tree = adata.uns['ori_flat_tree'] 
    else:
        print("'"+key_graph+"'"+'is not supported')
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    nodes_pos = np.array(list(dict_nodes_pos.values()))
    
    dict_nodes_label = nx.get_node_attributes(flat_tree,'label')
    branches_color = nx.get_edge_attributes(flat_tree,'color') 
    df_sample = adata.obs[['label','label_color']].copy()
    df_coord = pd.DataFrame(adata.obsm['X_dr'],index=adata.obs_names)
    list_patches = []
    for x in adata.uns['label_color'].keys():
        list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
    color = df_sample.sample(frac=1,random_state=100)['label_color'] 
    coord = df_coord.sample(frac=1,random_state=100)
    if(adata_new !=None):
        if(not show_all_cells):
            list_patches = []
        for x in adata_new.uns['label_color'].keys():
            list_patches.append(Patches.Patch(color = adata_new.uns['label_color'][x],label=x))
        df_sample_new = adata_new.obs[['label','label_color']].copy()
        df_coord_new = pd.DataFrame(adata_new.obsm['X_dr'],index=adata_new.obs_names)
        color_new = df_sample_new.sample(frac=1,random_state=100)['label_color'] 
        coord_new = df_coord_new.sample(frac=1,random_state=100)                    
    if(n_components>=3): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        if(adata_new is None):
            ax.scatter(coord[0], coord[1],coord[2],c=color,s=50,linewidth=0,alpha=0.8) 
        else:
            if(show_all_cells):
                ax.scatter(coord[0], coord[1],coord[2],c=color,s=50,linewidth=0,alpha=0.8) 
                ax.scatter(coord_new[0], coord_new[1],coord_new[2],c=color_new,s=50,linewidth=0,alpha=0.8)
            else:
                ax.scatter(coord_new[0], coord_new[1],coord_new[2],c=color_new,s=50,linewidth=0,alpha=0.8)
        for edge_i in flat_tree.edges():
            branch_i_nodes = flat_tree.edges[edge_i]['nodes']
            branch_i_color = flat_tree.edges[edge_i]['color']
            branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])
            ax.plot(branch_i_pos[:,0],branch_i_pos[:,1],branch_i_pos[:,2],c = branch_i_color,lw=5,zorder=None)
        for node_i in flat_tree.nodes():
            ax.text(dict_nodes_pos[node_i][0],dict_nodes_pos[node_i][1],dict_nodes_pos[node_i][2],
                    flat_tree.nodes[node_i]['label'],color='black',fontsize = 12,zorder=10)
        max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min(), coord[2].max()-coord[2].min()]).max() / 1.9
        mid_x = (coord[0].max()+coord[0].min()) * 0.5
        mid_y = (coord[1].max()+coord[1].min()) * 0.5
        mid_z = (coord[2].max()+coord[2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('Component1',labelpad=20)
        ax.set_ylabel('Component2',labelpad=20)
        ax.set_zlabel('Component3',labelpad=20)
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.1),
                  ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)
    if(n_components==2): 
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        if(adata_new is None):
            ax.scatter(coord[comp1], coord[comp2],c=color,s=50,linewidth=0,alpha=0.8) 
        else:
            if(show_all_cells):
                ax.scatter(coord[comp1], coord[comp2],c=color,s=50,linewidth=0,alpha=0.8) 
                ax.scatter(coord_new[comp1], coord_new[comp2],c=color_new,s=50,linewidth=0,alpha=0.8)
            else:
                ax.scatter(coord_new[comp1], coord_new[comp2],c=color_new,s=50,linewidth=0,alpha=0.8)
        for edge_i in flat_tree.edges():
            branch_i_nodes = flat_tree.edges[edge_i]['nodes']
            branch_i_color = flat_tree.edges[edge_i]['color']
            branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])
            ax.plot(branch_i_pos[:,comp1],branch_i_pos[:,comp2],c = branch_i_color,lw=5,zorder=None)
#         ax.scatter(nodes_pos[:,0],nodes_pos[:,1],color='gray',s=12,alpha=1,zorder=5)
        for node_i in flat_tree.nodes():
            ax.text(dict_nodes_pos[node_i][comp1],dict_nodes_pos[node_i][comp2],
                    flat_tree.nodes[node_i]['label'],color='black',fontsize = 12,zorder=10)
        max_range = np.array([coord[comp1].max()-coord[comp1].min(), coord[comp2].max()-coord[comp2].min()]).max() / 1.9
        mid_x = (coord[comp1].max()+coord[comp1].min()) * 0.5
        mid_y = (coord[comp2].max()+coord[comp2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_xlabel('Component'+str(comp1+1),labelpad=20)
        ax.set_ylabel('Component'+str(comp2+1),labelpad=20)
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.1),
                  ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
        if(save_fig):
            plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
            plt.close(fig)


def switch_to_low_dimension(adata,n_components=2):
    """Switch to low dimension space, in which the preliminary structure will be learnt.    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix
    n_components: `int`, optional (default: 2)
        Number of components used to infer the initial elastic principal graph.

    Returns
    -------
    adata_low: AnnData
        Annotated data matrix used in low dimensional space
    """
    if('X_dr' not in adata.obsm_keys()):
        print('Please run dimension reduction first')
        return
    if(adata.obsm['X_dr'].shape[1]<=n_components):         
        print('The number of components in adata should be greater than n_components ' + str(n_components))
        return
    adata_low = adata.copy()
    adata_low.obsm['X_dr'] = adata.obsm['X_dr'][:,:n_components]
    adata_low.obsm['X_dr_ori'] = adata.obsm['X_dr']
    return adata_low


def infer_initial_structure(adata_low,nb_min=5):
    """Infer the initial node positions and edges. It helps infer the initial structure used in high-dimensional space
    
    Parameters
    ----------
    adata_low: AnnData
        Annotated data matrix used in low dimensional space
    nb_min: `int`, optional (default: 2)
        Minimum number of neighbour cells when mapping elastic principal graph from low-dimension to high-dimension.
        if the number of cells within one node is greater than nb_min, these cells will be used to calculate the new position of this node in high dimensional space
        if the number of cells within one node is less than or equal to nb_min, then nb_min nearest neighbor cells of this node will be used to calculate its new position in high dimensional space    

    Returns
    -------
    init_nodes_pos: `array`, shape = [n_nodes,n_dimension], optional (default: `None`)
        initial node positions
    init_edges: `array`, shape = [n_edges,2], optional (default: `None`)
        initial edges, all the initial nodes should be included in the tree structure
    """    
    n_components = adata_low.obsm['X_dr'].shape[1]
    epg_low = adata_low.uns['epg']
    kdtree=cKDTree(adata_low.obsm['X_dr'])
    dict_nodes_pos = dict()
    nx.set_node_attributes(epg_low,values=False,name='inferred_by_knn')
    for node_i in epg_low.nodes():
        ids = np.where(adata_low.obs['node'] == node_i)[0]
        if(ids.shape[0]<=nb_min):
            ids = kdtree.query(nx.get_node_attributes(epg_low,'pos')[node_i],k=nb_min)[1]
            print('Node '+ str(node_i) +' is calculated using ' + str(nb_min) + 'nearest neighbor cells')
            epg_low.nodes[node_i]['inferred_by_knn'] = True
        dict_nodes_pos[node_i] = np.concatenate((nx.get_node_attributes(epg_low,'pos')[node_i],
                                                 np.mean(adata_low.obsm['X_dr_ori'][ids,n_components:],axis=0)))
        init_nodes_pos = np.array(list(dict_nodes_pos.values()))
        init_edges = epg_low.edges()
    return init_nodes_pos,init_edges

def seed_elastic_principal_graph(adata,init_nodes_pos=None,init_edges=None,clustering='ap',damping=0.75,pref_perc=50,n_clusters=20,max_n_clusters=200,nb_pct=0.1):
    
    """Seeding the initial elastic principal graph.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    init_nodes_pos: `array`, shape = [n_nodes,n_dimension], optional (default: `None`)
        initial node positions
    init_edges: `array`, shape = [n_edges,2], optional (default: `None`)
        initial edges, all the initial nodes should be included in the tree structure
    clustering: `str`, optional (default: 'ap')
        Choose from {{'ap','kmeans','sc'}}
        clustering method used to infer the initial nodes.
        'ap' affinity propagation
        'kmeans' K-Means clustering
        'sc' spectral clustering
    damping: `float`, optional (default: 0.75)
        Damping factor (between 0.5 and 1) for affinity propagation.
    pref_perc: `int`, optional (default: 50)
        Preference percentile (between 0 and 100). The percentile of the input similarities for affinity propagation.
    n_clusters: `int`, optional (default: 20)
        Number of clusters (only valid once 'clustering' is specificed as 'sc' or 'kmeans').
    max_n_clusters: `int`, optional (default: 200)
        The allowed maximum number of clusters for 'ap'.
    nb_pct: `float`, optional (default: 0.1)
        Neighbor percentage. The percentage of points used as neighbors for spectral clustering.

    Returns
    -------
    updates `adata` with the following fields.

    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.        
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    seed_epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Store seeded elastic principal graph structure
    seed_flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        Store seeded flat_tree

    Notes
    -------
    The default procedure is fast and good enough when seeding structure in low-dimensional space.

    when seeding structure in high-dimensional space, it's strongly recommended that using 'infer_initial_structure' to get the initial node positions and edges

    """

    print('Seeding initial elastic principal graph...')
    input_data = adata.obsm['X_dr']
    if(init_nodes_pos is None):
        print('Clustering...')
        if(clustering=='ap'):
            print('Affinity propagation ...')
            ap = AffinityPropagation(damping=damping,preference=np.percentile(-euclidean_distances(input_data,squared=True),pref_perc)).fit(input_data)
            # ap = AffinityPropagation(damping=damping).fit(input_data)
            if(ap.cluster_centers_.shape[0]>max_n_clusters):
                print('The number of clusters is ' + str(ap.cluster_centers_.shape[0]))
                print('Too many clusters are generated, please lower pref_perc or increase damping and retry it')
                return
            cluster_labels = ap.labels_
            init_nodes_pos = ap.cluster_centers_
            epg_nodes_pos = init_nodes_pos  
        elif(clustering=='sc'):
            print('Spectral clustering ...')
            sc = SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors',n_neighbors=np.int(input_data.shape[0]*nb_pct),
                                    eigen_solver='arpack',random_state=42).fit(input_data)
            cluster_labels = sc.labels_ 
            init_nodes_pos = np.empty((0,input_data.shape[1])) #cluster centers
            for x in np.unique(cluster_labels):
                id_cells = np.array(range(input_data.shape[0]))[cluster_labels==x]
                init_nodes_pos = np.vstack((init_nodes_pos,np.median(input_data[id_cells,:],axis=0)))
            epg_nodes_pos = init_nodes_pos
        elif(clustering=='kmeans'):
            print('K-Means clustering ...')
            kmeans = KMeans(n_clusters=n_clusters,init='k-means++').fit(input_data)
            cluster_labels = kmeans.labels_
            init_nodes_pos = kmeans.cluster_centers_
            epg_nodes_pos = init_nodes_pos     
        else:
            print("'"+clustering+"'" + ' is not supported')
    else:
        epg_nodes_pos = init_nodes_pos
        print('Setting initial nodes...')
    print('The number of initial nodes is ' + str(epg_nodes_pos.shape[0]))

    if(init_edges is None):
        #Minimum Spanning Tree
        print('Calculatng minimum spanning tree...')
        D=pairwise_distances(epg_nodes_pos)
        G=nx.from_numpy_matrix(D)
        mst=nx.minimum_spanning_tree(G)
        epg_edges = np.array(mst.edges())
    else:
        print('Setting initial edges...')
        epg_edges = init_edges


    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
    adata.uns['seed_epg'] = deepcopy(epg)
    adata.uns['seed_flat_tree'] = deepcopy(flat_tree)  
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of initial branches: ' + str(len(dict_branches))) 


def elastic_principal_graph(adata,epg_n_nodes = 50,incr_n_nodes=30,epg_lambda=0.02,epg_mu=0.1,epg_trimmingradius='Inf',
                            epg_finalenergy = 'Penalized',epg_alpha=0.02,epg_beta=0.0,epg_n_processes=1,
                            save_fig=False,fig_name='ElPiGraph_analysis.pdf',fig_path=None,fig_size=(8,8),**kwargs):
    """Elastic principal graph learning.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    epg_n_nodes: `int`, optional (default: 50)
        Number of nodes for elastic principal graph.
    incr_n_nodes: `int`, optional (default: 30)
        Incremental number of nodes for elastic principal graph when epg_n_nodes is not big enough.
    epg_lambda: `float`, optional (default: 0.02)
        lambda parameter used to compute the elastic energy.
    epg_mu: `float`, optional (default: 0.1)
        mu parameter used to compute the elastic energy.
    epg_trimmingradius: `float`, optional (default: 'Inf')  
        maximal distance from a node to the points it controls in the embedding.
    epg_finalenergy: `str`, optional (default: 'Penalized')
        indicate the final elastic energy associated with the configuration.
    epg_alpha: `float`, optional (default: 0.02)
        alpha parameter of the penalized elastic energy.
    epg_beta: `float`, optional (default: 0.0)
        beta parameter of the penalized elastic energy.
    epg_n_processes: `int`, optional (default: 1)
        the number of processes to use.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'ElPigraph_analysis.pdf')
        if save_fig is True, specify figure name.
    **kwargs: additional arguments to `ElPiGraph.computeElasticPrincipalTree`
   
    Returns
    -------
    updates `adata` with the following fields.
    
    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    ori_epg : `networkx.classes.graph.Graph` (`adata.uns['ori_epg']`)
        Store original elastic principal graph structure
    epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['epg_obj']`)
        R object of elastic principal graph learning.
    ori_epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['ori_epg_obj']`)
        Store original R object of elastic principal graph learning.
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    ori_flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        Store original flat_tree
    """
    if(fig_path is None):
        fig_path = adata.uns['workdir']
    input_data = adata.obsm['X_dr']
    if('seed_epg' in adata.uns_keys()):
        epg = adata.uns['seed_epg']
        dict_nodes_pos = nx.get_node_attributes(epg,'pos')
        init_nodes_pos = np.array(list(dict_nodes_pos.values()))
        init_edges = np.array(list(epg.edges())) 
        R_init_edges = init_edges + 1
        if((init_nodes_pos.shape[0]+incr_n_nodes)>=epg_n_nodes):
            print('epg_n_nodes is too small. It is corrected to the initial number of nodes plus incr_n_nodes')
            epg_n_nodes = init_nodes_pos.shape[0]+incr_n_nodes
    else:
        print('No initial structure is seeded')
        init_nodes_pos = robjects.NULL
        R_init_edges = robjects.NULL
        
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    print('Learning elastic principal graph...')
    R.pdf(os.path.join(fig_path,fig_name))
    epg_obj = ElPiGraph.computeElasticPrincipalTree(X=input_data,
                                                    NumNodes = epg_n_nodes, 
                                                    InitNodePositions = init_nodes_pos,
                                                    InitEdges=R_init_edges,
                                                    Lambda=epg_lambda, Mu=epg_mu,
                                                    TrimmingRadius= epg_trimmingradius,
                                                    FinalEnergy = epg_finalenergy,
                                                    alpha = epg_alpha,
                                                    beta = epg_beta,                                                    
                                                    Do_PCA=False,CenterData=False,
                                                    n_cores = epg_n_processes,
                                                    nReps=1,
                                                    ProbPoint=1.0,
                                                    **kwargs)
    R('dev.off()')

    epg_nodes_pos = np.array(epg_obj[0].rx2('NodePositions'))
    epg_edges = np.array((epg_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1

    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['ori_epg'] = deepcopy(epg)
    adata.uns['epg_obj'] = deepcopy(epg_obj)    
    adata.uns['ori_epg_obj'] = deepcopy(epg_obj)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after learning elastic principal graph: ' + str(len(dict_branches)))


def prune_elastic_principal_graph(adata,epg_collapse_mode = 'PointNumber',epg_collapse_par = 5,   
                                  epg_lambda=0.02,epg_mu=0.1,epg_trimmingradius='Inf',
                                  epg_finalenergy = 'base',epg_alpha=0.02,epg_beta=0.0,epg_n_processes=1,reset=False,**kwargs): 
    """Prune the learnt elastic principal graph by filtering out 'trivial' branches.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    epg_collapse_mode: `str`, optional (default: 'PointNumber')
        The mode used to prune the graph.
        Choose from {{'PointNumber','PointNumber_Extrema','PointNumber_Leaves','EdgesNumber','EdgesLength'}}
        'PointNumber': branches with less than epg_collapse_par points (points projected on the extreme points are not considered) are removed
        'PointNumber_Extrema', branches with less than epg_collapse_par (points projected on the extreme points are not considered) are removed
        'PointNumber_Leaves', branches with less than epg_collapse_par points (points projected on non-leaf extreme points are not considered) are removed
        'EdgesNumber', branches with less than epg_collapse_par edges are removed
        'EdgesLength', branches shorter than epg_collapse_par are removed        
    epg_collapse_par: `float`, optional (default: 5)
        The paramter used to control different modes.
    epg_lambda: `float`, optional (default: 0.02)
        lambda parameter used to compute the elastic energy.
    epg_mu: `float`, optional (default: 0.1)
        mu parameter used to compute the elastic energy.
    epg_trimmingradius: `float`, optional (default: 'Inf')  
        maximal distance from a node to the points it controls in the embedding.
    epg_finalenergy: `str`, optional (default: 'Penalized')
        indicate the final elastic energy associated with the configuration.
    epg_alpha: `float`, optional (default: 0.02)
        alpha parameter of the penalized elastic energy.
    epg_beta: `float`, optional (default: 0.0)
        beta parameter of the penalized elastic energy.
    epg_n_processes: `int`, optional (default: 1)
        The number of processes to use.
    reset: `bool`, optional (default: False)
        If true, reset the current elastic principal graph to the initial elastic principal graph (i.e. the graph obtained from running 'elastic_principal_graph')
    **kwargs: additional arguments to `ElPiGraph.CollapseBrances`
   
    Returns
    -------
    updates `adata` with the following fields.

    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.    
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['epg_obj']`)
        R object of elastic principal graph learning.
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    """

    print('Collasping small branches ...')
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    if(reset):
        epg_obj = adata.uns['ori_epg_obj']
        epg = adata.uns['ori_epg']
    else:
        epg_obj = adata.uns['epg_obj']
        epg = adata.uns['epg']
    input_data = adata.obsm['X_dr']
    epg_obj_collapse = ElPiGraph.CollapseBrances(X = input_data, TargetPG = epg_obj[0], Mode = epg_collapse_mode, ControlPar = epg_collapse_par, **kwargs)

    init_nodes_pos = np.array(epg_obj_collapse.rx2('Nodes'))
    init_edges = np.array(epg_obj_collapse.rx2('Edges')) - 1    
    epg_n_nodes = init_nodes_pos.shape[0]
    epg_obj = ElPiGraph.computeElasticPrincipalTree(X=input_data,
                                                    NumNodes = epg_n_nodes, 
                                                    InitNodePositions = init_nodes_pos,
                                                    InitEdges=init_edges + 1,
                                                    Lambda=epg_lambda, Mu=epg_mu,
                                                    TrimmingRadius= epg_trimmingradius,
                                                    FinalEnergy = epg_finalenergy,
                                                    alpha = epg_alpha,
                                                    beta = epg_beta,                                                    
                                                    Do_PCA=False,CenterData=False,
                                                    drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                                    n_cores = epg_n_processes,
                                                    nReps=1,
                                                    ProbPoint=1.0)
    
    epg_nodes_pos = np.array(epg_obj[0].rx2('NodePositions'))
    epg_edges = np.array((epg_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1    
        
    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['epg_obj'] = deepcopy(epg_obj)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after pruning ElPiGraph: ' + str(len(dict_branches)))


def optimize_branching(adata,incr_n_nodes=30,epg_maxsteps=50,mode=2,                                  
                       epg_lambda=0.01,epg_mu=0.1,epg_trimmingradius='Inf',
                       epg_finalenergy = 'base',epg_alpha=0.02,epg_beta=0.0,epg_n_processes=1,reset=False,**kwargs):
    """Optimize branching node by expanding the nodes around a branching point.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    incr_n_nodes: `int`, optional (default: 30)
        Incremental number of nodes for elastic principal graph.       
    epg_maxsteps: `float`, optional (default: 50)
        The maximum number of iteration steps .
    mode: `int`, optional (default: 2)
        The energy computation mode.
    epg_lambda: `float`, optional (default: 0.02)
        lambda parameter used to compute the elastic energy.
    epg_mu: `float`, optional (default: 0.1)
        mu parameter used to compute the elastic energy.
    epg_trimmingradius: `float`, optional (default: 'Inf')  
        maximal distance from a node to the points it controls in the embedding.
    epg_finalenergy: `str`, optional (default: 'Penalized')
        indicate the final elastic energy associated with the configuration.
    epg_alpha: `float`, optional (default: 0.02)
        alpha parameter of the penalized elastic energy.
    epg_beta: `float`, optional (default: 0.0)
        beta parameter of the penalized elastic energy.
    epg_n_processes: `int`, optional (default: 1)
        The number of processes to use.
    reset: `bool`, optional (default: False)
        If true, reset the current elastic principal graph to the initial elastic principal graph (i.e. the graph obtained from running 'elastic_principal_graph')
    **kwargs: additional arguments to `ElPiGraph.CollapseBrances`
   
    Returns
    -------
    updates `adata` with the following fields.

    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.        
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['epg_obj']`)
        R object of elastic principal graph learning.
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    """

    print('Optimizing branching...')
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    if(reset):
        epg_obj = adata.uns['ori_epg_obj']
        epg = adata.uns['ori_epg']
    else:
        epg_obj = adata.uns['epg_obj']
        epg = adata.uns['epg']
    input_data = adata.obsm['X_dr']

    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    init_nodes_pos = np.array(list(dict_nodes_pos.values()))
    init_edges = np.array(list(epg.edges()))  
    epg_n_nodes = init_nodes_pos.shape[0] + incr_n_nodes
    
    epg_obj = ElPiGraph.fineTuneBR(X=input_data,
                                    MaxSteps = epg_maxsteps,
                                    Mode = 2,
                                    NumNodes = epg_n_nodes, 
                                    InitNodePositions = init_nodes_pos,
                                    InitEdges=init_edges + 1,
                                    Lambda=epg_lambda, Mu=epg_mu,
                                    TrimmingRadius= epg_trimmingradius,
                                    FinalEnergy = epg_finalenergy,
                                    alpha = epg_alpha,
                                    beta = epg_beta,                                                    
                                    Do_PCA=False,CenterData=False,
                                    drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                    n_cores = epg_n_processes,
                                    nReps=1,
                                    ProbPoint=1.0,
                                    **kwargs)
    
    epg_nodes_pos = np.array(epg_obj[0].rx2('NodePositions'))
    epg_edges = np.array((epg_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1    
        
    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['epg_obj'] = deepcopy(epg_obj)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after optimizing branching: ' + str(len(dict_branches)))   


def shift_branching(adata,epg_shift_mode = 'NodeDensity',epg_shift_radius = 0.05,epg_shift_max=5,                             
                   epg_lambda=0.01,epg_mu=0.1,epg_trimmingradius='Inf',
                   epg_finalenergy = 'base',epg_alpha=0.02,epg_beta=0.0,epg_n_processes=1,reset=False,**kwargs):
    """Move branching node to the area with higher density.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    epg_shift_mode: `str`, optional (default: 'NodeDensity')
        The mode used to shift the branching nodes.
        Choose from {{'NodePoints','NodeDensity'}}
    epg_shift_radius: `float`, optional (default: 0.05)
        The radius used when computing point density if epg_shift_mode = 'NodeDensity'.
    epg_shift_max: `float`, optional (default: 5)
        The maxium distance (defined as the number of edges) to consider when exploring the neighborhood of branching point
    epg_lambda: `float`, optional (default: 0.02)
        lambda parameter used to compute the elastic energy.
    epg_mu: `float`, optional (default: 0.1)
        mu parameter used to compute the elastic energy.
    epg_trimmingradius: `float`, optional (default: 'Inf')  
        maximal distance from a node to the points it controls in the embedding.
    epg_finalenergy: `str`, optional (default: 'Penalized')
        indicate the final elastic energy associated with the configuration.
    epg_alpha: `float`, optional (default: 0.02)
        alpha parameter of the penalized elastic energy.
    epg_beta: `float`, optional (default: 0.0)
        beta parameter of the penalized elastic energy.
    epg_n_processes: `int`, optional (default: 1)
        The number of processes to use.
    reset: `bool`, optional (default: False)
        If true, reset the current elastic principal graph to the initial elastic principal graph (i.e. the graph obtained from running 'elastic_principal_graph')
    **kwargs: additional arguments to `ElPiGraph.CollapseBrances`
   
    Returns
    -------
    updates `adata` with the following fields.

    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.        
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['epg_obj']`)
        R object of elastic principal graph learning.
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    """
    
    print('Shifting branching point to denser area ...')
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    if(reset):
        epg_obj = adata.uns['ori_epg_obj']
        epg = adata.uns['ori_epg']
    else:
        epg_obj = adata.uns['epg_obj']
        epg = adata.uns['epg']
    input_data = adata.obsm['X_dr']

    epg_obj_shift = ElPiGraph.ShiftBranching(X = input_data, 
                                           TargetPG = epg_obj[0], 
                                           TrimmingRadius = epg_trimmingradius,                       
                                           SelectionMode = epg_shift_mode, 
                                           DensityRadius = epg_shift_radius,
                                           MaxShift = epg_shift_max,
                                           **kwargs)

    init_nodes_pos = np.array(epg_obj_shift.rx2('NodePositions'))
    init_edges = np.array(epg_obj_shift.rx2('Edges')) - 1  
    epg_n_nodes = init_nodes_pos.shape[0]
 
    epg_obj = ElPiGraph.computeElasticPrincipalTree(X=input_data,
                                                    NumNodes = epg_n_nodes, 
                                                    InitNodePositions = init_nodes_pos,
                                                    InitEdges=init_edges + 1,
                                                    Lambda=epg_lambda, Mu=epg_mu,
                                                    TrimmingRadius= epg_trimmingradius,
                                                    FinalEnergy = epg_finalenergy,
                                                    alpha = epg_alpha,
                                                    beta = epg_beta,                                                    
                                                    Do_PCA=False,CenterData=False,
                                                    drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                                    n_cores = epg_n_processes,
                                                    nReps=1,
                                                    ProbPoint=1.0)
    
    epg_nodes_pos = np.array(epg_obj[0].rx2('NodePositions'))
    epg_edges = np.array((epg_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1   
                    
    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['epg_obj'] = deepcopy(epg_obj)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after shifting branching: ' + str(len(dict_branches)))


def extend_elastic_principal_graph(adata,epg_ext_mode = 'QuantDists',epg_ext_par = 0.5,epg_trimmingradius='Inf',reset=False,**kwargs):
    """Extend the leaves of elastic principal graph with additional nodes.
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    epg_ext_mode: `str`, optional (default: 'QuantDists')
        The mode used to extend the leaves.
        Choose from {{'QuantDists','QuantCentroid','WeigthedCentroid'}}
        'QuantCentroid':for each leaf node, the extreme points are ordered by their distance from the node and the centroid of the points further than epg_ext_par is returned.
        'WeigthedCentroid':for each leaf node, a weight is computed for each points by raising the distance to the epg_ext_par power. Larger epg_ext_par results in a bigger influence of points further than the node
        'QuantDists':for each leaf node, the extreme points are ordered by their distance from the node and the 100*epg_ext_par th percentile of the points farther than epg_ext_par is returned
    epg_ext_par: `float`, optional (default: 0.5)
        The paramter used to control different modes.
    epg_trimmingradius: `float`, optional (default: 'Inf')  
        maximal distance from a node to the points it controls in the embedding.
    reset: `bool`, optional (default: False)
        If true, reset the current elastic principal graph to the initial elastic principal graph (i.e. the graph obtained from running 'elastic_principal_graph')
    **kwargs: additional arguments to `ElPiGraph.CollapseBrances`
   
    Returns
    -------
    updates `adata` with the following fields.

    adata.obs: `pandas.core.frame.DataFrame` (`adata.obs`)
        Update adata.obs with adding the columns of current root_node_pseudotime and removing the previous ones.        
    epg : `networkx.classes.graph.Graph` (`adata.uns['epg']`)
        Elastic principal graph structure. It contains node attributes ('pos')
    epg_obj : `rpy2.rinterface.ListSexpVector` (`adata.uns['epg_obj']`)
        R object of elastic principal graph learning.
    flat_tree : `networkx.classes.graph.Graph` (`adata.uns['flat_tree']`)
        An abstract of elastic principle graph structure by only keeping leaf nodes and branching nodes. 
        It contains node attribtutes ('pos','label') and edge attributes ('nodes','id','len','color').
    """

    print('Extending leaves with additional nodes ...')
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    if(reset):
        epg_obj = adata.uns['ori_epg_obj']
        epg = adata.uns['ori_epg']
    else:
        epg_obj = adata.uns['epg_obj']
        epg = adata.uns['epg']
    input_data = adata.obsm['X_dr']

    epg_obj_extend = ElPiGraph.ExtendLeaves(X = input_data, 
                                          TargetPG = epg_obj[0],
                                          TrimmingRadius = epg_trimmingradius,
                                          Mode = epg_ext_mode, 
                                          ControlPar = epg_ext_par,
                                          PlotSelected = False,
                                          **kwargs)
    epg_nodes_pos = np.array(epg_obj_extend.rx2('NodePositions'))
    epg_edges = np.array((epg_obj_extend.rx2('Edges')).rx2('Edges'),dtype=int)-1   
        
    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns['epg'] = deepcopy(epg)
    adata.uns['flat_tree'] = deepcopy(flat_tree)
#     adata.uns['epg_obj'] = deepcopy(epg_obj_extend)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after extending leaves: ' + str(len(dict_branches)))    


def plot_flat_tree(adata,adata_new=None,show_all_cells=True,save_fig=False,fig_path=None,fig_name='flat_tree.pdf',fig_size=(8,8),fig_legend_ncol=3):  
    """Plot flat tree based on a modified version of the force-directed layout Fruchterman-Reingold algorithm.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    show_all_cells: `bool`, optional (default: False)
        if show_all_cells is True and adata_new is speicified, both original cells and mapped cells will be shown
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'flat_tree.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.

    Returns
    -------
    updates `adata` with the following fields.
    X_spring: `numpy.ndarray` (`adata.obsm['X_spring']`)
        Store #observations × 2 coordinates of cells in flat tree.

    updates `adata_new` with the following fields.
    X_spring: `numpy.ndarray` (`adata_new.obsm['X_spring']`)
        Store #observations × 2 coordinates of new cells in flat tree.
    """

    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']

    flat_tree = adata.uns['flat_tree']
    dict_nodes_pos = nx.spring_layout(flat_tree,random_state=10)
    bfs_root = list(flat_tree.nodes())[0]
    bfs_edges = list(nx.bfs_edges(flat_tree, bfs_root))
    bfs_nodes = [bfs_root] + [v for u, v in bfs_edges]

    ## Update the positions of flat tree's nodes
    dict_nodes_pos_updated = deepcopy(dict_nodes_pos)
    flat_tree_copy = deepcopy(flat_tree)
    flat_tree_copy.remove_node(bfs_root)
    for i,edge_i in enumerate(bfs_edges):
        dist_nodes = distance.euclidean(dict_nodes_pos_updated[edge_i[0]],dict_nodes_pos_updated[edge_i[1]])
        len_edge = flat_tree.edges[edge_i]['len']
        st_x = dict_nodes_pos_updated[edge_i[0]][0]
        ed_x = dict_nodes_pos_updated[edge_i[1]][0]
        st_y = dict_nodes_pos_updated[edge_i[0]][1]
        ed_y = dict_nodes_pos_updated[edge_i[1]][1]
        p_x = st_x + (ed_x - st_x)*(len_edge/dist_nodes)
        p_y = st_y + (ed_y - st_y)*(len_edge/dist_nodes)
        dict_nodes_pos_updated[edge_i[1]] = np.array([p_x,p_y])

        con_components = list(nx.connected_components(flat_tree_copy))
        #update other reachable unvisited nodes
        for con_comp in con_components:
            if edge_i[1] in con_comp:
                reachable_unvisited = con_comp - {edge_i[1]}
                flat_tree_copy.remove_node(edge_i[1])
                break
        for nd in reachable_unvisited:
            nd_x = dict_nodes_pos_updated[nd][0] + p_x - ed_x
            nd_y = dict_nodes_pos_updated[nd][1] + p_y - ed_y
            dict_nodes_pos_updated[nd] = np.array([nd_x,nd_y])

    nx.set_node_attributes(flat_tree, values=dict_nodes_pos_updated,name='pos_spring')

    ## Update the positions of cells
    cells_pos = np.empty([adata.shape[0],2])
    list_branch_id = nx.get_edge_attributes(flat_tree,'id').values()
    for br_id in list_branch_id:
        s_pos = dict_nodes_pos_updated[br_id[0]] #start node position
        e_pos = dict_nodes_pos_updated[br_id[1]] #end node position
        dist_se = distance.euclidean(s_pos,e_pos)
        p_x = np.array(adata.obs[adata.obs['branch_id'] == br_id]['branch_lam'].tolist())
        dist_p = np.array(adata.obs[adata.obs['branch_id'] == br_id]['branch_dist'].tolist())
        np.random.seed(100)
        p_y = np.random.choice([1,-1],size=len(p_x))*dist_p
        #rotation matrix
        ro_angle = np.arctan2((e_pos-s_pos)[1],(e_pos-s_pos)[0])#counterclockwise angle
        p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y*math.sin(ro_angle)
        p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y*math.cos(ro_angle)
        p_pos = np.array((p_x_prime,p_y_prime)).T
        cells_pos[np.where(adata.obs['branch_id'] == br_id)[0],:] =[p_pos[i,:].tolist() for i in range(p_pos.shape[0])]
    adata.obsm['X_spring'] = cells_pos
    
    if(adata_new !=None):
        cells_pos = np.empty([adata_new.shape[0],2])
        list_branch_id = nx.get_edge_attributes(flat_tree,'id').values()
        for br_id in adata_new.obs['branch_id'].unique():
            s_pos = dict_nodes_pos_updated[br_id[0]] #start node position
            e_pos = dict_nodes_pos_updated[br_id[1]] #end node position
            dist_se = distance.euclidean(s_pos,e_pos)
            p_x = np.array(adata_new.obs[adata_new.obs['branch_id'] == br_id]['branch_lam'].tolist())
            dist_p = np.array(adata_new.obs[adata_new.obs['branch_id'] == br_id]['branch_dist'].tolist())
            np.random.seed(100)
            p_y = np.random.choice([1,-1],size=len(p_x))*dist_p
            #rotation matrix
            ro_angle = np.arctan2((e_pos-s_pos)[1],(e_pos-s_pos)[0])#counterclockwise angle
            p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y*math.sin(ro_angle)
            p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y*math.cos(ro_angle)
            p_pos = np.array((p_x_prime,p_y_prime)).T
            cells_pos[np.where(adata_new.obs['branch_id'] == br_id)[0],:] =[p_pos[i,:].tolist() for i in range(p_pos.shape[0])]
        adata_new.obsm['X_spring'] = cells_pos             

    ##plotting
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
    edges = flat_tree.edges()
    edge_color = [flat_tree[u][v]['color'] for u,v in edges]
    nx.draw_networkx(flat_tree,
                     pos=nx.get_node_attributes(flat_tree,'pos_spring'),
                     labels=nx.get_node_attributes(flat_tree,'label'),
                     edges=edges, 
                     edge_color=edge_color, 
                     node_color='white',alpha=1,width = 6,node_size=0,font_size=15)
    df_sample = adata.obs[['label','label_color']].copy()
    df_coord = pd.DataFrame(adata.obsm['X_spring'],index=adata.obs_names)
    list_patches = []
    for x in adata.uns['label_color'].keys():
        list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
    color = df_sample.sample(frac=1,random_state=100)['label_color'] 
    coord = df_coord.sample(frac=1,random_state=100)
    if(adata_new !=None):
        df_sample_new = adata_new.obs[['label','label_color']].copy()
        df_coord_new = pd.DataFrame(adata_new.obsm['X_spring'],index=adata_new.obs_names)
        color_new = df_sample_new.sample(frac=1,random_state=100)['label_color'] 
        coord_new = df_coord_new.sample(frac=1,random_state=100)  
        for x in adata_new.uns['label_color'].keys():
            list_patches.append(Patches.Patch(color = adata_new.uns['label_color'][x],label=x))
        if(show_all_cells):
            ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8) 
            ax.scatter(coord_new[0], coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8)
        else:
            ax.scatter(coord_new[0], coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8) 
    else:
        ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8) 
    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.15),
              ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
    if(save_fig):
        plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
        plt.close(fig) 

def plot_visualization_2D(adata,adata_new=None,show_all_colors=False,method='umap',nb_pct=0.1,perplexity=30.0,color_by='label',use_precomputed=True,
                          save_fig=False,fig_path=None,fig_name='visualization_2D.pdf',fig_size=(10,10),fig_legend_ncol=3):  

    """ Visualize the results in 2D plane
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    show_all_colors: `bool`, optional (default: False)
        if show_all_colors is True and adata_new is speicified, original cells will be colored based on its color file. Otherwise, the original cells won't be distinguished and will be colored with 'grey' 
    method: `str`, optional (default: 'umap')
        Choose from {{'umap','tsne'}}
        Method used for visualization.
        'umap': Uniform Manifold Approximation and Projection      
        'tsne': t-Distributed Stochastic Neighbor Embedding
    nb_pct: `float`, optional (default: 0.1)
        The percentage neighbor cells (only valid when 'umap' is specified). 
    perplexity: `float`, optional (default: 30.0)
        The perplexity used for tSNE.       
    color_by: `str`, optional (default: 'label')
        Choose from {{'label','branch'}}
        Specify how to color cells.
        'label': the cell labels (stored in adata.obs['label'])
        'branch': the bracnh id identifed by STREAM
    use_precomputed: `bool`, optional (default: True)
        If True, the visualization coordinates from previous computing will be used
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (10,10))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'visualization_2D.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.

    Returns
    -------
    updates `adata` with the following fields. (Depending on `method`)
    X_vis_umap: `numpy.ndarray` (`adata.obsm['X_vis_umap']`)
        Store #observations × 2 umap data matrix. 
    X_vis_tsne: `numpy.ndarray` (`adata.obsm['X_vis_tsne']`)
        Store #observations × 2 tsne data matrix.     
    
    updates `adata_new` with the following fields. (Depending on `method`)
    merged_X_vis_umap: `numpy.ndarray` (`adata_new.uns['merged_X_vis_umap']`)
        Store the umap data matrix after merging original cells and new cells to be mapped.
    merged_X_vis_tsne: `numpy.ndarray` (`adata_new.uns['merged_X_vis_tsne']`)
        Store the tsne data matrix after merging original cells and new cells to be mapped.

    """


    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']
    input_data = adata.obsm['X_dr']
    if(adata_new != None):
        input_data = np.vstack((input_data,adata_new.obsm['X_dr']))
    if(method == 'umap'):
        if(adata_new is None):
            if(use_precomputed and ('X_vis_umap' in adata.obsm_keys())):
                print('Importing precomputed umap visualization ...')
                embedding = adata.obsm['X_vis_umap']
            else:
                reducer = umap.UMAP(n_neighbors=int(input_data.shape[0]*nb_pct),n_components=2,random_state=42)
                embedding = reducer.fit_transform(input_data)
                adata.obsm['X_vis_umap'] = embedding
        else:
            if(use_precomputed and ('merged_X_vis_umap' in adata_new.uns_keys())):
                print('Importing precomputed umap visualization ...')
                embedding = adata_new.uns['merged_X_vis_umap']
            else:
                reducer = umap.UMAP(n_neighbors=int(input_data.shape[0]*nb_pct),n_components=2,random_state=42)
                embedding = reducer.fit_transform(input_data)  
                adata_new.uns['merged_X_vis_umap'] = embedding

    if(method == 'tsne'):
        if(adata_new is None):
            if(use_precomputed and ('X_vis_tsne' in adata.obsm_keys())):
                print('Importing precomputed tsne visualization ...')
                embedding = adata.obsm['X_vis_tsne']
            else:
                reducer = TSNE(n_components=2, init='pca',perplexity=perplexity, random_state=0)
                embedding = reducer.fit_transform(input_data)
                adata.obsm['X_vis_tsne'] = embedding
        else:
            if(use_precomputed and ('merged_X_vis_tsne' in adata_new.uns_keys())):
                print('Importing precomputed tsne visualization ...')
                embedding = adata_new.uns['X_vis_tsne']
            else:
                reducer = TSNE(n_components=2, init='pca',perplexity=perplexity, random_state=0)
                embedding = reducer.fit_transform(input_data)
                adata_new.uns['merged_X_vis_tsne'] = embedding            
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1,1,1)        
    if(adata_new is None):
        df_sample = adata.obs.copy()
        df_coord = pd.DataFrame(embedding,index=adata.obs_names)
        if(color_by=='label'):
            list_patches = []
            for x in adata.uns['label_color'].keys():
                list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
            color = df_sample.sample(frac=1,random_state=100)['label_color'] 
            coord = df_coord.sample(frac=1,random_state=100)    
        if(color_by=='branch'):
            df_sample = adata.obs.copy()
            df_coord = pd.DataFrame(embedding,index=adata.obs_names)
            flat_tree = adata.uns['flat_tree']
            list_patches = []
            df_sample['branch_color'] = '' 
            for edge in flat_tree.edges():
                br_id = flat_tree.edges[edge]['id']
                id_cells = np.where(df_sample['branch_id']==br_id)[0]
                df_sample.loc[df_sample.index[id_cells],'branch_color'] = flat_tree.edges[edge]['color']
                list_patches.append(Patches.Patch(color = flat_tree.edges[edge]['color'],
                    label='branch '+flat_tree.nodes[br_id[0]]['label']+'_'+flat_tree.nodes[br_id[1]]['label']))
            color = df_sample.sample(frac=1,random_state=100)['branch_color'] 
            coord = df_coord.sample(frac=1,random_state=100) 
        ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8)   
    else:
        if(color_by=='label'):  
            df_sample = adata.obs.copy()
            df_coord = pd.DataFrame(embedding[:adata.shape[0],:],index=adata.obs_names)
            color = df_sample.sample(frac=1,random_state=100)['label_color'] 
            coord = df_coord.sample(frac=1,random_state=100)    
            df_sample_new = adata_new.obs.copy()
            df_coord_new = pd.DataFrame(embedding[adata.shape[0]:embedding.shape[0],:],index=adata_new.obs_names)
            color_new = df_sample_new.sample(frac=1,random_state=100)['label_color'] 
            coord_new = df_coord_new.sample(frac=1,random_state=100)         
            if(show_all_colors):
                list_patches = []
                for x in adata.uns['label_color'].keys():
                    list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))            
                ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8) 
                for x in adata_new.uns['label_color'].keys():
                    if(x not in adata.uns['label_color'].keys()):
                        list_patches.append(Patches.Patch(color = adata_new.uns['label_color'][x],label=x))            
                ax.scatter(coord_new[0],coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8)
            else:
                ax.scatter(coord[0], coord[1],c='gray',s=50,linewidth=0,alpha=0.8) 
                list_patches = [Patches.Patch(color = 'gray',label='trajectory_cells')]
                for x in adata_new.uns['label_color'].keys():
                    if(x not in adata.uns['label_color'].keys()):
                        list_patches.append(Patches.Patch(color = adata_new.uns['label_color'][x],label=x))            
                ax.scatter(coord_new[0],coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8)  
        if(color_by=='branch'):
            df_sample = adata.obs.copy()
            df_coord = pd.DataFrame(embedding[:adata.shape[0],:],index=adata.obs_names)
            flat_tree = adata.uns['flat_tree']
            list_patches = []
            df_sample['branch_color'] = '' 
            for edge in flat_tree.edges():
                br_id = flat_tree.edges[edge]['id']
                id_cells = np.where(df_sample['branch_id']==br_id)[0]
                df_sample.loc[df_sample.index[id_cells],'branch_color'] = flat_tree.edges[edge]['color']
                list_patches.append(Patches.Patch(color = flat_tree.edges[edge]['color'],
                    label='branch '+flat_tree.nodes[br_id[0]]['label']+'_'+flat_tree.nodes[br_id[1]]['label']))
            color = df_sample.sample(frac=1,random_state=100)['branch_color'] 
            coord = df_coord.sample(frac=1,random_state=100)   
            df_sample_new = adata_new.obs.copy()
            df_coord_new = pd.DataFrame(embedding[adata.shape[0]:embedding.shape[0],:],index=adata_new.obs_names)
            df_sample_new['branch_color'] = '' 
            for edge in flat_tree.edges():
                br_id = flat_tree.edges[edge]['id']
                id_cells = np.where(df_sample_new['branch_id']==br_id)[0]
                df_sample_new.loc[df_sample_new.index[id_cells],'branch_color'] = flat_tree.edges[edge]['color']  
            color_new = df_sample_new.sample(frac=1,random_state=100)['branch_color'] 
            coord_new = df_coord_new.sample(frac=1,random_state=100)               
            ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8)           
            ax.scatter(coord_new[0],coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8)                    
    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.1),
              ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
    if(save_fig):
        plt.savefig(os.path.join(fig_path,fig_name),pad_inches=1,bbox_inches='tight')
        plt.close(fig) 


def subwaymap_plot(adata,adata_new=None,show_all_cells=True,root='S0',percentile_dist=98,factor=2.0,color_by='label',preference=None,
                   save_fig=False,fig_path=None,fig_name='subway_map.pdf',fig_size=(10,6),fig_legend_ncol=3):  
    """Generate subway map plots
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    show_all_cells: `bool`, optional (default: False)
        if show_all_cells is True and adata_new is speicified, both original cells and mapped cells will be shown
    root: `str`, optional (default: 'S0'): 
        The starting node
    percentile_dist: `int`, optional (default: 98)
        Percentile of cells' distances from branches (between 0 and 100) used for calculating the distances between branches of subway map.
    factor: `float`, optional (default: 2.0)
        The factor used to adjust the distances between branches of subway map.
    color_by: `str`, optional (default: 'label')
        Choose from {{'label','branch'}}
        Specify how to color cells.
        'label': the cell labels (stored in adata.obs['label'])
        'branch': the bracnh id identifed by STREAM
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot. The higher ranks the node have, the closer to the top the branch with that node is.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'subway_map.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.

    Returns
    -------
    updates `adata` with the following fields.
    X_subwaymap_root: `numpy.ndarray` (`adata.obsm['X_subwaymap_root']`)
        Store #observations × 2 coordinates of cells in subwaymap plot.
    subwaymap_root: `dict` (`adata.uns['subwaymap_root']`)
        Store the coordinates of nodes ('nodes') and edges ('edges') in subwaymap plot.

    updates `adata_new` with the following fields.
    X_subwaymap_root: `numpy.ndarray` (`adata_new.obsm['X_subwaymap_root']`)
        Store #observations × 2 coordinates of new cells in subwaymap plot.
    """


    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']

    flat_tree = adata.uns['flat_tree']
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(root not in dict_label_node.keys()):
        print('There is no root '+root)
    else:
        file_path_S = os.path.join(fig_path,root)
        if(not os.path.exists(file_path_S)):
            os.makedirs(file_path_S)   
        root_node = dict_label_node[root]
        dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree,root_node))
        dict_bfs_suc = dict(nx.bfs_successors(flat_tree,root_node))
        dict_edge_shift_dist = calculate_shift_distance(adata,root=root,percentile=percentile_dist,factor=factor,preference=preference)
        dict_path_len = nx.shortest_path_length(flat_tree,source=root_node,weight='len')
        df_cells_pos = pd.DataFrame(index=adata.obs.index,columns=['cells_pos'])
        dict_edges_pos = {}
        dict_nodes_pos = {}
        for edge in dict_edge_shift_dist.keys():
            node_pos_st = np.array([dict_path_len[edge[0]],dict_edge_shift_dist[edge]])
            node_pos_ed = np.array([dict_path_len[edge[1]],dict_edge_shift_dist[edge]])  
            br_id = flat_tree.edges[edge]['id']
            id_cells = np.where(adata.obs['branch_id']==br_id)[0]
            # cells_pos_x = flat_tree.nodes[root_node]['pseudotime'].iloc[id_cells]
            cells_pos_x = adata.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
            np.random.seed(100)
            cells_pos_y = node_pos_st[1] + adata.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
            cells_pos = np.array((cells_pos_x,cells_pos_y)).T
            df_cells_pos.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
            dict_edges_pos[edge] = np.array([node_pos_st,node_pos_ed])    
            if(edge[0] not in dict_bfs_pre.keys()):
                dict_nodes_pos[edge[0]] = node_pos_st
            dict_nodes_pos[edge[1]] = node_pos_ed
        adata.obsm['X_subwaymap_'+root] = np.array(df_cells_pos['cells_pos'].tolist())
        
        if(adata_new!=None):
            df_cells_pos_new = pd.DataFrame(index=adata_new.obs.index,columns=['cells_pos'])
            for edge in dict_edge_shift_dist.keys():       
                node_pos_st = np.array([dict_path_len[edge[0]],dict_edge_shift_dist[edge]])
                node_pos_ed = np.array([dict_path_len[edge[1]],dict_edge_shift_dist[edge]])  
                br_id = flat_tree.edges[edge]['id']
                list_br_id_new = adata_new.obs['branch_id'].unique().tolist()
                flat_tree_new = adata_new.uns['flat_tree']
                if(br_id in list_br_id_new):
                    id_cells = np.where(adata_new.obs['branch_id']==br_id)[0]
                    # cells_pos_x = flat_tree_new.nodes[root_node]['pseudotime'].iloc[id_cells]
                    cells_pos_x = adata_new.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
                    np.random.seed(100)
                    cells_pos_y = node_pos_st[1] + adata_new.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
                    cells_pos = np.array((cells_pos_x,cells_pos_y)).T
                    df_cells_pos_new.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]  
            adata_new.obsm['X_subwaymap_'+root] = np.array(df_cells_pos_new['cells_pos'].tolist())
            
        if(flat_tree.degree(root_node)>1):
            suc_nodes = dict_bfs_suc[root_node]
            edges = [(root_node,sn) for sn in suc_nodes]
            max_y_pos = max([dict_edges_pos[x][0,1] for x in edges])
            min_y_pos = min([dict_edges_pos[x][0,1] for x in edges])
            median_y_pos = np.median([dict_edges_pos[x][0,1] for x in edges])
            x_pos = dict_edges_pos[edges[0]][0,0]
            dict_nodes_pos[root_node] = np.array([x_pos,median_y_pos])
            
        adata.uns['subwaymap_'+root] = dict()
        adata.uns['subwaymap_'+root]['nodes'] = dict_nodes_pos
        adata.uns['subwaymap_'+root]['edges'] = dict()

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1,1,1)
        legend_labels = []
        for edge in dict_edges_pos.keys():  
            edge_pos = dict_edges_pos[edge]
            edge_color = flat_tree.edges[edge]['color']
            ax.plot(edge_pos[:,0],edge_pos[:,1],c=edge_color,alpha=1,lw=5,zorder=None)
            if(edge[0] in dict_bfs_pre.keys()):
                pre_node = dict_bfs_pre[edge[0]]
                link_edge_pos = np.array([dict_edges_pos[(pre_node,edge[0])][1,],dict_edges_pos[edge][0,]])
                ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
                edge_pos = np.vstack((link_edge_pos,edge_pos))
            adata.uns['subwaymap_'+root]['edges'][edge]=edge_pos
        if(flat_tree.degree(root_node)>1):
            suc_nodes = dict_bfs_suc[root_node]
            edges = [(root_node,sn) for sn in suc_nodes]
            max_y_pos = max([dict_edges_pos[x][0,1] for x in edges])
            min_y_pos = min([dict_edges_pos[x][0,1] for x in edges])
            x_pos = dict_nodes_pos[root_node][0]
            link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
            ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
            adata.uns['subwaymap_'+root]['edges'][(root_node,root_node)]=link_edge_pos
        for node_i in flat_tree.nodes():
            ax.text(dict_nodes_pos[node_i][0],dict_nodes_pos[node_i][1],
                    flat_tree.nodes[node_i]['label'],color='black',
                    fontsize = 15,horizontalalignment='center',verticalalignment='center',zorder=20)    

        if(color_by=='label'):
            list_patches = []
            df_sample = adata.obs[['label','label_color']].copy()
            df_coord = pd.DataFrame(df_cells_pos['cells_pos'].tolist(),index=adata.obs_names)
            for x in adata.uns['label_color'].keys():
                list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
            color = df_sample.sample(frac=1,random_state=100)['label_color'] 
            coord = df_coord.sample(frac=1,random_state=100)
            if(adata_new!=None):
                if(not show_all_cells):
                    list_patches = []
                for x in adata_new.uns['label_color'].keys():
                    list_patches.append(Patches.Patch(color = adata_new.uns['label_color'][x],label=x))
                df_sample_new = adata_new.obs[['label','label_color']].copy()
                df_coord_new = pd.DataFrame(df_cells_pos_new['cells_pos'].tolist(),index=adata_new.obs_names)
                color_new = df_sample_new.sample(frac=1,random_state=100)['label_color'] 
                coord_new = df_coord_new.sample(frac=1,random_state=100)                
        if(color_by=='branch'):
            list_patches = []
            df_sample = adata.obs.copy()
            df_coord = pd.DataFrame(df_cells_pos['cells_pos'].tolist(),index=adata.obs_names)
            df_sample['branch_color'] = '' 
            for edge in flat_tree.edges():
                br_id = flat_tree.edges[edge]['id']
                id_cells = np.where(df_sample['branch_id']==br_id)[0]
                df_sample.loc[df_sample.index[id_cells],'branch_color'] = flat_tree.edges[edge]['color']
                list_patches.append(Patches.Patch(color = flat_tree.edges[edge]['color'],
                    label='branch '+flat_tree.nodes[br_id[0]]['label']+'_'+flat_tree.nodes[br_id[1]]['label']))
            color = df_sample.sample(frac=1,random_state=100)['branch_color'] 
            coord = df_coord.sample(frac=1,random_state=100) 
            if(adata_new!=None):
                df_sample_new = adata_new.obs.copy()
                df_coord_new = pd.DataFrame(df_cells_pos_new['cells_pos'].tolist(),index=adata_new.obs_names)
                df_sample_new['branch_color'] = '' 
                for edge in flat_tree.edges():
                    br_id = flat_tree.edges[edge]['id']
                    if(br_id in list_br_id_new):
                        id_cells = np.where(df_sample_new['branch_id']==br_id)[0]
                        df_sample_new.loc[df_sample_new.index[id_cells],'branch_color'] = flat_tree.edges[edge]['color']
                color_new = df_sample_new.sample(frac=1,random_state=100)['branch_color'] 
                coord_new = df_coord_new.sample(frac=1,random_state=100) 
        if(adata_new is None):   
            ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8,zorder=10) 
        else:
            if(show_all_cells):
                ax.scatter(coord[0], coord[1],c=color,s=50,linewidth=0,alpha=0.8,zorder=10) 
                ax.scatter(coord_new[0], coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8,zorder=10) 
            else:
                ax.scatter(coord_new[0], coord_new[1],c=color_new,s=50,linewidth=0,alpha=0.8,zorder=10)
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.1),
                  ncol=fig_legend_ncol, fancybox=True, shadow=True,markerscale=2.5)
        ax.set_xlabel('pseudotime')
        if(save_fig):
            plt.savefig(os.path.join(file_path_S,fig_name), pad_inches=1,bbox_inches='tight')
            plt.close(fig)


def subwaymap_plot_gene(adata,adata_new=None,show_all_cells=True,genes=None,root='S0',percentile_dist=98,percentile_expr=95,factor=2.0,preference=None,
                        save_fig=False,fig_path=None,fig_format='pdf',fig_size=(10,6)):  
    """Generate subway map plots of genes
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    show_all_cells: `bool`, optional (default: False)
        if show_all_cells is True and adata_new is speicified, both original cells and mapped cells will be shown
    genes: `list`, optional (default: None): 
        A list of genes to be displayed
    root: `str`, optional (default: 'S0'): 
        The starting node
    percentile_dist: `int`, optional (default: 98)
        Percentile of cells' distances from branches (between 0 and 100) used for calculating the distances between branches of subway map.
    factor: `float`, optional (default: 2.0)
        The factor used to adjust the distances between branches of subway map.
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot. The higher ranks the node have, the closer to the top the branch with that node is.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.


    Returns
    -------
    None

    """

    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']
    if(genes is None):
        print('Please provide gene names');
    else:
        experiment = adata.uns['experiment']
        genes = np.unique(genes).tolist()
        for x in genes:
            if x not in adata.var_names:
                print(x + ' is not in the gene expression matrix')
                return    
        df_gene_expr = pd.DataFrame(index= adata.obs_names.tolist(),
                                    data = adata.raw[:,genes].X,
                                    columns=genes)
        if(adata_new!=None):
            df_gene_expr_new = pd.DataFrame(index= adata_new.obs_names.tolist(),
                                        data = adata_new.raw[:,genes].X,
                                        columns=genes)              
        flat_tree = adata.uns['flat_tree']
        dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
        if(root not in dict_label_node.keys()):
            print('there is no root '+root)
        else:
            file_path_S = os.path.join(fig_path,root)
            if(not os.path.exists(file_path_S)):
                os.makedirs(file_path_S)   
            root_node = dict_label_node[root]
            dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree,root_node))
            dict_bfs_suc = dict(nx.bfs_successors(flat_tree,root_node))
            dict_edge_shift_dist = calculate_shift_distance(adata,root=root,percentile=percentile_dist,factor=factor,preference=preference)
            dict_path_len = nx.shortest_path_length(flat_tree,source=root_node,weight='len')
            df_cells_pos = pd.DataFrame(index=adata.obs.index,columns=['cells_pos'])
            dict_edges_pos = {}
            dict_nodes_pos = {}
            for edge in dict_edge_shift_dist.keys():
                node_pos_st = np.array([dict_path_len[edge[0]],dict_edge_shift_dist[edge]])
                node_pos_ed = np.array([dict_path_len[edge[1]],dict_edge_shift_dist[edge]])  
                br_id = flat_tree.edges[edge]['id']
                id_cells = np.where(adata.obs['branch_id']==br_id)[0]
                # cells_pos_x = flat_tree.nodes[root_node]['pseudotime'].iloc[id_cells]
                cells_pos_x = adata.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + adata.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
                cells_pos = np.array((cells_pos_x,cells_pos_y)).T
                df_cells_pos.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
                dict_edges_pos[edge] = np.array([node_pos_st,node_pos_ed])    
                if(edge[0] not in dict_bfs_pre.keys()):
                    dict_nodes_pos[edge[0]] = node_pos_st
                dict_nodes_pos[edge[1]] = node_pos_ed 
            if(adata_new!=None):
                df_cells_pos_new = pd.DataFrame(index=adata_new.obs.index,columns=['cells_pos'])
                for edge in dict_edge_shift_dist.keys():       
                    node_pos_st = np.array([dict_path_len[edge[0]],dict_edge_shift_dist[edge]])
                    node_pos_ed = np.array([dict_path_len[edge[1]],dict_edge_shift_dist[edge]])  
                    br_id = flat_tree.edges[edge]['id']
                    list_br_id_new = adata_new.obs['branch_id'].unique().tolist()
                    flat_tree_new = adata_new.uns['flat_tree']
                    if(br_id in list_br_id_new):
                        id_cells = np.where(adata_new.obs['branch_id']==br_id)[0]
                        # cells_pos_x = flat_tree_new.nodes[root_node]['pseudotime'].iloc[id_cells]
                        cells_pos_x = adata_new.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
                        np.random.seed(100)
                        cells_pos_y = node_pos_st[1] + adata_new.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
                        cells_pos = np.array((cells_pos_x,cells_pos_y)).T
                        df_cells_pos_new.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]                 
            if(flat_tree.degree(root_node)>1):
                suc_nodes = dict_bfs_suc[root_node]
                edges = [(root_node,sn) for sn in suc_nodes]
                max_y_pos = max([dict_edges_pos[x][0,1] for x in edges])
                min_y_pos = min([dict_edges_pos[x][0,1] for x in edges])
                median_y_pos = np.median([dict_edges_pos[x][0,1] for x in edges])
                x_pos = dict_edges_pos[edges[0]][0,0]
                dict_nodes_pos[root_node] = np.array([x_pos,median_y_pos])
        cm = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))      
        for g in genes:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1,1,1)
            for edge in dict_edges_pos.keys():  
                edge_pos = dict_edges_pos[edge]
                edge_color = flat_tree.edges[edge]['color']
                ax.plot(edge_pos[:,0],edge_pos[:,1],c='gray',alpha=1,lw=5,zorder=None)
                if(edge[0] in dict_bfs_pre.keys()):
                    pre_node = dict_bfs_pre[edge[0]]
                    link_edge_pos = np.array([dict_edges_pos[(pre_node,edge[0])][1,],dict_edges_pos[edge][0,]])
                    ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
            if(flat_tree.degree(root_node)>1):
                suc_nodes = dict_bfs_suc[root_node]
                edges = [(root_node,sn) for sn in suc_nodes]
                max_y_pos = max([dict_edges_pos[x][0,1] for x in edges])
                min_y_pos = min([dict_edges_pos[x][0,1] for x in edges])
                x_pos = dict_nodes_pos[root_node][0]
                link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
                ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)

            for node_i in flat_tree.nodes():
                ax.text(dict_nodes_pos[node_i][0],dict_nodes_pos[node_i][1],
                        flat_tree.nodes[node_i]['label'],color='black',
                        fontsize = 15,horizontalalignment='center',verticalalignment='center',zorder=20)  
            if(experiment=='rna-seq'):
                gene_expr = df_gene_expr[g].copy()
                max_gene_expr = np.percentile(gene_expr[gene_expr>0],percentile_expr)
                gene_expr[gene_expr>max_gene_expr] = max_gene_expr   
                vmin = 0
                vmax = max_gene_expr
            elif(experiment=='atac-seq'):
                gene_expr = df_gene_expr[g].copy()
                min_gene_expr = np.percentile(gene_expr[gene_expr<0],100-percentile_expr)
                max_gene_expr = np.percentile(gene_expr[gene_expr>0],percentile_expr)
                gene_expr[gene_expr>max_gene_expr] = max_gene_expr
                gene_expr[gene_expr<min_gene_expr] = min_gene_expr
                vmin = -max(abs(min_gene_expr),max_gene_expr)
                vmax = max(abs(min_gene_expr),max_gene_expr)
            else:
                print('The experiment '+experiment +' is not supported')
                return
            df_coord = pd.DataFrame(df_cells_pos['cells_pos'].tolist(),index=adata.obs_names)
            color = pd.Series(gene_expr).sample(frac=1,random_state=100)
            coord = df_coord.sample(frac=1,random_state=100)
            if(adata_new!=None):
                if(experiment=='rna-seq'):
                    gene_expr_new = df_gene_expr_new[g].copy()
                    max_gene_expr_new = np.percentile(gene_expr_new[gene_expr_new>0],percentile_expr)
                    gene_expr_new[gene_expr_new>max_gene_expr_new] = max_gene_expr_new   
                elif(experiment=='atac-seq'):
                    gene_expr_new = df_gene_expr_new[g].copy()
                    min_gene_expr_new = np.percentile(gene_expr_new[gene_expr_new<0],100-percentile_expr)
                    max_gene_expr_new = np.percentile(gene_expr_new[gene_expr_new>0],percentile_expr)
                    gene_expr_new[gene_expr_new>max_gene_expr_new] = max_gene_expr_new
                df_coord_new = pd.DataFrame(df_cells_pos_new['cells_pos'].tolist(),index=adata_new.obs_names)
                color_new = pd.Series(gene_expr_new).sample(frac=1,random_state=100)
                coord_new = df_coord_new.sample(frac=1,random_state=100)     
                if(show_all_cells):
                    if(experiment=='rna-seq'):
                        vmin = 0
                        vmax = max(max_gene_expr,max_gene_expr_new)   
                    if(experiment=='atac-seq'):
                        vmin = -max(max(abs(min_gene_expr),abs(min_gene_expr_new)),max(max_gene_expr,max_gene_expr_new))
                        vmax = max(max_gene_expr,max_gene_expr_new)                    
                    sc=ax.scatter(pd.concat([coord[0],coord_new[0]]), pd.concat([coord[1],coord_new[1]]),c=pd.concat([color,color_new]),
                                  vmin=vmin, vmax=vmax, s=50, cmap=cm, linewidths=0,alpha=0.5,zorder=10)
                else:
                    if(experiment=='rna-seq'):
                        vmin = 0
                        vmax = max_gene_expr_new
                    if(experiment=='atac-seq'):
                        vmin = -max(abs(min_gene_expr_new),max_gene_expr_new)
                        vmax = max(abs(min_gene_expr_new),max_gene_expr_new)
                    sc=ax.scatter(coord_new[0], coord_new[1],c=color_new,vmin=vmin, vmax=vmax, s=50, cmap=cm, linewidths=0,alpha=0.5,zorder=10)                      
            else:            
                sc=ax.scatter(coord[0], coord[1],c=color,vmin=vmin, vmax=vmax, s=50, cmap=cm, linewidths=0,alpha=0.5,zorder=10) 
            cbar=plt.colorbar(sc)
            cbar.ax.tick_params(labelsize=20)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.set_alpha(1)
            cbar.draw_all()
            ax.set_title(g,size=15)    
            ax.set_xlabel('pseudotime')       
            if(save_fig):
                plt.savefig(os.path.join(file_path_S,'subway_map_' + slugify(g) + '.' + fig_format),pad_inches=1,bbox_inches='tight')
                plt.close(fig) 


def stream_plot(adata,adata_new=None,show_all_colors=False,root='S0',factor_num_win=10,factor_min_win=2.0,factor_width=2.5,flag_log_view = False,preference=None,
                save_fig=False,fig_path=None,fig_name='stream_plot.pdf',fig_size=(12,8),fig_legend_ncol=3,tick_fontsize=20,label_fontsize=25):  
    """Generate stream plots
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    root: `str`, optional (default: 'S0'): 
        The starting node
    factor_num_win: `int`, optional (default: 10)
        Number of sliding windows. 
    factor_min_win: `float`, optional (default: 2.0)
        The factor used to calculate the sliding window size based on shortest branch. 
    factor_width: `float`, optional (default: 2.5)
        The ratio between length and width of stream plot. 
    flag_log_view: `bool`, optional (default: False)
        If True,the number of cells (the width) is logarithmized when outputing stream plot.
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot. The higher ranks the node have, the closer to the top the branch with that node is.
    show_all_colors: `bool`, optional (default: False)
        if show_all_colors is True and adata_new is speicified, original cells will be colored based on its color file. Otherwise, the original cells won't be distinguished and will be colored with 'grey' 
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_name: `str`, optional (default: 'stream_plot.pdf')
        if save_fig is True, specify figure name.
    fig_legend_ncol: `int`, optional (default: 3)
        The number of columns that the legend has.
    tick_fontsize: `int`, optional (default: 20)
        Tick label fontsize
    label_fontsize: `int`, optional (default: 25)
        The label fontsize of the x-axis

    Returns
    -------
    None

    """

    if(fig_path is None):
        if(adata_new==None):
            fig_path = adata.uns['workdir']
        else:
            fig_path = adata_new.uns['workdir']

    flat_tree = adata.uns['flat_tree']
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    if(root not in dict_label_node.keys()):
        print('There is no root '+root)
    else:
        file_path_S = os.path.join(fig_path,root)
        dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
        dict_node_state = nx.get_node_attributes(flat_tree,'label')
        input_cell_label_uni = list(adata.uns['label_color'].keys())
        input_cell_label_uni_color = adata.uns['label_color']        
        root_node = dict_label_node[root]
        node_start = root_node
        if(not os.path.exists(file_path_S)):
            os.makedirs(file_path_S) 
        bfs_edges = bfs_edges_modified(flat_tree,node_start,preference=preference_nodes)
        bfs_nodes = []
        for x in bfs_edges:
            if x[0] not in bfs_nodes:
                bfs_nodes.append(x[0])
            if x[1] not in bfs_nodes:
                bfs_nodes.append(x[1])   
        df_rooted_tree = adata.obs.copy()
        df_rooted_tree = df_rooted_tree.astype('object')
        df_rooted_tree['CELL_LABEL'] = df_rooted_tree['label']
        df_rooted_tree['edge'] = ''
        df_rooted_tree['lam_ordered'] = ''
        for x in bfs_edges:
            if x in nx.get_edge_attributes(flat_tree,'id').values():
                id_cells = np.where(df_rooted_tree['branch_id']==x)[0]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'edge'] = [x]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] = df_rooted_tree.loc[df_rooted_tree.index[id_cells],'branch_lam']
            else:
                id_cells = np.where(df_rooted_tree['branch_id']==(x[1],x[0]))[0]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'edge'] = [x]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] = flat_tree.edges[x]['len'] - df_rooted_tree.loc[df_rooted_tree.index[id_cells],'branch_lam']        
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_ordered']].copy()
        if(adata_new != None):
            df_rooted_tree_new = adata_new.obs.copy()
            df_rooted_tree_new = df_rooted_tree_new.astype('object')
            df_rooted_tree_new['CELL_LABEL'] = df_rooted_tree_new['label']
            df_rooted_tree_new['edge'] = ''
            df_rooted_tree_new['lam_ordered'] = ''
            input_cell_label_uni_new = list(adata_new.uns['label_color'].keys())
            input_cell_label_uni_color_new = adata_new.uns['label_color']     
            for x in bfs_edges:
                if x in nx.get_edge_attributes(flat_tree,'id').values():
                    id_cells = np.where(df_rooted_tree_new['branch_id']==x)[0]
                    df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'edge'] = [x]
                    df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'lam_ordered'] = df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'branch_lam']
                else:
                    id_cells = np.where(df_rooted_tree_new['branch_id']==(x[1],x[0]))[0]
                    df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'edge'] = [x]
                    df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'lam_ordered'] = flat_tree.edges[x]['len'] - df_rooted_tree_new.loc[df_rooted_tree_new.index[id_cells],'branch_lam']    
            if(show_all_colors):
                input_cell_label_uni = list(set(input_cell_label_uni + input_cell_label_uni_new))
                input_cell_label_uni_color = {x: input_cell_label_uni_color[x] if x in input_cell_label_uni_color.keys() 
                                              else input_cell_label_uni_color_new[x]
                                              for x in input_cell_label_uni}
            else:
                df_rooted_tree['CELL_LABEL'] = 'trajectory_cells'
                input_cell_label_uni =  ['trajectory_cells'] + input_cell_label_uni_new
                input_cell_label_uni_color = deepcopy(input_cell_label_uni_color_new)
                input_cell_label_uni_color['trajectory_cells'] = 'grey'
            df_stream = pd.concat([df_rooted_tree,df_rooted_tree_new],sort=True)[['CELL_LABEL','edge','lam_ordered']].copy()
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len']        

        dict_tree = {}
        bfs_prev = dict(nx.bfs_predecessors(flat_tree,node_start))
        bfs_next = dict(nx.bfs_successors(flat_tree,node_start))
        for x in bfs_nodes:
            dict_tree[x] = {'prev':"",'next':[]}
            if(x in bfs_prev.keys()):
                dict_tree[x]['prev'] = bfs_prev[x]
            if(x in bfs_next.keys()):
                x_rank = [bfs_nodes.index(x_next) for x_next in bfs_next[x]]
                dict_tree[x]['next'] = [y for _,y in sorted(zip(x_rank,bfs_next[x]),key=lambda y: y[0])]

        ##shift distance of each branch
        dict_shift_dist = dict()
        #modified depth first search
        dfs_nodes = dfs_nodes_modified(flat_tree,node_start,preference=preference_nodes)
        leaves=[n for n,d in flat_tree.degree() if d==1]
        id_leaf = 0
        dfs_nodes_copy = deepcopy(dfs_nodes)
        num_nonroot_leaf = len(list(set(leaves) - set([node_start])))
        while len(dfs_nodes_copy)>1:
            node = dfs_nodes_copy.pop()
            prev_node = dict_tree[node]['prev']
            if(node in leaves):
                dict_shift_dist[(prev_node,node)] = -1.1*(num_nonroot_leaf-1)/2.0 + id_leaf*1.1
                id_leaf = id_leaf+1
            else:
                next_nodes = dict_tree[node]['next']
                dict_shift_dist[(prev_node,node)] = (sum([dict_shift_dist[(node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))
        if (flat_tree.degree(node_start))>1:
            next_nodes = dict_tree[node_start]['next']
            dict_shift_dist[(node_start,node_start)] = (sum([dict_shift_dist[(node_start,next_node)] for next_node in next_nodes]))/float(len(next_nodes))


        #dataframe of bins
        df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary','center','edge'])    
        list_paths = find_root_to_leaf_paths(flat_tree, node_start)
        max_path_len = find_longest_path(list_paths,len_ori)
        size_w = max_path_len/np.float(factor_num_win)
        if(size_w>min(len_ori.values())/np.float(factor_min_win)):
            size_w = min(len_ori.values())/np.float(factor_min_win)
            
        step_w = size_w/2 #step of sliding window (the divisor should be even)    

        max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
        dict_shift_dist = {x: dict_shift_dist[x]*max_width for x in dict_shift_dist.keys()}
        min_width = 0.0 #min width of branch
        min_cellnum = 0 #the minimal cell number in one branch
        min_bin_cellnum = 0 #the minimal cell number in each bin
        dict_edge_filter = dict() #filter out cells whose total count on one edge is below the min_cellnum
        df_edge_cellnum = pd.DataFrame(index = df_stream['CELL_LABEL'].unique(),columns=bfs_edges,dtype=float)

        for i,edge_i in enumerate(bfs_edges):
            df_edge_i = df_stream[df_stream.edge==edge_i]
            cells_kept = df_edge_i.CELL_LABEL.value_counts()[df_edge_i.CELL_LABEL.value_counts()>min_cellnum].index
            df_edge_i = df_edge_i[df_edge_i['CELL_LABEL'].isin(cells_kept)]
            dict_edge_filter[edge_i] = df_edge_i
            for cell_i in df_stream['CELL_LABEL'].unique():
                df_edge_cellnum[edge_i][cell_i] = float(df_edge_i[df_edge_i['CELL_LABEL']==cell_i].shape[0])


        for i,edge_i in enumerate(bfs_edges):
            #degree of the start node
            degree_st = flat_tree.degree(edge_i[0])
            #degree of the end node
            degree_end = flat_tree.degree(edge_i[1])
            #matrix of windows only appearing on one edge
            mat_w = np.vstack([np.arange(0,len_ori[edge_i]-size_w+(len_ori[edge_i]/10**6),step_w),\
                           np.arange(size_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w)]).T
            mat_w[-1,-1] = len_ori[edge_i]
            if(degree_st==1):
                mat_w = np.insert(mat_w,0,[0,size_w/2.0],axis=0)
            if(degree_end == 1):
                mat_w = np.insert(mat_w,mat_w.shape[0],[len_ori[edge_i]-size_w/2.0,len_ori[edge_i]],axis=0)
            total_bins = df_bins.shape[1] # current total number of bins

            if(degree_st>1 and i==0):
                #matrix of windows appearing on multiple edges
                mat_w_common = np.array([[0,size_w/2.0],[0,size_w]])
                #neighbor nodes
                nb_nodes = list(flat_tree.neighbors(edge_i[0]))
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                #matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1] # current total number of bins
                for i_win in range(mat_w_common.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                    df_bins.loc['edge',"win"+str(total_bins+i_win)] = [(node_start,node_start)]
                    for j in range(degree_st):
                        df_edge_j = dict_edge_filter[(edge_i[0],nb_nodes[j])]
                        cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                        df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[0],nb_nodes[j]))
                    df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                    if(i_win == 0):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = 0
                    else:
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = size_w/2

            max_binnum = np.around((len_ori[edge_i]/4.0-size_w)/step_w) # the maximal number of merging bins
            df_edge_i = dict_edge_filter[edge_i]
            total_bins = df_bins.shape[1] # current total number of bins

            if(max_binnum<=1):
                for i_win in range(mat_w.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=mat_w[i_win,0],\
                                                        df_edge_i.lam_ordered<=mat_w[i_win,1])]['CELL_LABEL'].value_counts()
                    df_bins.loc[cell_num.index,"win"+str(total_bins+i_win)] = cell_num
                    df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w[i_win,:]
                    if(degree_st == 1 and i_win==0):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = 0
                    elif(degree_end == 1 and i_win==(mat_w.shape[0]-1)):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = len_ori[edge_i]
                    else:
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = np.mean(mat_w[i_win,:])
                df_bins.loc['edge',["win"+str(total_bins+i_win) for i_win in range(mat_w.shape[0])]] = [[edge_i]]

            if(max_binnum>1):
                id_stack = []
                for i_win in range(mat_w.shape[0]):
                    id_stack.append(i_win)
                    bd_bins = [mat_w[id_stack[0],0],mat_w[id_stack[-1],1]]#boundary of merged bins
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=bd_bins[0],\
                                                        df_edge_i.lam_ordered<=bd_bins[1])]['CELL_LABEL'].value_counts()
                    if(len(id_stack) == max_binnum or any(cell_num>min_bin_cellnum) or i_win==mat_w.shape[0]-1):
                        df_bins["win"+str(total_bins)] = ""
                        df_bins.loc[df_bins.index[:-3],"win"+str(total_bins)] = 0
                        df_bins.loc[cell_num.index,"win"+str(total_bins)] = cell_num
                        df_bins.loc['boundary',"win"+str(total_bins)] = bd_bins
                        df_bins.loc['edge',"win"+str(total_bins)] = [edge_i]
                        if(degree_st == 1 and (0 in id_stack)):
                            df_bins.loc['center',"win"+str(total_bins)] = 0
                        elif(degree_end == 1 and i_win==(mat_w.shape[0]-1)):
                            df_bins.loc['center',"win"+str(total_bins)] = len_ori[edge_i]
                        else:
                            df_bins.loc['center',"win"+str(total_bins)] = np.mean(bd_bins)
                        total_bins = total_bins + 1
                        id_stack = []

            if(degree_end>1):
                #matrix of windows appearing on multiple edges
                mat_w_common = np.vstack([np.arange(len_ori[edge_i]-size_w+step_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w),\
                                          np.arange(step_w,size_w+(len_ori[edge_i]/10**6),step_w)]).T
                #neighbor nodes
                nb_nodes = list(flat_tree.neighbors(edge_i[1]))
                nb_nodes.remove(edge_i[0])
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()

                #matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1] # current total number of bins
                if(mat_w_common.shape[0]>0):
                    for i_win in range(mat_w_common.shape[0]):
                        df_bins["win"+str(total_bins+i_win)] = ""
                        df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                        cell_num_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
                                                                    df_edge_i.lam_ordered<=len_ori[edge_i])]['CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common1.index,"win"+str(total_bins+i_win)] = cell_num_common1
                        df_bins.loc['edge',"win"+str(total_bins+i_win)] = [edge_i]
                        for j in range(degree_end - 1):
                            df_edge_j = dict_edge_filter[(edge_i[1],nb_nodes[j])]
                            cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                        df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                            df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                            df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                            if abs(((sum(mat_w_common[i_win,:])+len_ori[edge_i])/2)-(len_ori[edge_i]+size_w/2.0))< step_w/100.0:
                                df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[1],nb_nodes[j]))
                        df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = (sum(mat_w_common[i_win,:])+len_ori[edge_i])/2

        #order cell names by the index of first non-zero
        cell_list = df_bins.index[:-3]
        id_nonzero = []
        for i_cn,cellname in enumerate(cell_list):
            if(np.flatnonzero(df_bins.loc[cellname,]).size==0):
                print('Cell '+cellname+' does not exist')
                break
            else:
                id_nonzero.append(np.flatnonzero(df_bins.loc[cellname,])[0])
        cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()
        #original count
        df_bins_ori = df_bins.reindex(cell_list_sorted+['boundary','center','edge'])
        df_bins_cumsum = df_bins_ori.copy()
        df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

        if(flag_log_view):
            df_bins_cumsum.iloc[:-3,:] = (df_bins_cumsum.iloc[:-3,:].values.astype(float))/(df_bins_cumsum.iloc[:-3,:]).values.max()
            df_bins_cumsum.iloc[:-3,:] = np.log2(df_bins_cumsum.iloc[:-3,:].values.astype(float)+0.01)

        #normalization  
        df_bins_cumsum_norm = df_bins_cumsum.copy()
        df_bins_cumsum_norm.iloc[:-3,:] = min_width + max_width*(df_bins_cumsum.iloc[:-3,:]-(df_bins_cumsum.iloc[:-3,:]).values.min())/\
                                                         ((df_bins_cumsum.iloc[:-3,:]).values.max()-(df_bins_cumsum.iloc[:-3,:]).values.min())


        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3,:] = df_bins_cumsum_norm.iloc[:-3,:].subtract(df_bins_cumsum_norm.iloc[0,:]/2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4,:] = df_bins_top.iloc[1:-3,:].values
        df_bins_base.iloc[-4,:] = 0-df_bins_cumsum_norm.iloc[0,:]/2.0
        dict_forest = {cellname: {nodename:{'prev':"",'next':"",'div':""} for nodename in bfs_nodes}\
                       for cellname in df_edge_cellnum.index}
        for cellname in cell_list_sorted:
            for node_i in bfs_nodes:
                nb_nodes = list(flat_tree.neighbors(node_i))
                index_in_bfs = [bfs_nodes.index(nb) for nb in nb_nodes]
                nb_nodes_sorted = np.array(nb_nodes)[np.argsort(index_in_bfs)].tolist()
                if node_i == node_start:
                    next_nodes = nb_nodes_sorted
                    prev_nodes = ''
                else:
                    next_nodes = nb_nodes_sorted[1:]
                    prev_nodes = nb_nodes_sorted[0]
                dict_forest[cellname][node_i]['next'] = next_nodes
                dict_forest[cellname][node_i]['prev'] = prev_nodes
                if(len(next_nodes)>1):
                    pro_next_edges = [] #proportion of next edges
                    for nt in next_nodes:
                        id_wins = [ix for ix,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x == [(node_i,nt)]]
                        pro_next_edges.append(df_bins_cumsum_norm.loc[cellname,'win'+str(id_wins[0])])
                    if(sum(pro_next_edges)==0):
                        dict_forest[cellname][node_i]['div'] = np.cumsum(np.repeat(1.0/len(next_nodes),len(next_nodes))).tolist()
                    else:
                        dict_forest[cellname][node_i]['div'] = (np.cumsum(pro_next_edges)/sum(pro_next_edges)).tolist()

        #Shift
        dict_ep_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of end points
        dict_ep_base = {cellname:dict() for cellname in cell_list_sorted}
        dict_ep_center = dict() #center coordinates of end points in each branch

        df_top_x = df_bins_top.copy() # x coordinates in top line
        df_top_y = df_bins_top.copy() # y coordinates in top line
        df_base_x = df_bins_base.copy() # x coordinates in base line
        df_base_y = df_bins_base.copy() # y coordinates in base line

        for edge_i in bfs_edges:
            id_wins = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==edge_i]
            prev_node = dict_tree[edge_i[0]]['prev']
            if(prev_node == ''):
                x_st = 0
                if(flat_tree.degree(node_start)>1):
                    id_wins = id_wins[1:]
            else:
                id_wins = id_wins[1:] # remove the overlapped window
                x_st = dict_ep_center[(prev_node,edge_i[0])][0] - step_w
            y_st = dict_shift_dist[edge_i]
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',map(lambda x: 'win' + str(x), id_wins)]
                py_top = df_bins_top.loc[cellname,map(lambda x: 'win' + str(x), id_wins)]
                px_top_prime = x_st  + px_top
                py_top_prime = y_st  + py_top
                dict_ep_top[cellname][edge_i] = [px_top_prime[-1],py_top_prime[-1]]
                df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = px_top_prime
                df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',map(lambda x: 'win' + str(x), id_wins)]
                py_base = df_bins_base.loc[cellname,map(lambda x: 'win' + str(x), id_wins)]
                px_base_prime = x_st + px_base
                py_base_prime = y_st + py_base
                dict_ep_base[cellname][edge_i] = [px_base_prime[-1],py_base_prime[-1]]
                df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = px_base_prime
                df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = py_base_prime
            dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

        id_wins_start = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==(node_start,node_start)]
        if(len(id_wins_start)>0):
            mean_shift_dist = np.mean([dict_shift_dist[(node_start,x)] \
                                    for x in dict_forest[cell_list_sorted[0]][node_start]['next']])
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',map(lambda x: 'win' + str(x), id_wins_start)]
                py_top = df_bins_top.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)]
                px_top_prime = 0  + px_top
                py_top_prime = mean_shift_dist  + py_top
                df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = px_top_prime
                df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',map(lambda x: 'win' + str(x), id_wins_start)]
                py_base = df_bins_base.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)]
                px_base_prime = 0 + px_base
                py_base_prime = mean_shift_dist + py_base
                df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = px_base_prime
                df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = py_base_prime

        #determine joints points
        dict_joint_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        dict_joint_base = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        if(flat_tree.degree(node_start)==1):
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1]
        else:
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1 and x[0]!=(node_start,node_start)]
            id_joints.insert(0,1)
        for id_j in id_joints:
            joint_edges = df_bins_cumsum_norm.loc['edge','win'+str(id_j)]
            for id_div,edge_i in enumerate(joint_edges[1:]):
                id_wins = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x==[edge_i]]
                for cellname in cell_list_sorted:
                    if(len(dict_forest[cellname][edge_i[0]]['div'])>0):
                        prev_node_top_x = df_top_x.loc[cellname,'win'+str(id_j)]
                        prev_node_top_y = df_top_y.loc[cellname,'win'+str(id_j)]
                        prev_node_base_x = df_base_x.loc[cellname,'win'+str(id_j)]
                        prev_node_base_y = df_base_y.loc[cellname,'win'+str(id_j)]
                        div = dict_forest[cellname][edge_i[0]]['div']
                        if(id_div==0):
                            px_top_prime_st = prev_node_top_x
                            py_top_prime_st = prev_node_top_y
                        else:
                            px_top_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x)*div[id_div-1]
                            py_top_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y)*div[id_div-1]
                        px_base_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x)*div[id_div]
                        py_base_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y)*div[id_div]
                        df_top_x.loc[cellname,'win'+str(id_wins[0])] = px_top_prime_st
                        df_top_y.loc[cellname,'win'+str(id_wins[0])] = py_top_prime_st
                        df_base_x.loc[cellname,'win'+str(id_wins[0])] = px_base_prime_st
                        df_base_y.loc[cellname,'win'+str(id_wins[0])] = py_base_prime_st
                        dict_joint_top[cellname][edge_i] = np.array([px_top_prime_st,py_top_prime_st])
                        dict_joint_base[cellname][edge_i] = np.array([px_base_prime_st,py_base_prime_st])

        dict_tree_copy = deepcopy(dict_tree)
        dict_paths_top,dict_paths_base = find_paths(dict_tree_copy,bfs_nodes)

        #identify boundary of each edge
        dict_edge_bd = dict()
        for edge_i in bfs_edges:
            id_wins = [i for i,x in enumerate(df_top_x.loc['edge',:]) if edge_i in x]
            dict_edge_bd[edge_i] = [df_top_x.iloc[0,id_wins[0]],df_top_x.iloc[0,id_wins[-1]]]

        x_smooth = np.unique(np.arange(min(df_base_x.iloc[0,:]),max(df_base_x.iloc[0,:]),step = step_w/20).tolist() \
                    + [max(df_base_x.iloc[0,:])]).tolist()
        x_joints = df_top_x.iloc[0,id_joints].tolist()
        #replace nearest value in x_smooth by x axis of joint points
        for x in x_joints:
            x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

        dict_smooth_linear = {cellname:{'top':dict(),'base':dict()} for cellname in cell_list_sorted}
        #interpolation
        for edge_i_top in dict_paths_top.keys():
            path_i_top = dict_paths_top[edge_i_top]
            id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_top))]
            if(flat_tree.degree(node_start)>1 and \
               edge_i_top==(node_start,dict_forest[cell_list_sorted[0]][node_start]['next'][0])):
                id_wins_top.insert(0,1)
                id_wins_top.insert(0,0)
            for cellname in cell_list_sorted:
                x_top = df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_top)].tolist()
                y_top = df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_top)].tolist()
                f_top_linear = interpolate.interp1d(x_top, y_top, kind = 'linear')
                x_top_new = [x for x in x_smooth if (x>=x_top[0]) and (x<=x_top[-1])] + [x_top[-1]]
                x_top_new = np.unique(x_top_new).tolist()
                y_top_new_linear = f_top_linear(x_top_new)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node],path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_top_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_linear[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected],\
                                                                         np.array(y_top_new_linear)[id_selected]],index=['x','y'])
        for edge_i_base in dict_paths_base.keys():
            path_i_base = dict_paths_base[edge_i_base]
            id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_base))]
            if(flat_tree.degree(node_start)>1 and \
               edge_i_base==(node_start,dict_forest[cell_list_sorted[0]][node_start]['next'][-1])):
                id_wins_base.insert(0,1)
                id_wins_base.insert(0,0)
            for cellname in cell_list_sorted:
                x_base = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_base)].tolist()
                y_base = df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_base)].tolist()
                f_base_linear = interpolate.interp1d(x_base, y_base, kind = 'linear')
                x_base_new = [x for x in x_smooth if (x>=x_base[0]) and (x<=x_base[-1])] + [x_base[-1]]
                x_base_new = np.unique(x_base_new).tolist()
                y_base_new_linear = f_base_linear(x_base_new)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node],path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_base_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_linear[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected],\
                                                                          np.array(y_base_new_linear)[id_selected]],index=['x','y'])

        #searching for edges which cell exists based on the linear interpolation
        dict_edges_CE = {cellname:[] for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                if(sum(abs(dict_smooth_linear[cellname]['top'][edge_i].loc['y'] - \
                       dict_smooth_linear[cellname]['base'][edge_i].loc['y']) > 1e-12)):
                    dict_edges_CE[cellname].append(edge_i)


        #determine paths which cell exists
        dict_paths_CE_top = {cellname:{} for cellname in cell_list_sorted}
        dict_paths_CE_base = {cellname:{} for cellname in cell_list_sorted}
        dict_forest_CE = dict()
        for cellname in cell_list_sorted:
            edges_cn = dict_edges_CE[cellname]
            nodes = [nodename for nodename in bfs_nodes if nodename in set(itertools.chain(*edges_cn))]
            dict_forest_CE[cellname] = {nodename:{'prev':"",'next':[]} for nodename in nodes}
            for node_i in nodes:
                prev_node = dict_tree[node_i]['prev']
                if((prev_node,node_i) in edges_cn):
                    dict_forest_CE[cellname][node_i]['prev'] = prev_node
                next_nodes = dict_tree[node_i]['next']
                for x in next_nodes:
                    if (node_i,x) in edges_cn:
                        (dict_forest_CE[cellname][node_i]['next']).append(x)
            dict_paths_CE_top[cellname],dict_paths_CE_base[cellname] = find_paths(dict_forest_CE[cellname],nodes)


        dict_smooth_new = deepcopy(dict_smooth_linear)
        for cellname in cell_list_sorted:
            paths_CE_top = dict_paths_CE_top[cellname]
            for edge_i_top in paths_CE_top.keys():
                path_i_top = paths_CE_top[edge_i_top]
                edges_top = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_top))]
                id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_top))]

                x_top = []
                y_top = []
                for e_t in edges_top:
                    if(e_t == edges_top[-1]):
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].loc['y']
                        px = dict_smooth_linear[cellname]['top'][e_t].loc['x']
                    else:
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].iloc[1,:-1]
                        px = dict_smooth_linear[cellname]['top'][e_t].iloc[0,:-1]
                    x_top = x_top + px.tolist()
                    y_top = y_top + py_top_linear.tolist()
                x_top_new = x_top
                y_top_new = savgol_filter(y_top,11,polyorder=1)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node],path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_top_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_new[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected],\
                                                                         np.array(y_top_new)[id_selected]],index=['x','y'])

            paths_CE_base = dict_paths_CE_base[cellname]
            for edge_i_base in paths_CE_base.keys():
                path_i_base = paths_CE_base[edge_i_base]
                edges_base = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_base))]
                id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_base))]

                x_base = []
                y_base = []
                for e_b in edges_base:
                    if(e_b == edges_base[-1]):
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].loc['y']
                        px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                    else:
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].iloc[1,:-1]
                        px = dict_smooth_linear[cellname]['base'][e_b].iloc[0,:-1]
                    x_base = x_base + px.tolist()
                    y_base = y_base + py_base_linear.tolist()
                x_base_new = x_base
                y_base_new = savgol_filter(y_base,11,polyorder=1)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node],path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_base_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_new[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected],\
                                                                          np.array(y_base_new)[id_selected]],index=['x','y'])

        #find all edges of polygon
        poly_edges = []
        dict_tree_copy = deepcopy(dict_tree)
        cur_node = node_start
        next_node = dict_tree_copy[cur_node]['next'][0]
        dict_tree_copy[cur_node]['next'].pop(0)
        poly_edges.append((cur_node,next_node))
        cur_node = next_node
        while(not(next_node==node_start and cur_node == dict_tree[node_start]['next'][-1])):
            while(len(dict_tree_copy[cur_node]['next'])!=0):
                next_node = dict_tree_copy[cur_node]['next'][0]
                dict_tree_copy[cur_node]['next'].pop(0)
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[node_start]['next'][-1] and next_node==node_start):
                    break
                cur_node = next_node
            while(len(dict_tree_copy[cur_node]['next'])==0):
                next_node = dict_tree_copy[cur_node]['prev']
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[node_start]['next'][-1] and next_node==node_start):
                    break
                cur_node = next_node


        verts = {cellname: np.empty((0,2)) for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in poly_edges:
                if edge_i in bfs_edges:
                    x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                    y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                    pxy = np.array([x_top,y_top]).T
                else:
                    edge_i = (edge_i[1],edge_i[0])
                    x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                    y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                    x_base = x_base[::-1]
                    y_base = y_base[::-1]
                    pxy = np.array([x_base,y_base]).T
                verts[cellname] = np.vstack((verts[cellname],pxy))


        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
        patches = []
        legend_labels = []
        for cellname in cell_list_sorted:
            legend_labels.append(cellname)
            verts_cell = verts[cellname]
            polygon = Polygon(verts_cell,closed=True,color=input_cell_label_uni_color[cellname],alpha=0.8,lw=0)
            ax.add_patch(polygon)

        plt.xticks(fontsize=tick_fontsize)
        # plt.xticks([])
        plt.yticks([])
        plt.xlabel('Pseudotime',fontsize=label_fontsize)
        xloc = plt.MaxNLocator(5)
        ax.xaxis.set_major_locator(xloc)        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.autoscale_view()

        fig_xmin, fig_xmax = ax.get_xlim()
        fig_ymin, fig_ymax = ax.get_ylim()
        ax.set_ylim([fig_ymin-(fig_ymax-fig_ymin)*0.15,fig_ymax+(fig_ymax-fig_ymin)*0.1])
        # manual arrowhead width and length
        fig_hw = 1./20.*(fig_ymax-fig_ymin)
        fig_hl = 1./20.*(fig_xmax-fig_xmin)
        ax.arrow(fig_xmin, fig_ymin-(fig_ymax-fig_ymin)*0.1, fig_xmax-fig_xmin, 0., fc='k', ec='k', lw = 1.0,
                 head_width=fig_hw, head_length=fig_hl, overhang = 0.3,
                 length_includes_head= True, clip_on = False)
        plt.legend(legend_labels,prop={'size':tick_fontsize},loc='center', bbox_to_anchor=(0.5, 1.20),ncol=fig_legend_ncol, \
                   fancybox=True, shadow=True)

        if(save_fig):
            plt.savefig(os.path.join(file_path_S,fig_name),pad_inches=1,bbox_inches='tight',dpi=120)
            plt.close(fig)  


def fill_im_array(dict_im_array,df_bins_gene,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge):
    pad_ratio = 0.008
    xmin_edge = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].min()
    xmax_edge = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].max()
    id_st_x = int(np.floor(((xmin_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
    id_ed_x =  int(np.floor(((xmax_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
    if (flat_tree.degree(edge_i[1])==1):
        id_ed_x = id_ed_x + 1
    if(id_st_x < 0):
        id_st_x = 0
    if(id_st_x >0):
        id_st_x  = id_st_x + 1
    if(id_ed_x>(im_ncol-1)):
        id_ed_x = im_ncol - 1
    if(prev_edge != ''):
        shift_dist = dict_shift_dist[edge_i] - dict_shift_dist[prev_edge]
        gene_color = df_bins_gene.loc[cellname,map(lambda x: 'win' + str(x), [id_wins_prev[-1]] + id_wins[1:])].tolist()
    else:
        gene_color = df_bins_gene.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].tolist()
    x_axis = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].tolist()
    x_base = np.linspace(x_axis[0],x_axis[-1],id_ed_x-id_st_x+1)
    gene_color_new = np.interp(x_base,x_axis,gene_color)
    y_axis_base = df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].tolist()
    y_axis_top = df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].tolist()
    f_base_linear = interpolate.interp1d(x_axis, y_axis_base, kind = 'linear')
    f_top_linear = interpolate.interp1d(x_axis, y_axis_top, kind = 'linear')
    y_base = f_base_linear(x_base)
    y_top = f_top_linear(x_base)
    id_y_base = np.ceil((1-(y_base-ymin)/(ymax-ymin))*(im_nrow-1)).astype(int) + int(im_ncol * pad_ratio)
    id_y_base[id_y_base<0]=0
    id_y_base[id_y_base>(im_nrow-1)]=im_nrow-1
    id_y_top = np.floor((1-(y_top-ymin)/(ymax-ymin))*(im_nrow-1)).astype(int) - int(im_ncol * pad_ratio)
    id_y_top[id_y_top<0]=0
    id_y_top[id_y_top>(im_nrow-1)]=im_nrow-1
    id_x_base = range(id_st_x,(id_ed_x+1))
    for x in range(len(id_y_base)):
        if(x in range(int(step_w/xmax * im_ncol)) and prev_edge != ''):
            if(shift_dist>0):
                id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio)
                id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio) - \
                                int(abs(shift_dist)/abs(ymin -ymax) * im_nrow * 0.3)
                if(id_y_top[x] < 0):
                    id_y_top[x] = 0
            if(shift_dist<0):
                id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio) + \
                                int(abs(shift_dist)/abs(ymin -ymax) * im_nrow * 0.3)
                id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio)
                if(id_y_base[x] > im_nrow-1):
                    id_y_base[x] = im_nrow-1
        dict_im_array[cellname][id_y_top[x]:(id_y_base[x]+1),id_x_base[x]] =  np.tile(gene_color_new[x],\
                                                                          (id_y_base[x]-id_y_top[x]+1))
    return dict_im_array


def stream_plot_gene(adata,genes=None,percentile_expr=95,root='S0',factor_num_win=10,factor_min_win=2.0,factor_width=2.5,flag_log_view = False,preference=None,
                    save_fig=False,fig_path=None,fig_size=(12,8),fig_format='pdf',tick_fontsize=20,label_fontsize=25):  
    """Generate stream plots of genes
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    genes: `list`, optional (default: None): 
        A list of genes to be displayed
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Specify the percentile of gene expression greater than 0 to filter out some extreme gene expressions.      
    root: `str`, optional (default: 'S0'): 
        The starting node
    factor_num_win: `int`, optional (default: 10)
        Number of sliding windows. 
    factor_min_win: `float`, optional (default: 2.0)
        The factor used to calculate the sliding window size based on shortest branch. 
    factor_width: `float`, optional (default: 2.5)
        The ratio between length and width of stream plot. 
    flag_log_view: `bool`, optional (default: False)
        If True,the number of cells (the width) is logarithmized when outputing stream plot.
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot. The higher ranks the node have, the closer to the top the branch with that node is.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_size: `tuple`, optional (default: (8,8))
        figure size.
    fig_path: `str`, optional (default: None)
        if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.
    tick_fontsize: `int`, optional (default: 20)
        Tick label fontsize
    label_fontsize: `int`, optional (default: 25)
        The label fontsize of the x-axis

    Returns
    -------
    None

    """

    if(fig_path is None):
        fig_path = adata.uns['workdir']
    experiment = adata.uns['experiment']
    flat_tree = adata.uns['flat_tree']
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    if(root not in dict_label_node.keys()):
        print('There is no root '+root)
    elif(genes is None):
        print('Please provide gene names');
    else:
        genes = np.unique(genes).tolist()
        for x in genes:
            if x not in adata.var_names:
                print(x + ' is not in the gene expression matrix')
                return       
        file_path_S = os.path.join(fig_path,root)
        dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
        dict_node_state = nx.get_node_attributes(flat_tree,'label')
        input_cell_label_uni = ['unknown']
        input_cell_label_uni_color = {'unknown':'gray'}  
        root_node = dict_label_node[root]
        node_start = root_node
        if(not os.path.exists(file_path_S)):
            os.makedirs(file_path_S) 
        bfs_edges = bfs_edges_modified(flat_tree,node_start,preference=preference_nodes)
        bfs_nodes = []
        for x in bfs_edges:
            if x[0] not in bfs_nodes:
                bfs_nodes.append(x[0])
            if x[1] not in bfs_nodes:
                bfs_nodes.append(x[1])   
        df_rooted_tree = adata.obs.copy()
        df_rooted_tree = df_rooted_tree.astype('object')
        df_rooted_tree['CELL_LABEL'] = df_rooted_tree['label']
        df_rooted_tree['edge'] = ''
        df_rooted_tree['lam_ordered'] = ''
        for x in bfs_edges:
            if x in nx.get_edge_attributes(flat_tree,'id').values():
                id_cells = np.where(df_rooted_tree['branch_id']==x)[0]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'edge'] = [x]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] = df_rooted_tree.loc[df_rooted_tree.index[id_cells],'branch_lam']
            else:
                id_cells = np.where(df_rooted_tree['branch_id']==(x[1],x[0]))[0]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'edge'] = [x]
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] = flat_tree.edges[x]['len'] - df_rooted_tree.loc[df_rooted_tree.index[id_cells],'branch_lam']        
        df_gene_expr = pd.DataFrame(index= adata.obs_names.tolist(),
                                    data = adata.raw[:,genes].X,
                                    columns=genes)
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_ordered']].copy()
        df_stream['CELL_LABEL'] = 'unknown'
        df_stream[genes] = df_gene_expr[genes]

        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len']        

        dict_tree = {}
        bfs_prev = dict(nx.bfs_predecessors(flat_tree,node_start))
        bfs_next = dict(nx.bfs_successors(flat_tree,node_start))
        for x in bfs_nodes:
            dict_tree[x] = {'prev':"",'next':[]}
            if(x in bfs_prev.keys()):
                dict_tree[x]['prev'] = bfs_prev[x]
            if(x in bfs_next.keys()):
                x_rank = [bfs_nodes.index(x_next) for x_next in bfs_next[x]]
                dict_tree[x]['next'] = [y for _,y in sorted(zip(x_rank,bfs_next[x]),key=lambda y: y[0])]

        ##shift distance of each branch
        dict_shift_dist = dict()
        #modified depth first search
        dfs_nodes = dfs_nodes_modified(flat_tree,node_start,preference=preference_nodes)
        leaves=[n for n,d in flat_tree.degree() if d==1]
        id_leaf = 0
        dfs_nodes_copy = deepcopy(dfs_nodes)
        num_nonroot_leaf = len(list(set(leaves) - set([node_start])))
        while len(dfs_nodes_copy)>1:
            node = dfs_nodes_copy.pop()
            prev_node = dict_tree[node]['prev']
            if(node in leaves):
                dict_shift_dist[(prev_node,node)] = -1.1*(num_nonroot_leaf-1)/2.0 + id_leaf*1.1
                id_leaf = id_leaf+1
            else:
                next_nodes = dict_tree[node]['next']
                dict_shift_dist[(prev_node,node)] = (sum([dict_shift_dist[(node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))
        if (flat_tree.degree(node_start))>1:
            next_nodes = dict_tree[node_start]['next']
            dict_shift_dist[(node_start,node_start)] = (sum([dict_shift_dist[(node_start,next_node)] for next_node in next_nodes]))/float(len(next_nodes))


        #dataframe of bins
        df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary','center','edge'])
        dict_genes = {gene: pd.DataFrame(index=list(df_stream['CELL_LABEL'].unique())) for gene in genes}
        dict_merge_num = {gene:[] for gene in genes} #number of merged sliding windows          
        list_paths = find_root_to_leaf_paths(flat_tree, node_start)
        max_path_len = find_longest_path(list_paths,len_ori)
        size_w = max_path_len/np.float(factor_num_win)
        if(size_w>min(len_ori.values())/np.float(factor_min_win)):
            size_w = min(len_ori.values())/np.float(factor_min_win)
            
        step_w = size_w/2 #step of sliding window (the divisor should be even)    

        max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
        dict_shift_dist = {x: dict_shift_dist[x]*max_width for x in dict_shift_dist.keys()}
        min_width = 0.0 #min width of branch
        min_cellnum = 0 #the minimal cell number in one branch
        min_bin_cellnum = 0 #the minimal cell number in each bin
        dict_edge_filter = dict() #filter out cells whose total count on one edge is below the min_cellnum
        df_edge_cellnum = pd.DataFrame(index = df_stream['CELL_LABEL'].unique(),columns=bfs_edges,dtype=float)

        for i,edge_i in enumerate(bfs_edges):
            df_edge_i = df_stream[df_stream.edge==edge_i]
            cells_kept = df_edge_i.CELL_LABEL.value_counts()[df_edge_i.CELL_LABEL.value_counts()>min_cellnum].index
            df_edge_i = df_edge_i[df_edge_i['CELL_LABEL'].isin(cells_kept)]
            dict_edge_filter[edge_i] = df_edge_i
            for cell_i in df_stream['CELL_LABEL'].unique():
                df_edge_cellnum[edge_i][cell_i] = float(df_edge_i[df_edge_i['CELL_LABEL']==cell_i].shape[0])


        for i,edge_i in enumerate(bfs_edges):
            #degree of the start node
            degree_st = flat_tree.degree(edge_i[0])
            #degree of the end node
            degree_end = flat_tree.degree(edge_i[1])
            #matrix of windows only appearing on one edge
            mat_w = np.vstack([np.arange(0,len_ori[edge_i]-size_w+(len_ori[edge_i]/10**6),step_w),\
                           np.arange(size_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w)]).T
            mat_w[-1,-1] = len_ori[edge_i]
            if(degree_st==1):
                mat_w = np.insert(mat_w,0,[0,size_w/2.0],axis=0)
            if(degree_end == 1):
                mat_w = np.insert(mat_w,mat_w.shape[0],[len_ori[edge_i]-size_w/2.0,len_ori[edge_i]],axis=0)
            total_bins = df_bins.shape[1] # current total number of bins

            if(degree_st>1 and i==0):
                #matrix of windows appearing on multiple edges
                mat_w_common = np.array([[0,size_w/2.0],[0,size_w]])
                #neighbor nodes
                nb_nodes = list(flat_tree.neighbors(edge_i[0]))
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                #matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1] # current total number of bins
                for i_win in range(mat_w_common.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                    df_bins.loc['edge',"win"+str(total_bins+i_win)] = [(node_start,node_start)]
                    dict_df_genes_common = dict()
                    for gene in genes:
                        dict_df_genes_common[gene] = list()
                    for j in range(degree_st):
                        df_edge_j = dict_edge_filter[(edge_i[0],nb_nodes[j])]
                        cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                        for gene in genes:
                            dict_df_genes_common[gene].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
        #                     gene_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
        #                                                             df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[gene].mean()
        #                     dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] = \
        #                     dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] + gene_values_common2
                        df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[0],nb_nodes[j]))
                    for gene in genes:
                        gene_values_common = pd.concat(dict_df_genes_common[gene]).groupby(['CELL_LABEL'])[gene].mean()
                        dict_genes[gene].loc[gene_values_common.index,"win"+str(total_bins+i_win)] = gene_values_common
                    df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                    if(i_win == 0):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = 0
                    else:
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = size_w/2

            max_binnum = np.around((len_ori[edge_i]/4.0-size_w)/step_w) # the maximal number of merging bins
            df_edge_i = dict_edge_filter[edge_i]
            total_bins = df_bins.shape[1] # current total number of bins

            if(max_binnum<=1):
                for i_win in range(mat_w.shape[0]):
                    df_bins["win"+str(total_bins+i_win)] = ""
                    df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=mat_w[i_win,0],\
                                                        df_edge_i.lam_ordered<=mat_w[i_win,1])]['CELL_LABEL'].value_counts()
                    df_bins.loc[cell_num.index,"win"+str(total_bins+i_win)] = cell_num
                    df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w[i_win,:]
                    for gene in genes:
                        dict_genes[gene]["win"+str(total_bins+i_win)] = 0
                        gene_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=mat_w[i_win,0],\
                                                        df_edge_i.lam_ordered<=mat_w[i_win,1])].groupby(['CELL_LABEL'])[gene].mean()
                        dict_genes[gene].loc[gene_values.index,"win"+str(total_bins+i_win)] = gene_values
                        dict_merge_num[gene].append(1)
                    if(degree_st == 1 and i_win==0):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = 0
                    elif(degree_end == 1 and i_win==(mat_w.shape[0]-1)):
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = len_ori[edge_i]
                    else:
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = np.mean(mat_w[i_win,:])
                df_bins.loc['edge',["win"+str(total_bins+i_win) for i_win in range(mat_w.shape[0])]] = [[edge_i]]

            if(max_binnum>1):
                id_stack = []
                for i_win in range(mat_w.shape[0]):
                    id_stack.append(i_win)
                    bd_bins = [mat_w[id_stack[0],0],mat_w[id_stack[-1],1]]#boundary of merged bins
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=bd_bins[0],\
                                                        df_edge_i.lam_ordered<=bd_bins[1])]['CELL_LABEL'].value_counts()
                    if(len(id_stack) == max_binnum or any(cell_num>min_bin_cellnum) or i_win==mat_w.shape[0]-1):
                        df_bins["win"+str(total_bins)] = ""
                        df_bins.loc[df_bins.index[:-3],"win"+str(total_bins)] = 0
                        df_bins.loc[cell_num.index,"win"+str(total_bins)] = cell_num
                        df_bins.loc['boundary',"win"+str(total_bins)] = bd_bins
                        df_bins.loc['edge',"win"+str(total_bins)] = [edge_i]
                        for gene in genes:
                            dict_genes[gene]["win"+str(total_bins)] = 0
                            gene_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=bd_bins[0],\
                                                            df_edge_i.lam_ordered<=bd_bins[1])].groupby(['CELL_LABEL'])[gene].mean()
                            dict_genes[gene].loc[gene_values.index,"win"+str(total_bins)] = gene_values
                            dict_merge_num[gene].append(len(id_stack))
                        if(degree_st == 1 and (0 in id_stack)):
                            df_bins.loc['center',"win"+str(total_bins)] = 0
                        elif(degree_end == 1 and i_win==(mat_w.shape[0]-1)):
                            df_bins.loc['center',"win"+str(total_bins)] = len_ori[edge_i]
                        else:
                            df_bins.loc['center',"win"+str(total_bins)] = np.mean(bd_bins)
                        total_bins = total_bins + 1
                        id_stack = []

            if(degree_end>1):
                #matrix of windows appearing on multiple edges
                mat_w_common = np.vstack([np.arange(len_ori[edge_i]-size_w+step_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w),\
                                          np.arange(step_w,size_w+(len_ori[edge_i]/10**6),step_w)]).T
                #neighbor nodes
                nb_nodes = list(flat_tree.neighbors(edge_i[1]))
                nb_nodes.remove(edge_i[0])
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()

                #matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1] # current total number of bins
                if(mat_w_common.shape[0]>0):
                    for i_win in range(mat_w_common.shape[0]):
                        df_bins["win"+str(total_bins+i_win)] = ""
                        df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                        cell_num_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
                                                                    df_edge_i.lam_ordered<=len_ori[edge_i])]['CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common1.index,"win"+str(total_bins+i_win)] = cell_num_common1
                        dict_df_genes_common = dict()
                        for gene in genes:
                            dict_genes[gene]["win"+str(total_bins+i_win)] = 0
                            dict_df_genes_common[gene] = list()
                            dict_df_genes_common[gene].append(df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
                                                                    df_edge_i.lam_ordered<=len_ori[edge_i])])
        #                     gene_values_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
        #                                                             df_edge_i.lam_ordered<=len_ori[edge_i])].groupby(['CELL_LABEL'])[gene].mean()
        #                     dict_genes[gene].ix[gene_values_common1.index,"win"+str(total_bins+i_win)] = gene_values_common1
                            dict_merge_num[gene].append(1)
                        df_bins.loc['edge',"win"+str(total_bins+i_win)] = [edge_i]
                        for j in range(degree_end - 1):
                            df_edge_j = dict_edge_filter[(edge_i[1],nb_nodes[j])]
                            cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                        df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                            df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                            df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                            for gene in genes:
                                dict_df_genes_common[gene].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                        df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
        #                         gene_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
        #                                                                 df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[gene].mean()
        #                         dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] = \
        #                         dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] + gene_values_common2
                            if abs(((sum(mat_w_common[i_win,:])+len_ori[edge_i])/2)-(len_ori[edge_i]+size_w/2.0))< step_w/100.0:
                                df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[1],nb_nodes[j]))
                        for gene in genes:
                            gene_values_common = pd.concat(dict_df_genes_common[gene]).groupby(['CELL_LABEL'])[gene].mean()
                            dict_genes[gene].loc[gene_values_common.index,"win"+str(total_bins+i_win)] = gene_values_common
                        df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                        df_bins.loc['center',"win"+str(total_bins+i_win)] = (sum(mat_w_common[i_win,:])+len_ori[edge_i])/2

        #order cell names by the index of first non-zero
        cell_list = df_bins.index[:-3]
        id_nonzero = []
        for i_cn,cellname in enumerate(cell_list):
            if(np.flatnonzero(df_bins.loc[cellname,]).size==0):
                print('Cell '+cellname+' does not exist')
                break
            else:
                id_nonzero.append(np.flatnonzero(df_bins.loc[cellname,])[0])
        cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()
        #original count
        df_bins_ori = df_bins.reindex(cell_list_sorted+['boundary','center','edge'])
        df_bins_cumsum = df_bins_ori.copy()
        df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

        if(flag_log_view):
            df_bins_cumsum.iloc[:-3,:] = (df_bins_cumsum.iloc[:-3,:].values.astype(float))/(df_bins_cumsum.iloc[:-3,:]).values.max()
            df_bins_cumsum.iloc[:-3,:] = np.log2(df_bins_cumsum.iloc[:-3,:].values.astype(float)+0.01)

        #normalization  
        df_bins_cumsum_norm = df_bins_cumsum.copy()
        df_bins_cumsum_norm.iloc[:-3,:] = min_width + max_width*(df_bins_cumsum.iloc[:-3,:]-(df_bins_cumsum.iloc[:-3,:]).values.min())/\
                                                         ((df_bins_cumsum.iloc[:-3,:]).values.max()-(df_bins_cumsum.iloc[:-3,:]).values.min())

        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3,:] = df_bins_cumsum_norm.iloc[:-3,:].subtract(df_bins_cumsum_norm.iloc[0,:]/2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4,:] = df_bins_top.iloc[1:-3,:].values
        df_bins_base.iloc[-4,:] = 0-df_bins_cumsum_norm.iloc[0,:]/2.0
        dict_genes_norm = deepcopy(dict_genes)
        
        if(experiment=='rna-seq'):
            for gene in genes:
                gene_values = dict_genes[gene].iloc[0,].values
                max_gene_values = np.percentile(gene_values[gene_values>0],percentile_expr)
                dict_genes_norm[gene] = dict_genes[gene].reindex(cell_list_sorted)
                dict_genes_norm[gene][dict_genes_norm[gene]>max_gene_values] = max_gene_values
        elif(experiment=='atac-seq'):
            for gene in genes:
                gene_values = dict_genes[gene].iloc[0,].values
                min_gene_values = np.percentile(gene_values[gene_values<0],100-percentile_expr)
                max_gene_values = np.percentile(gene_values[gene_values>0],percentile_expr)
                dict_genes_norm[gene] = dict_genes[gene].reindex(cell_list_sorted)
                dict_genes_norm[gene][dict_genes_norm[gene]<min_gene_values] = min_gene_values
                dict_genes_norm[gene][dict_genes_norm[gene]>max_gene_values] = max_gene_values             
        else:
            print('The experiment '+experiment +' is not supported')
            return
            
        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3,:] = df_bins_cumsum_norm.iloc[:-3,:].subtract(df_bins_cumsum_norm.iloc[0,:]/2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4,:] = df_bins_top.iloc[1:-3,:].values
        df_bins_base.iloc[-4,:] = 0-df_bins_cumsum_norm.iloc[0,:]/2.0

        dict_forest = {cellname: {nodename:{'prev':"",'next':"",'div':""} for nodename in bfs_nodes}\
                       for cellname in df_edge_cellnum.index}
        for cellname in cell_list_sorted:
            for node_i in bfs_nodes:
                nb_nodes = list(flat_tree.neighbors(node_i))
                index_in_bfs = [bfs_nodes.index(nb) for nb in nb_nodes]
                nb_nodes_sorted = np.array(nb_nodes)[np.argsort(index_in_bfs)].tolist()
                if node_i == node_start:
                    next_nodes = nb_nodes_sorted
                    prev_nodes = ''
                else:
                    next_nodes = nb_nodes_sorted[1:]
                    prev_nodes = nb_nodes_sorted[0]
                dict_forest[cellname][node_i]['next'] = next_nodes
                dict_forest[cellname][node_i]['prev'] = prev_nodes
                if(len(next_nodes)>1):
                    pro_next_edges = [] #proportion of next edges
                    for nt in next_nodes:
                        id_wins = [ix for ix,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x == [(node_i,nt)]]
                        pro_next_edges.append(df_bins_cumsum_norm.loc[cellname,'win'+str(id_wins[0])])
                    if(sum(pro_next_edges)==0):
                        dict_forest[cellname][node_i]['div'] = np.cumsum(np.repeat(1.0/len(next_nodes),len(next_nodes))).tolist()
                    else:
                        dict_forest[cellname][node_i]['div'] = (np.cumsum(pro_next_edges)/sum(pro_next_edges)).tolist()

        #Shift
        dict_ep_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of end points
        dict_ep_base = {cellname:dict() for cellname in cell_list_sorted}
        dict_ep_center = dict() #center coordinates of end points in each branch

        df_top_x = df_bins_top.copy() # x coordinates in top line
        df_top_y = df_bins_top.copy() # y coordinates in top line
        df_base_x = df_bins_base.copy() # x coordinates in base line
        df_base_y = df_bins_base.copy() # y coordinates in base line

        for edge_i in bfs_edges:
            id_wins = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==edge_i]
            prev_node = dict_tree[edge_i[0]]['prev']
            if(prev_node == ''):
                x_st = 0
                if(flat_tree.degree(node_start)>1):
                    id_wins = id_wins[1:]
            else:
                id_wins = id_wins[1:] # remove the overlapped window
                x_st = dict_ep_center[(prev_node,edge_i[0])][0] - step_w
            y_st = dict_shift_dist[edge_i]
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',map(lambda x: 'win' + str(x), id_wins)]
                py_top = df_bins_top.loc[cellname,map(lambda x: 'win' + str(x), id_wins)]
                px_top_prime = x_st  + px_top
                py_top_prime = y_st  + py_top
                dict_ep_top[cellname][edge_i] = [px_top_prime[-1],py_top_prime[-1]]
                df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = px_top_prime
                df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',map(lambda x: 'win' + str(x), id_wins)]
                py_base = df_bins_base.loc[cellname,map(lambda x: 'win' + str(x), id_wins)]
                px_base_prime = x_st + px_base
                py_base_prime = y_st + py_base
                dict_ep_base[cellname][edge_i] = [px_base_prime[-1],py_base_prime[-1]]
                df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = px_base_prime
                df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins)] = py_base_prime
            dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

        id_wins_start = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==(node_start,node_start)]
        if(len(id_wins_start)>0):
            mean_shift_dist = np.mean([dict_shift_dist[(node_start,x)] \
                                    for x in dict_forest[cell_list_sorted[0]][node_start]['next']])
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',map(lambda x: 'win' + str(x), id_wins_start)]
                py_top = df_bins_top.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)]
                px_top_prime = 0  + px_top
                py_top_prime = mean_shift_dist  + py_top
                df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = px_top_prime
                df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',map(lambda x: 'win' + str(x), id_wins_start)]
                py_base = df_bins_base.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)]
                px_base_prime = 0 + px_base
                py_base_prime = mean_shift_dist + py_base
                df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = px_base_prime
                df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_start)] = py_base_prime

        #determine joints points
        dict_joint_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        dict_joint_base = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        if(flat_tree.degree(node_start)==1):
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1]
        else:
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1 and x[0]!=(node_start,node_start)]
            id_joints.insert(0,1)
        for id_j in id_joints:
            joint_edges = df_bins_cumsum_norm.loc['edge','win'+str(id_j)]
            for id_div,edge_i in enumerate(joint_edges[1:]):
                id_wins = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x==[edge_i]]
                for cellname in cell_list_sorted:
                    if(len(dict_forest[cellname][edge_i[0]]['div'])>0):
                        prev_node_top_x = df_top_x.loc[cellname,'win'+str(id_j)]
                        prev_node_top_y = df_top_y.loc[cellname,'win'+str(id_j)]
                        prev_node_base_x = df_base_x.loc[cellname,'win'+str(id_j)]
                        prev_node_base_y = df_base_y.loc[cellname,'win'+str(id_j)]
                        div = dict_forest[cellname][edge_i[0]]['div']
                        if(id_div==0):
                            px_top_prime_st = prev_node_top_x
                            py_top_prime_st = prev_node_top_y
                        else:
                            px_top_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x)*div[id_div-1]
                            py_top_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y)*div[id_div-1]
                        px_base_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x)*div[id_div]
                        py_base_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y)*div[id_div]
                        df_top_x.loc[cellname,'win'+str(id_wins[0])] = px_top_prime_st
                        df_top_y.loc[cellname,'win'+str(id_wins[0])] = py_top_prime_st
                        df_base_x.loc[cellname,'win'+str(id_wins[0])] = px_base_prime_st
                        df_base_y.loc[cellname,'win'+str(id_wins[0])] = py_base_prime_st
                        dict_joint_top[cellname][edge_i] = np.array([px_top_prime_st,py_top_prime_st])
                        dict_joint_base[cellname][edge_i] = np.array([px_base_prime_st,py_base_prime_st])

        dict_tree_copy = deepcopy(dict_tree)
        dict_paths_top,dict_paths_base = find_paths(dict_tree_copy,bfs_nodes)

        #identify boundary of each edge
        dict_edge_bd = dict()
        for edge_i in bfs_edges:
            id_wins = [i for i,x in enumerate(df_top_x.loc['edge',:]) if edge_i in x]
            dict_edge_bd[edge_i] = [df_top_x.iloc[0,id_wins[0]],df_top_x.iloc[0,id_wins[-1]]]

        x_smooth = np.unique(np.arange(min(df_base_x.iloc[0,:]),max(df_base_x.iloc[0,:]),step = step_w/20).tolist() \
                    + [max(df_base_x.iloc[0,:])]).tolist()
        x_joints = df_top_x.iloc[0,id_joints].tolist()
        #replace nearest value in x_smooth by x axis of joint points
        for x in x_joints:
            x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

        dict_smooth_linear = {cellname:{'top':dict(),'base':dict()} for cellname in cell_list_sorted}
        #interpolation
        for edge_i_top in dict_paths_top.keys():
            path_i_top = dict_paths_top[edge_i_top]
            id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_top))]
            if(flat_tree.degree(node_start)>1 and \
               edge_i_top==(node_start,dict_forest[cell_list_sorted[0]][node_start]['next'][0])):
                id_wins_top.insert(0,1)
                id_wins_top.insert(0,0)
            for cellname in cell_list_sorted:
                x_top = df_top_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_top)].tolist()
                y_top = df_top_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_top)].tolist()
                f_top_linear = interpolate.interp1d(x_top, y_top, kind = 'linear')
                x_top_new = [x for x in x_smooth if (x>=x_top[0]) and (x<=x_top[-1])] + [x_top[-1]]
                x_top_new = np.unique(x_top_new).tolist()
                y_top_new_linear = f_top_linear(x_top_new)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node],path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_top_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_linear[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected],\
                                                                         np.array(y_top_new_linear)[id_selected]],index=['x','y'])
        for edge_i_base in dict_paths_base.keys():
            path_i_base = dict_paths_base[edge_i_base]
            id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_base))]
            if(flat_tree.degree(node_start)>1 and \
               edge_i_base==(node_start,dict_forest[cell_list_sorted[0]][node_start]['next'][-1])):
                id_wins_base.insert(0,1)
                id_wins_base.insert(0,0)
            for cellname in cell_list_sorted:
                x_base = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins_base)].tolist()
                y_base = df_base_y.loc[cellname,map(lambda x: 'win' + str(x), id_wins_base)].tolist()
                f_base_linear = interpolate.interp1d(x_base, y_base, kind = 'linear')
                x_base_new = [x for x in x_smooth if (x>=x_base[0]) and (x<=x_base[-1])] + [x_base[-1]]
                x_base_new = np.unique(x_base_new).tolist()
                y_base_new_linear = f_base_linear(x_base_new)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node],path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_base_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_linear[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected],\
                                                                          np.array(y_base_new_linear)[id_selected]],index=['x','y'])

        #searching for edges which cell exists based on the linear interpolation
        dict_edges_CE = {cellname:[] for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                if(sum(abs(dict_smooth_linear[cellname]['top'][edge_i].loc['y'] - \
                       dict_smooth_linear[cellname]['base'][edge_i].loc['y']) > 1e-12)):
                    dict_edges_CE[cellname].append(edge_i)


        #determine paths which cell exists
        dict_paths_CE_top = {cellname:{} for cellname in cell_list_sorted}
        dict_paths_CE_base = {cellname:{} for cellname in cell_list_sorted}
        dict_forest_CE = dict()
        for cellname in cell_list_sorted:
            edges_cn = dict_edges_CE[cellname]
            nodes = [nodename for nodename in bfs_nodes if nodename in set(itertools.chain(*edges_cn))]
            dict_forest_CE[cellname] = {nodename:{'prev':"",'next':[]} for nodename in nodes}
            for node_i in nodes:
                prev_node = dict_tree[node_i]['prev']
                if((prev_node,node_i) in edges_cn):
                    dict_forest_CE[cellname][node_i]['prev'] = prev_node
                next_nodes = dict_tree[node_i]['next']
                for x in next_nodes:
                    if (node_i,x) in edges_cn:
                        (dict_forest_CE[cellname][node_i]['next']).append(x)
            dict_paths_CE_top[cellname],dict_paths_CE_base[cellname] = find_paths(dict_forest_CE[cellname],nodes)


        dict_smooth_new = deepcopy(dict_smooth_linear)
        for cellname in cell_list_sorted:
            paths_CE_top = dict_paths_CE_top[cellname]
            for edge_i_top in paths_CE_top.keys():
                path_i_top = paths_CE_top[edge_i_top]
                edges_top = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_top))]
                id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_top))]

                x_top = []
                y_top = []
                for e_t in edges_top:
                    if(e_t == edges_top[-1]):
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].loc['y']
                        px = dict_smooth_linear[cellname]['top'][e_t].loc['x']
                    else:
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].iloc[1,:-1]
                        px = dict_smooth_linear[cellname]['top'][e_t].iloc[0,:-1]
                    x_top = x_top + px.tolist()
                    y_top = y_top + py_top_linear.tolist()
                x_top_new = x_top
                y_top_new = savgol_filter(y_top,11,polyorder=1)
                for id_node in range(len(path_i_top)-1):
                    edge_i = (path_i_top[id_node],path_i_top[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_top_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_new[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected],\
                                                                         np.array(y_top_new)[id_selected]],index=['x','y'])

            paths_CE_base = dict_paths_CE_base[cellname]
            for edge_i_base in paths_CE_base.keys():
                path_i_base = paths_CE_base[edge_i_base]
                edges_base = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_base))]
                id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if set(np.unique(x)).issubset(set(path_i_base))]

                x_base = []
                y_base = []
                for e_b in edges_base:
                    if(e_b == edges_base[-1]):
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].loc['y']
                        px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                    else:
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].iloc[1,:-1]
                        px = dict_smooth_linear[cellname]['base'][e_b].iloc[0,:-1]
                    x_base = x_base + px.tolist()
                    y_base = y_base + py_base_linear.tolist()
                x_base_new = x_base
                y_base_new = savgol_filter(y_base,11,polyorder=1)
                for id_node in range(len(path_i_base)-1):
                    edge_i = (path_i_base[id_node],path_i_base[id_node+1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x,x in enumerate(x_base_new) if x>=edge_i_bd[0] and x<=edge_i_bd[1]]
                    dict_smooth_new[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected],\
                                                                          np.array(y_base_new)[id_selected]],index=['x','y'])

        #find all edges of polygon
        poly_edges = []
        dict_tree_copy = deepcopy(dict_tree)
        cur_node = node_start
        next_node = dict_tree_copy[cur_node]['next'][0]
        dict_tree_copy[cur_node]['next'].pop(0)
        poly_edges.append((cur_node,next_node))
        cur_node = next_node
        while(not(next_node==node_start and cur_node == dict_tree[node_start]['next'][-1])):
            while(len(dict_tree_copy[cur_node]['next'])!=0):
                next_node = dict_tree_copy[cur_node]['next'][0]
                dict_tree_copy[cur_node]['next'].pop(0)
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[node_start]['next'][-1] and next_node==node_start):
                    break
                cur_node = next_node
            while(len(dict_tree_copy[cur_node]['next'])==0):
                next_node = dict_tree_copy[cur_node]['prev']
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[node_start]['next'][-1] and next_node==node_start):
                    break
                cur_node = next_node


        verts = {cellname: np.empty((0,2)) for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in poly_edges:
                if edge_i in bfs_edges:
                    x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                    y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                    pxy = np.array([x_top,y_top]).T
                else:
                    edge_i = (edge_i[1],edge_i[0])
                    x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                    y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                    x_base = x_base[::-1]
                    y_base = y_base[::-1]
                    pxy = np.array([x_base,y_base]).T
                verts[cellname] = np.vstack((verts[cellname],pxy))

        dict_extent = {'xmin':"",'xmax':"",'ymin':"",'ymax':""}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
                xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
                ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
                ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
                if(dict_extent['xmin']==""):
                    dict_extent['xmin'] = xmin
                else:
                    if(xmin < dict_extent['xmin']) :
                        dict_extent['xmin'] = xmin

                if(dict_extent['xmax']==""):
                    dict_extent['xmax'] = xmax
                else:
                    if(xmax > dict_extent['xmax']):
                        dict_extent['xmax'] = xmax

                if(dict_extent['ymin']==""):
                    dict_extent['ymin'] = ymin
                else:
                    if(ymin < dict_extent['ymin']):
                        dict_extent['ymin'] = ymin

                if(dict_extent['ymax']==""):
                    dict_extent['ymax'] = ymax
                else:
                    if(ymax > dict_extent['ymax']):
                        dict_extent['ymax'] = ymax


        for gene_name in genes:
            #calculate gradient image
            #image array
            im_nrow = 100
            im_ncol = 400
            xmin = dict_extent['xmin']
            xmax = dict_extent['xmax']
            ymin = dict_extent['ymin'] - (dict_extent['ymax'] - dict_extent['ymin'])*0.1
            ymax = dict_extent['ymax'] + (dict_extent['ymax'] - dict_extent['ymin'])*0.1
            dict_im_array = {cellname: np.zeros((im_nrow,im_ncol)) for cellname in cell_list_sorted}
            df_bins_gene = dict_genes_norm[gene_name]
            for cellname in cell_list_sorted:
                for edge_i in bfs_edges:
                    id_wins_all = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==edge_i]
                    prev_edge = ''
                    id_wins_prev = []
                    if(flat_tree.degree(node_start)>1):
                        if(edge_i == bfs_edges[0]):
                            id_wins = [0,1]
                            dict_im_array = fill_im_array(dict_im_array,df_bins_gene,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)
                        id_wins = id_wins_all
                        if(edge_i[0] == node_start):
                            prev_edge = (node_start,node_start)
                            id_wins_prev = [0,1]
                        else:
                            prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                            id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                        dict_im_array = fill_im_array(dict_im_array,df_bins_gene,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)
                    else:
                        id_wins = id_wins_all
                        if(edge_i[0]!=node_start):
                            prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                            id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                        dict_im_array = fill_im_array(dict_im_array,df_bins_gene,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)

            #clip parts according to determined polygon
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
            ax.set_title(gene_name,size=20)
            patches = []

            dict_imshow = dict()
            cmap1 = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))
            # cmap1 = mpl.colors.ListedColormap(sns.diverging_palette(250, 10,s=90,l=35, n=256))
            for cellname in cell_list_sorted:
                if(experiment=='rna-seq'):
                    vmin = 0
                    vmax = df_bins_gene.values.max()
                elif(experiment=='atac-seq'):
                    vmin = -max(abs(df_bins_gene.values.min()),df_bins_gene.values.max())
                    vmax = max(abs(df_bins_gene.values.min()),df_bins_gene.values.max())
                else:
                    print('The experiment '+experiment +' is not supported')
                    return                
                im = ax.imshow(dict_im_array[cellname], cmap=cmap1,interpolation='bicubic',\
                               extent=[xmin,xmax,ymin,ymax],vmin=vmin,vmax=vmax) 
                dict_imshow[cellname] = im
                verts_cell = verts[cellname]
                clip_path = Polygon(verts_cell, facecolor='none', edgecolor='none', closed=True)
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)
                ax.autoscale(True)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            plt.xticks(fontsize=tick_fontsize)
            plt.yticks([])
            plt.xlabel('Pseudotime',fontsize=label_fontsize)
            xloc = plt.MaxNLocator(5)
            ax.xaxis.set_major_locator(xloc)     

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad='2%')
            cbar = plt.colorbar(dict_imshow[cellname],cax=cax,orientation='vertical')
            cbar.ax.tick_params(labelsize=20)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            for cellname in cell_list_sorted:
                for edge_i in bfs_edges:
                    if(df_edge_cellnum.loc[cellname,[edge_i]].values>0):
                        ax.plot(dict_smooth_new[cellname]['top'][edge_i].loc['x'],dict_smooth_new[cellname]['top'][edge_i].loc['y'],\
                                c = 'gray',ls = 'solid',lw=0.1)
                        ax.plot(dict_smooth_new[cellname]['base'][edge_i].loc['x'],dict_smooth_new[cellname]['base'][edge_i].loc['y'],\
                                c = 'gray',ls = 'solid',lw=0.1)

            fig_xmin, fig_xmax = ax.get_xlim()
            fig_ymin, fig_ymax = ax.get_ylim()
            # manual arrowhead width and length
            fig_hw = 1./20.*(fig_ymax-fig_ymin)
            fig_hl = 1./20.*(fig_xmax-fig_xmin)
            ax.arrow(fig_xmin, fig_ymin, fig_xmax-fig_xmin, 0., fc='k', ec='k', lw = 1.0,
                     head_width=fig_hw, head_length=fig_hl, overhang = 0.3,
                     length_includes_head= True, clip_on = False)
            if(save_fig):
                plt.savefig(os.path.join(file_path_S,'stream_plot_' + slugify(gene_name) + '.' + fig_format),dpi=120)
                plt.close(fig) 

def detect_transistion_genes(adata,cutoff_spearman=0.4, cutoff_logfc = 0.25, percentile_expr=95, n_jobs = multiprocessing.cpu_count(),
                             use_precomputed=True, root='S0',preference=None):

    """Detect transition genes along one branch.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    cutoff_spearman: `float`, optional (default: 0.4)
        Between 0 and 1. The cutoff used for Spearman's rank correlation coefficient.
    cutoff_logfc: `float`, optional (default: 0.25)
        The log-transformed fold change cutoff between cells around start and end node.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of gene expression greater than 0 to filter out some extreme gene expressions. 
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run when scaling the gene expressions .
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled gene expression will be used
    root: `str`, optional (default: 'S0'): 
        The starting node
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed. 
        This will help generate the consistent results as shown in subway map and stream plot.

    Returns
    -------
    updates `adata` with the following fields.
    scaled_gene_expr: `list` (`adata.uns['scaled_gene_expr']`)
        Scaled gene expression for marker gene detection.    
    transition_genes: `dict` (`adata.uns['transition_genes']`)
        Transition genes for each branch deteced by STREAM.
    """


    file_path = os.path.join(adata.uns['workdir'],'transition_genes')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)    
    
    flat_tree = adata.uns['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    df_gene_detection = adata.obs.copy()
    df_gene_detection.rename(columns={"label": "CELL_LABEL", "branch_lam": "lam"},inplace = True)
    df_sc = pd.DataFrame(index= adata.obs_names.tolist(),
                         data = adata.raw.X,
                         columns=adata.raw.var_names.tolist())
    input_genes = adata.raw.var_names.tolist()
    #exclude genes that are expressed in fewer than min_num_cells cells
    min_num_cells = max(5,int(round(df_gene_detection.shape[0]*0.001)))
    print('Minimum number of cells expressing genes: '+ str(min_num_cells))
    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()
    if(use_precomputed and ('scaled_gene_expr' in adata.uns_keys())):
        print('Importing precomputed scaled gene expression matrix ...')
        results = adata.uns['scaled_gene_expr']        
    else:
        params = [(df_gene_detection,x,percentile_expr) for x in input_genes_expressed]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_gene_expr,params)
        pool.close()
        adata.uns['scaled_gene_expr'] = results
        
    df_gene_detection[input_genes_expressed] = pd.DataFrame(results).T
    #### TG (Transition Genes) along each branch
    dict_tg_edges = dict()
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    root_node = dict_label_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
#     all_branches = np.unique(df_gene_detection['branch_id']).tolist()
    for edge_i in bfs_edges:
        if edge_i in nx.get_edge_attributes(flat_tree,'id').values():
            df_cells_edge_i = deepcopy(df_gene_detection[df_gene_detection.branch_id==edge_i])
            df_cells_edge_i['lam_ordered'] = df_cells_edge_i['lam']
        else:
            df_cells_edge_i = deepcopy(df_gene_detection[df_gene_detection.branch_id==(edge_i[1],edge_i[0])])
            df_cells_edge_i['lam_ordered'] = flat_tree.edges[edge_i]['len'] - df_cells_edge_i['lam']
        df_cells_edge_i_sort = df_cells_edge_i.sort_values(['lam_ordered'])
        df_stat_pval_qval = pd.DataFrame(columns = ['stat','logfc','pval','qval'],dtype=float)
        for genename in input_genes_expressed:
            id_initial = range(0,int(df_cells_edge_i_sort.shape[0]*0.2))
            id_final = range(int(df_cells_edge_i_sort.shape[0]*0.8),int(df_cells_edge_i_sort.shape[0]*1))
            values_initial = df_cells_edge_i_sort.iloc[id_initial,:][genename]
            values_final = df_cells_edge_i_sort.iloc[id_final,:][genename]
            diff_initial_final = abs(values_final.mean() - values_initial.mean())
            if(diff_initial_final>0):
                logfc = np.log2(max(values_final.mean(),values_initial.mean())/(min(values_final.mean(),values_initial.mean())+diff_initial_final/1000.0))
            else:
                logfc = 0
            if(logfc>cutoff_logfc):
                df_stat_pval_qval.loc[genename] = np.nan
                df_stat_pval_qval.loc[genename,['stat','pval']] = spearmanr(df_cells_edge_i_sort.loc[:,genename],\
                                                                            df_cells_edge_i_sort.loc[:,'lam_ordered'])
                df_stat_pval_qval.loc[genename,'logfc'] = logfc
        if(df_stat_pval_qval.shape[0]==0):
            print('No Transition genes are detected in branch ' + dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]])
        else:
            p_values = df_stat_pval_qval['pval']
            q_values = multipletests(p_values, method='fdr_bh')[1]
            df_stat_pval_qval['qval'] = q_values
            dict_tg_edges[edge_i] = df_stat_pval_qval[(abs(df_stat_pval_qval.stat)>=cutoff_spearman)].sort_values(['qval'])
            dict_tg_edges[edge_i].to_csv(os.path.join(file_path,'transition_genes_'+ dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]] + '.tsv'),sep = '\t',index = True)
    adata.uns['transition_genes'] = dict_tg_edges   


def plot_transition_genes(adata,num_genes = 15,
                          save_fig=False,fig_path=None,fig_size=(12,8)):
    if(fig_path is None):
        fig_path = os.path.join(adata.uns['workdir'],'transition_genes')
    if(not os.path.exists(fig_path)):
        os.makedirs(fig_path) 

    dict_tg_edges = adata.uns['transition_genes']
    flat_tree = adata.uns['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')    
    colors = sns.color_palette("Set1", n_colors=8, desat=0.8)
    for edge_i in dict_tg_edges.keys():
        df_tg_edge_i = deepcopy(dict_tg_edges[edge_i])
        df_tg_edge_i = df_tg_edge_i.iloc[:num_genes,:]

        stat = df_tg_edge_i.stat[::-1]
        qvals = df_tg_edge_i.qval[::-1]

        pos = np.arange(df_tg_edge_i.shape[0])-1
        bar_colors = np.tile(colors[4],(len(stat),1))
        # bar_colors = repeat(colors[0],len(stat))
        id_neg = np.arange(len(stat))[np.array(stat<0)]
        bar_colors[id_neg]=colors[2]

        fig = plt.figure(figsize=(12,np.ceil(0.4*len(stat))))
        ax = fig.add_subplot(1,1,1, adjustable='box')
        ax.barh(pos,stat,align='center',height=0.8,tick_label=[''],color = bar_colors)
        ax.set_xlabel('Spearman Correlation Coefficient')
        ax.set_title("branch " + dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]])

        adjust_spines(ax, ['bottom'])
        ax.spines['left'].set_position('center')
        ax.spines['left'].set_color('none')
        ax.set_xlim((-1,1))
        ax.set_ylim((min(pos)-1,max(pos)+1))

        rects = ax.patches
        for i,rect in enumerate(rects):
            if(stat[i]>0):
                alignment = {'horizontalalignment': 'left', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()+0.02, rect.get_y() + rect.get_height()/2.0, \
                        qvals.index[i],fontsize=12,**alignment)
                ax.text(rect.get_x()+0.02, rect.get_y()+rect.get_height()/2.0, \
                        "{:.2E}".format(Decimal(str(qvals[i]))),color='black',fontsize=9,**alignment)
            else:
                alignment = {'horizontalalignment': 'right', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()-0.02, rect.get_y()+rect.get_height()/2.0, \
                        qvals.index[i],fontsize=12,**alignment)
                ax.text(rect.get_x()-0.02, rect.get_y()+rect.get_height()/2.0, \
                        "{:.2E}".format(Decimal(str(qvals[i]))),color='w',fontsize=9,**alignment)
        if(save_fig):        
            plt.savefig(os.path.join(fig_path,'transition_genes_'+ dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]]+'.pdf'),\
                        pad_inches=1,bbox_inches='tight')
            plt.close(fig)    


def detect_de_genes(adata,cutoff_zscore=2,cutoff_logfc = 0.25,percentile_expr=95,n_jobs = multiprocessing.cpu_count(),
                    use_precomputed=True, root='S0',preference=None):

    """Detect differentially expressed genes between different sub-branches.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    cutoff_zscore: `float`, optional (default: 2)
        The z-score cutoff used for Mann–Whitney U test.
    cutoff_logfc: `float`, optional (default: 0.25)
        The log-transformed fold change cutoff between a pair of branches.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of gene expression greater than 0 to filter out some extreme gene expressions. 
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run when scaling the gene expressions .
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled gene expression will be used
    root: `str`, optional (default: 'S0'): 
        The starting node
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed. 
        This will help generate the consistent results as shown in subway map and stream plot.

    Returns
    -------
    updates `adata` with the following fields.
    scaled_gene_expr: `list` (`adata.uns['scaled_gene_expr']`)
        Scaled gene expression for marker gene detection.    
    de_genes_greater: `dict` (`adata.uns['de_genes_greater']`)
        DE(differentially expressed) genes for each pair of branches deteced by STREAM. 
        Store the genes that have higher expression on the former part of one branch pair.
    de_genes_less: `dict` (`adata.uns['de_genes_less']`)
        DE(differentially expressed) genes for each pair of branches deteced by STREAM. 
        Store the genes that have higher expression on the latter part of one branch pair.
    """


    file_path = os.path.join(adata.uns['workdir'],'de_genes')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)    

    flat_tree = adata.uns['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    df_gene_detection = adata.obs.copy()
    df_gene_detection.rename(columns={"label": "CELL_LABEL", "branch_lam": "lam"},inplace = True)
    df_sc = pd.DataFrame(index= adata.obs_names.tolist(),
                         data = adata.raw.X,
                         columns=adata.raw.var_names.tolist())
    input_genes = adata.raw.var_names.tolist()
    #exclude genes that are expressed in fewer than min_num_cells cells
    min_num_cells = max(5,int(round(df_gene_detection.shape[0]*0.001)))
    print('Minimum number of cells expressing genes: '+ str(min_num_cells))
    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()

    if(use_precomputed and ('scaled_gene_expr' in adata.uns_keys())):
        print('Importing precomputed scaled gene expression matrix ...')
        results = adata.uns['scaled_gene_expr']          
    else:
        params = [(df_gene_detection,x,percentile_expr) for x in input_genes_expressed]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_gene_expr,params)
        pool.close()
        adata.uns['scaled_gene_expr'] = results

    df_gene_detection[input_genes_expressed] = pd.DataFrame(results).T    

    #### DE (Differentially expressed genes) between sub-branches
    dict_de_greater = dict()
    dict_de_less = dict()
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    root_node = dict_label_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
    pairs_branches = list()
#     all_branches = np.unique(df_gene_detection['branch_id']).tolist()
    for node_i in dict_node_state.keys():
        neighbor_branches = [x for x in bfs_edges if node_i in x]
        if(len(neighbor_branches)>1):
            pairs_branches += list(itertools.combinations(neighbor_branches,r=2))
    for pair_i in pairs_branches:
        if(pair_i[0] in nx.get_edge_attributes(flat_tree,'id').values()):
            df_cells_sub1 = df_gene_detection[df_gene_detection.branch_id==pair_i[0]]
        else:
            df_cells_sub1 = df_gene_detection[df_gene_detection.branch_id==(pair_i[0][1],pair_i[0][0])]
        if(pair_i[1] in nx.get_edge_attributes(flat_tree,'id').values()):
            df_cells_sub2 = df_gene_detection[df_gene_detection.branch_id==pair_i[1]]  
        else:
            df_cells_sub2 = df_gene_detection[df_gene_detection.branch_id==(pair_i[1][1],pair_i[1][0])]
        #only use Mann-Whitney U test when the number of observation in each sample is > 20
        if(df_cells_sub1.shape[0]>20 and df_cells_sub2.shape[0]>20):
            df_de_pval_qval = pd.DataFrame(columns = ['z_score','U','logfc','mean_up','mean_down','pval','qval'],dtype=float)
            for genename in input_genes_expressed:
                sub1_values = df_cells_sub1.loc[:,genename].tolist()
                sub2_values = df_cells_sub2.loc[:,genename].tolist()
                diff_mean = abs(np.mean(sub1_values) - np.mean(sub2_values))
                if(diff_mean>0):
                    logfc = np.log2(max(np.mean(sub1_values),np.mean(sub2_values))/(min(np.mean(sub1_values),np.mean(sub2_values))+diff_mean/1000.0))
                else:
                    logfc = 0
                if(logfc>cutoff_logfc):
                    df_de_pval_qval.loc[genename] = np.nan
                    df_de_pval_qval.loc[genename,['U','pval']] = mannwhitneyu(sub1_values,sub2_values,alternative='two-sided')
                    df_de_pval_qval.loc[genename,'logfc'] = logfc
                    df_de_pval_qval.loc[genename,'mean_up'] = np.mean(sub1_values)
                    df_de_pval_qval.loc[genename,'mean_down'] = np.mean(sub2_values)
                    sub1_sub2_values = sub1_values + sub2_values
                    len_sub1 = len(sub1_values)
                    len_sub2 = len(sub2_values)
                    len_sub = len_sub1+len_sub2
                    ranks_values = stats.rankdata(sub1_sub2_values)
                    ranks_id, ranks_count = np.unique(ranks_values,return_counts=True)
                    sum_k = 0
                    for i,t in enumerate(ranks_count):
                        sum_k = sum_k + (t**3-t)/float(len_sub*(len_sub-1))
                    mu_U = (len_sub1*len_sub2)/2.0
                    sigma_U = np.sqrt(len_sub1*len_sub2*(len_sub+1-sum_k)/12.0)
                    df_de_pval_qval.loc[genename,'z_score'] = (df_de_pval_qval.loc[genename,'U']-mu_U)/sigma_U
            if(df_de_pval_qval.shape[0]==0):
                print('No DE genes are detected between branches ' + dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]]+\
                      ' and '+dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]])
            else:
                p_values = df_de_pval_qval['pval']
                q_values = multipletests(p_values, method='fdr_bh')[1]
                df_de_pval_qval['qval'] = q_values
                dict_de_greater[pair_i] = df_de_pval_qval[(abs(df_de_pval_qval['z_score'])>cutoff_zscore)&
                                                          (df_de_pval_qval['z_score']>0)].sort_values(['z_score'],ascending=False)
                dict_de_greater[pair_i].to_csv(os.path.join(file_path,'de_genes_greater_'+dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] + ' and '\
                                        + dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]] + '.tsv'),sep = '\t',index = True)
                dict_de_less[pair_i] = df_de_pval_qval[(abs(df_de_pval_qval['z_score'])>cutoff_zscore)&
                                                       (df_de_pval_qval['z_score']<0)].sort_values(['z_score'])
                dict_de_less[pair_i].to_csv(os.path.join(file_path,'de_genes_less_'+dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] + ' and '\
                                     + dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]] + '.tsv'),sep = '\t',index = True)   
        else:
            print('There are not sufficient cells (should be greater than 20) between branches '+\
                  dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] +' and '+\
                  dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]]+ '. fold_change is calculated')
            df_de_pval_qval = pd.DataFrame(columns = ['logfc','mean_up','mean_down'],dtype=float)
            for genename in input_genes_expressed:
                sub1_values = df_cells_sub1.loc[:,genename].tolist()
                sub2_values = df_cells_sub2.loc[:,genename].tolist()
                diff_mean = abs(np.mean(sub1_values) - np.mean(sub2_values))
                if(diff_mean>0):
                    logfc = np.log2(max(np.mean(sub1_values),np.mean(sub2_values))/(min(np.mean(sub1_values),np.mean(sub2_values))+diff_mean/1000.0))
                else:
                    logfc = 0
                if(logfc>cutoff_logfc):
                    df_de_pval_qval.loc[genename] = np.nan
                    # #make sure the largest fold change is 5
                    # df_de_pval_qval.loc[genename,'fold_change'] = np.log2((np.mean(sub1_values)+1/24.0)/(np.mean(sub2_values)+1/24.0))
                    df_de_pval_qval.loc[genename,'logfc'] = logfc
                    df_de_pval_qval.loc[genename,'mean_up'] = np.mean(sub1_values)
                    df_de_pval_qval.loc[genename,'mean_down'] = np.mean(sub2_values)
            if(df_de_pval_qval.shape[0]==0):
                print('No DE genes are detected between branches ' + dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]]+\
                      ' and '+dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]])
            else:
                dict_de_greater[pair_i] = df_de_pval_qval[(abs(df_de_pval_qval['logfc'])>cutoff_logfc)&
                                                          (df_de_pval_qval['logfc']>0)].sort_values(['logfc'],ascending=False)
                dict_de_greater[pair_i].to_csv(os.path.join(file_path,'de_genes_greater_'+dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] + ' and '\
                                        + dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]] + '.tsv'),sep = '\t',index = True)                
                dict_de_less[pair_i] = df_de_pval_qval[(abs(df_de_pval_qval['logfc'])>cutoff_logfc)&
                                                       (df_de_pval_qval['logfc']<0)].sort_values(['logfc'])
                dict_de_less[pair_i].to_csv(os.path.join(file_path,'de_genes_less_'+dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] + ' and '\
                                     + dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]] + '.tsv'),sep = '\t',index = True)   
    adata.uns['de_genes_greater'] = dict_de_greater
    adata.uns['de_genes_less'] = dict_de_less


def plot_de_genes(adata,num_genes = 15,cutoff_zscore=2,cutoff_logfc = 0.25,
                  save_fig=False,fig_path=None,fig_size=(12,8)):
    if(fig_path is None):
        fig_path = os.path.join(adata.uns['workdir'],'de_genes')
    if(not os.path.exists(fig_path)):
        os.makedirs(fig_path)  

    dict_de_greater = adata.uns['de_genes_greater']
    dict_de_less = adata.uns['de_genes_less']
    flat_tree = adata.uns['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    colors = sns.color_palette("Set1", n_colors=8, desat=0.8)
    for sub_edges_i in dict_de_greater.keys():
        fig = plt.figure(figsize=(20,12))
        gs = gridspec.GridSpec(2,1)
        ax = fig.add_subplot(gs[0],adjustable='box')
        if('z_score' in dict_de_greater[sub_edges_i].columns):
            if(not dict_de_greater[sub_edges_i].empty):
                val_greater = dict_de_greater[sub_edges_i].iloc[:num_genes,:]['z_score'].values  # the bar lengths
                pos_greater = np.arange(dict_de_greater[sub_edges_i].iloc[:num_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_greater = np.repeat(0,num_genes)
                pos_greater = np.arange(num_genes)-1
            ax.bar(pos_greater,val_greater, align='center',color = colors[0])
            ax.plot([pos_greater[0]-1,pos_greater[-1]+1], [cutoff_zscore, cutoff_zscore], "k--",lw=2)
            q_vals = dict_de_greater[sub_edges_i].iloc[:num_genes,:]['qval'].values
            for i, q in enumerate(q_vals):
                alignment = {'horizontalalignment': 'center', 'verticalalignment': 'bottom'}
                ax.text(pos_greater[i], val_greater[i]+.1, \
                        "{:.2E}".format(Decimal(str(q))),color='black',fontsize=15,**alignment)
            plt.xticks(pos_greater,dict_de_greater[sub_edges_i].index,rotation=90)
            ax.set_ylim(0,max(val_greater)+1.5)
            ax.set_ylabel('z_score')
            ax.set_title('DE genes between branches ' + dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and ' + \
                         dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]])
            ax1 = fig.add_subplot(gs[1], adjustable='box')
            if(not dict_de_less[sub_edges_i].empty):
                val_less = dict_de_less[sub_edges_i].iloc[:num_genes,:]['z_score'].values  # the bar lengths
                pos_less = np.arange(dict_de_less[sub_edges_i].iloc[:num_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_less = np.repeat(0,num_genes)
                pos_less = np.arange(num_genes)-1
            ax1.bar(pos_less,val_less, align='center',color = colors[1])
            ax1.plot([pos_less[0]-1,pos_less[-1]+1], [-cutoff_zscore, -cutoff_zscore], "k--",lw=2)
            q_vals = dict_de_less[sub_edges_i].iloc[:num_genes,:]['qval'].values
            for i, q in enumerate(q_vals):
                alignment = {'horizontalalignment': 'center', 'verticalalignment': 'top'}
                ax1.text(pos_less[i], val_less[i]-.1, \
                        "{:.2E}".format(Decimal(str(q))),color='black',fontsize=15,**alignment)
            plt.xticks(pos_less,dict_de_less[sub_edges_i].index)
            ax1.set_ylim(min(val_less)-1.5,0)
            ax1.set_xticklabels(dict_de_less[sub_edges_i].index,rotation=90)
            ax1.set_ylabel('z_score')

            ax.set_xlim(-2,14)
            ax1.set_xlim(-2,14)
            ax1.xaxis.tick_top()
            plt.tight_layout()
            if(save_fig):
                plt.savefig(os.path.join(fig_path,'de_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.pdf'),pad_inches=1,bbox_inches='tight')
                plt.close(fig)
        else:
            if(not dict_de_greater[sub_edges_i].empty):
                val_greater = dict_de_greater[sub_edges_i].iloc[:num_genes,:]['logfc'].values  # the bar lengths
                pos_greater = np.arange(dict_DE_greater[sub_edges_i].iloc[:num_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_greater = np.repeat(0,num_genes)
                pos_greater = np.arange(num_genes)-1
            ax.bar(pos_greater,val_greater, align='center',color = colors[0])
            ax.plot([pos_greater[0]-1,pos_greater[-1]+1], [cutoff_logfc, cutoff_logfc], "k--",lw=2)
            plt.xticks(pos_greater,dict_de_greater[sub_edges_i].index,rotation=90)
            ax.set_ylim(0,max(val_greater)+1.5)
            ax.set_ylabel('log_fold_change')
            ax.set_title('DE genes between branches ' + dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and ' + \
                         dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]])
            ax1 = fig.add_subplot(gs[1], adjustable='box')
            if(not dict_de_less[sub_edges_i].empty):
                val_less = dict_de_less[sub_edges_i].iloc[:num_genes,:]['logfc'].values  # the bar lengths
                pos_less = np.arange(dict_de_less[sub_edges_i].iloc[:num_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_less = np.repeat(0,num_genes)
                pos_less = np.arange(num_genes)-1
            ax1.bar(pos_less,val_less, align='center',color = colors[1])
            ax1.plot([pos_less[0]-1,pos_less[-1]+1], [-cutoff_logfc, -cutoff_logfc], "k--",lw=2)
            plt.xticks(pos_less,dict_de_less[sub_edges_i].index)
            ax1.set_ylim(min(val_less)-1.5,0)
            ax1.set_xticklabels(dict_de_less[sub_edges_i].index,rotation=90)
            ax1.set_ylabel('log_fold_change')

            ax.set_xlim(-2,14)
            ax1.set_xlim(-2,14)
            ax1.xaxis.tick_top()
            plt.tight_layout()
            if(save_fig):
                plt.savefig(os.path.join(fig_path,'de_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.pdf'),pad_inches=1,bbox_inches='tight')
                plt.close(fig) 


def detect_leaf_genes(adata,cutoff_zscore=1.5,cutoff_pvalue=1e-2,percentile_expr=95,n_jobs = multiprocessing.cpu_count(),
                      use_precomputed=True, root='S0',preference=None):
    """Detect leaf genes for each branch.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    cutoff_zscore: `float`, optional (default: 1.5)
        The z-score cutoff used for mean values of all leaf branches.
    cutoff_pvalue: `float`, optional (default: 1e-2)
        The p value cutoff used for Kruskal-Wallis H-test and post-hoc pairwise Conover’s test.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of gene expression greater than 0 to filter out some extreme gene expressions. 
    n_jobs: `int`, optional (default: all available cpus)
        The number of parallel jobs to run when scaling the gene expressions .
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled gene expression will be used
    root: `str`, optional (default: 'S0'): 
        The starting node
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed. 
        This will help generate the consistent results as shown in subway map and stream plot.

    Returns
    -------
    updates `adata` with the following fields.
    scaled_gene_expr: `list` (`adata.uns['scaled_gene_expr']`)
        Scaled gene expression for marker gene detection.    
    leaf_genes_all: `pandas.core.frame.DataFrame` (`adata.uns['leaf_genes_all']`)
        All the leaf genes from all leaf branches.
    leaf_genes: `dict` (`adata.uns['leaf_genes']`)
        Leaf genes for each branch.
    """


    file_path = os.path.join(adata.uns['workdir'],'leaf_genes')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)    

    flat_tree = adata.uns['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    df_gene_detection = adata.obs.copy()
    df_gene_detection.rename(columns={"label": "CELL_LABEL", "branch_lam": "lam"},inplace = True)
    df_sc = pd.DataFrame(index= adata.obs_names.tolist(),
                         data = adata.raw.X,
                         columns=adata.raw.var_names.tolist())
    input_genes = adata.raw.var_names.tolist()
    #exclude genes that are expressed in fewer than min_num_cells cells
    min_num_cells = max(5,int(round(df_gene_detection.shape[0]*0.001)))
    print('Minimum number of cells expressing genes: '+ str(min_num_cells))
    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()

    if(use_precomputed and ('scaled_gene_expr' in adata.uns_keys())):
        print('Importing precomputed scaled gene expression matrix ...')
        results = adata.uns['scaled_gene_expr']          
    else:
        params = [(df_gene_detection,x,percentile_expr) for x in input_genes_expressed]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_gene_expr,params)
        pool.close()
        adata.uns['scaled_gene_expr'] = results

    df_gene_detection[input_genes_expressed] = pd.DataFrame(results).T    

    #### find marker genes that are specific to one leaf branch
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    root_node = dict_label_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
    leaves = [k for k,v in flat_tree.degree() if v==1]
    leaf_edges = [x for x in bfs_edges if (x[0] in leaves) or (x[1] in leaves)]    

    df_gene_detection['bfs_edges'] = df_gene_detection['branch_id']
    df_gene_detection.astype('object')
    for x in df_gene_detection['branch_id'].unique():
        id_ = df_gene_detection[df_gene_detection['branch_id']==x].index
        if x not in bfs_edges:
            df_gene_detection.loc[id_,'bfs_edges'] =pd.Series(index=id_,data=[(x[1],x[0])]*len(id_))
    
    df_leaf_genes = pd.DataFrame(columns=['zscore','H_statistic','H_pvalue']+leaf_edges)
    for gene in input_genes_expressed:
        meann_values = df_gene_detection[['bfs_edges',gene]].groupby(by = 'bfs_edges')[gene].mean()
        br_values = df_gene_detection[['bfs_edges',gene]].groupby(by = 'bfs_edges')[gene].apply(list)
        leaf_mean_values = meann_values[leaf_edges]
        leaf_mean_values.sort_values(inplace=True)
        leaf_br_values = br_values[leaf_edges]
        if(leaf_mean_values.shape[0]<=2):
            print('There are not enough leaf branches')
        else:
            zscores = stats.zscore(leaf_mean_values)
            if(abs(zscores)[abs(zscores)>cutoff_zscore].shape[0]>=1):
                if(any(zscores>cutoff_zscore)):
                    cand_br = leaf_mean_values.index[-1]
                    cand_zscore = zscores[-1]
                else:
                    cand_br = leaf_mean_values.index[0]
                    cand_zscore = zscores[0]
                list_br_values = [leaf_br_values[x] for x in leaf_edges]
                kurskal_statistic,kurskal_pvalue = stats.kruskal(*list_br_values)
                if(kurskal_pvalue<cutoff_pvalue):  
                    df_conover_pvalues= posthoc_conover(df_gene_detection[[x in leaf_edges for x in df_gene_detection['bfs_edges']]], 
                                                       val_col=gene, group_col='bfs_edges', p_adjust = 'fdr_bh')
                    cand_conover_pvalues = df_conover_pvalues[~df_conover_pvalues.columns.isin([cand_br])][cand_br]
                    if(all(cand_conover_pvalues < cutoff_pvalue)):
                        df_leaf_genes.loc[gene,:] = "Null"
                        df_leaf_genes.loc[gene,['zscore','H_statistic','H_pvalue']] = [cand_zscore,kurskal_statistic,kurskal_pvalue]
                        df_leaf_genes.loc[gene,cand_conover_pvalues.index] = cand_conover_pvalues
    df_leaf_genes.rename(columns={x:dict_node_state[x[0]]+dict_node_state[x[1]]+'_pvalue' for x in leaf_edges},inplace=True)
    df_leaf_genes.sort_values(by='H_pvalue',inplace=True)
    df_leaf_genes.to_csv(os.path.join(file_path,'leaf_genes.tsv'),sep = '\t',index = True)
    dict_leaf_genes = dict()
    for x in leaf_edges:
        dict_leaf_genes[x] = df_leaf_genes[df_leaf_genes[dict_node_state[x[0]]+dict_node_state[x[1]]+'_pvalue']=="Null"]
        dict_leaf_genes[x].to_csv(os.path.join(file_path,'leaf_genes'+dict_node_state[x[0]]+'_'+dict_node_state[x[1]] + '.tsv'),sep = '\t',index = True)
    adata.uns['leaf_genes_all'] = df_leaf_genes
    adata.uns['leaf_genes'] = dict_leaf_genes


def map_new_data(adata,adata_new,feature='var_genes',method='mlle',use_radius=True):
    """ Map new data to the inferred trajectories
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    adata_new: AnnData
        Annotated data matrix for new data (to be mapped).
    feature: `str`, optional (default: 'var_genes')
        Choose from {{'var_genes','all'}}
        Feature used for mapping. This should be consistent with the feature used for inferring trajectories
        'var_genes': most variable genes
        'all': all genes
    method: `str`, optional (default: 'mlle')
        Choose from {{'mlle','umap','pca'}}
        Method used for mapping. This should be consistent with the dimension reduction method used for inferring trajectories
        'mlle': Modified locally linear embedding algorithm
        'umap': Uniform Manifold Approximation and Projection
        'pca': Principal component analysis
    use_radius: `bool`, optional (default: True)
        If True, when searching for the neighbors for each cell in MLLE space, STREAM uses a fixed radius instead of a fixed number of cells.

    Returns
    -------  
    
    updates `adata_new` with the following fields.(depending on the `feature` or `method`)
    var_genes: `numpy.ndarray` (`adata_new.obsm['var_genes']`)
        Store #observations × #var_genes data matrix used mapping.
    var_genes: `pandas.core.indexes.base.Index` (`adata_new.uns['var_genes']`)
        The selected variable gene names.
    X_dr : `numpy.ndarray` (`adata_new.obsm['X_dr']`)
        A #observations × n_components data matrix after dimension reduction.
    X_mlle : `numpy.ndarray` (`adata_new.obsm['X_mlle']`)
        Store #observations × n_components data matrix after mlle.
    X_umap : `numpy.ndarray` (`adata_new.obsm['X_umap']`)
        Store #observations × n_components data matrix after umap.
    X_pca : `numpy.ndarray` (`adata_new.obsm['X_pca']`)
        Store #observations × n_components data matrix after pca.
    """

    if(feature == 'var_genes'):
        adata_new.uns['var_genes'] = adata.uns['var_genes'].copy()
        adata_new.obsm['var_genes'] = adata_new[:,adata_new.uns['var_genes']].X.copy()
        input_data = adata_new.obsm['var_genes']
    if(feature == 'all'):
        input_data = adata_new[:,adata.var.index].X
    adata_new.uns['epg'] = adata.uns['epg'].copy()
    adata_new.uns['flat_tree'] = adata.uns['flat_tree'].copy() 
    if(method == 'mlle'):
        trans = adata.uns['trans_mlle']
        if(use_radius):
            dist_nb = trans.nbrs_.kneighbors(input_data, n_neighbors=trans.n_neighbors,return_distance=True)[0]
            ind = trans.nbrs_.radius_neighbors(input_data, radius = dist_nb.max(),return_distance=False)    
            new_X_mlle = np.empty((input_data.shape[0], trans.n_components))
            for i in range(input_data.shape[0]):
                weights = barycenter_weights_modified(input_data[i], trans.nbrs_._fit_X[ind[i]],reg=trans.reg)
                new_X_mlle[i] = np.dot(trans.embedding_[ind[i]].T, weights) 
            adata_new.obsm['X_mlle_mapping'] = new_X_mlle              
        else:
            adata_new.obsm['X_mlle_mapping'] = trans.transform(input_data)
        adata_new.obsm['X_dr'] = adata_new.obsm['X_mlle_mapping'].copy()
    if(method == 'umap'):
        trans = adata.uns['trans_umap']
        adata_new.obsm['X_umap_mapping'] = trans.transform(input_data)
        adata_new.obsm['X_dr'] = adata_new.obsm['X_umap_mapping'].copy()
    if(method == 'pca'):
        trans = adata.uns['trans_pca']
        adata_new.obsm['X_pca_mapping'] = trans.transform(input_data)
        adata_new.obsm['X_dr'] = adata_new.obsm['X_pca_mapping'].copy()
    project_cells_to_epg(adata_new)
    calculate_pseudotime(adata_new)