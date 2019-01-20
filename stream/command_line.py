#!/usr/bin/env python
# -*- coding: utf-8 -*-

__tool_name__='STREAM'

print('''
   _____ _______ _____  ______          __  __ 
  / ____|__   __|  __ \|  ____|   /\   |  \/  |
 | (___    | |  | |__) | |__     /  \  | \  / |
  \___ \   | |  |  _  /|  __|   / /\ \ | |\/| |
  ____) |  | |  | | \ \| |____ / ____ \| |  | |
 |_____/   |_|  |_|  \_\______/_/    \_\_|  |_|
                                               
''')

import stream as st
import argparse
import multiprocessing
import os
from slugify import slugify
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('pdf', fonttype=42)

os.environ['KMP_DUPLICATE_LIB_OK']='True'


print('- STREAM Single-cell Trajectory Reconstruction And Mapping -')
print('Version %s\n' % st.__version__)
    

def output_for_website(adata):
    workdir = adata.uns['workdir']
    experiment = adata.uns['experiment']
    epg = adata.uns['epg']
    flat_tree = adata.uns['flat_tree']
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    dict_nodes_label = nx.get_node_attributes(flat_tree,'label')
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}

    #coordinates of cells in 3D plots
    df_sample = adata.obs[['label','label_color']].copy()
    df_coord = pd.DataFrame(adata.obsm['X_dr'],index=adata.obs_names)
    color = df_sample.sample(frac=1,random_state=100)['label_color'] 
    coord = df_coord.sample(frac=1,random_state=100)
    df_coord_cells = pd.concat([color, coord], axis=1)
    df_coord_cells.columns = ['color','D0','D1','D2']
    df_coord_cells.to_csv(os.path.join(workdir,'coord_cells.csv'),sep='\t')

    #coordinates of curves in 3D plots
    for edge_i in flat_tree.edges():
        branch_i_nodes = flat_tree.edges[edge_i]['nodes']
        branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])
        df_coord_curve_i = pd.DataFrame(branch_i_pos)
        df_coord_curve_i.to_csv(os.path.join(workdir, 'coord_curve_'+dict_nodes_label[edge_i[0]] + '_' + dict_nodes_label[edge_i[1]]+'.csv'),sep='\t',index=False)

    #coordinates of states(nodes) in 3D plots
    df_coord_states = pd.DataFrame(columns=[0,1,2])
    for x in dict_nodes_label.keys():
        df_coord_states.loc[dict_nodes_label[x]] = dict_nodes_pos[x]
    df_coord_states.sort_index(inplace=True)
    df_coord_states.to_csv(os.path.join(workdir,'coord_states.csv'),sep='\t')


    #coordinates of cells in flat tree
    workdir = adata.uns['workdir']
    df_sample = adata.obs[['label','label_color']].copy()
    df_coord = pd.DataFrame(adata.obsm['X_spring'],index=adata.obs_names)
    color = df_sample.sample(frac=1,random_state=100)['label_color'] 
    coord = df_coord.sample(frac=1,random_state=100)
    df_flat_tree_coord_cells = pd.concat([color, coord], axis=1)
    df_flat_tree_coord_cells.columns = ['color','D0','D1']
    df_flat_tree_coord_cells.to_csv(os.path.join(workdir, 'flat_tree_coord_cells.csv'),sep='\t')
 
    #nodes in flat tree
    df_nodes_flat_tree = pd.DataFrame(columns=['D0','D1'])
    for x in dict_nodes_label.keys():
        df_nodes_flat_tree.loc[dict_nodes_label[x]] = flat_tree.node[x]['pos_spring']
    df_nodes_flat_tree.sort_index(inplace=True)
    df_nodes_flat_tree.to_csv(os.path.join(workdir,'nodes.csv'),sep='\t')

    #edges in flat tree
    df_edges_flat_tree = pd.DataFrame(columns=[0,1])
    for i,edge_i in enumerate(flat_tree.edges()):
        df_edges_flat_tree.loc[i] = [dict_nodes_label[edge_i[0]],dict_nodes_label[edge_i[1]]]
        df_edges_flat_tree.to_csv(os.path.join(workdir,'edges.tsv'),sep = '\t',index = False,header=False)

    #coordinates of cells in subwaymap plot
    workdir = adata.uns['workdir']
    list_node_start = dict_label_node.keys()
    for root in list_node_start:
        df_sample = adata.obs[['label','label_color']].copy()
        df_coord = pd.DataFrame(adata.obsm['X_subwaymap_'+root],index=adata.obs_names)
        color = df_sample.sample(frac=1,random_state=100)['label_color'] 
        coord = df_coord.sample(frac=1,random_state=100)
        df_subwaymap_coord_cells = pd.concat([color, coord], axis=1)
        df_subwaymap_coord_cells.columns = ['color','D0','D1']
        df_subwaymap_coord_cells.to_csv(os.path.join(workdir,root,'subway_coord_cells.csv'),sep='\t')

        for edge_i in adata.uns['subwaymap_'+root]['edges'].keys():
            df_edge_pos =  pd.DataFrame(adata.uns['subwaymap_'+root]['edges'][edge_i])
            df_edge_pos.to_csv(os.path.join(workdir,root, 'subway_coord_line_'+dict_nodes_label[edge_i[0]] + '_' + dict_nodes_label[edge_i[1]]+'.csv'),sep='\t',index=False)

def output_for_website_subwaymap_gene(adata,gene_list):
    workdir = adata.uns['workdir']
    df_gene_expr = pd.DataFrame(index= adata.obs_names.tolist(),
                                data = adata.raw[:,gene_list].X,
                                columns=gene_list)
    list_node_start = dict_label_node.keys()
    for root in list_node_start:        
        for g in gene_list:
            if(experiment=='rna-seq'):
                gene_expr = df_gene_expr[g].copy()
                max_gene_expr = np.percentile(gene_expr[gene_expr>0],95)
                gene_expr[gene_expr>max_gene_expr] = max_gene_expr   
            elif(experiment=='atac-seq'):
                gene_expr = df_gene_expr[g].copy()
                min_gene_expr = np.percentile(gene_expr[gene_expr<0],100-95)
                max_gene_expr = np.percentile(gene_expr[gene_expr>0],95)  
                gene_expr[gene_expr>max_gene_expr] = max_gene_expr
                gene_expr[gene_expr<min_gene_expr] = min_gene_expr
            df_subwaymap_gene_expr = pd.Series(gene_expr).sample(frac=1,random_state=100)
            df_subwaymap_coord_cells_expr = pd.concat([df_subwaymap_coord_cells[['D0','D1']],df_subwaymap_gene_expr], axis=1)
            df_subwaymap_coord_cells_expr.to_csv(os.path.join(workdir,root, 'subway_coord_' + slugify(g) + '.csv'),sep='\t')


def output_cell_info(adata):
    workdir = adata.uns['workdir']
    adata.obs.to_csv(os.path.join(workdir,'cell_info.tsv'),sep = '\t',index = True)


def main():
    sns.set_style('white')
    sns.set_context('poster')
    parser = argparse.ArgumentParser(description='%s Parameters' % __tool_name__ ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--matrix", dest="input_filename",default = None,
                        help="input file name", metavar="FILE")
    parser.add_argument("-l", "--cell_labels",dest="cell_label_filename", default=None,
                        help="filename of cell labels")
    parser.add_argument("-c","--cell_labels_colors",dest="cell_label_color_filename", default=None,
                        help="filename of cell label colors")
    parser.add_argument("-s","--select_features",dest="s_method",default = 'LOESS',
                        help="LOESS,PCA or all: Select variable genes using LOESS or principal components using PCA or all the genes are kept")
    parser.add_argument("--TG","--detect_TG_genes",dest="flag_gene_TG_detection", action="store_true",
                        help="detect transition genes automatically")
    parser.add_argument("--DE","--detect_DE_genes",dest="flag_gene_DE_detection", action="store_true",
                        help="detect DE genes automatically")
    parser.add_argument("--LG","--detect_LG_genes",dest="flag_gene_LG_detection", action="store_true",
                        help="detect leaf genes automatically")    
    parser.add_argument("-g","--genes",dest="genes", default=None,
                        help="genes to visualize, it can either be filename which contains all the genes in one column or a set of gene names separated by comma")
    parser.add_argument("-p","--use_precomputed",dest="use_precomputed", action="store_true",
                        help="use precomputed data files without re-computing structure learning part")
    parser.add_argument("--new", dest="new_filename",default = None,
                        help="file name of data to be mapped")
    parser.add_argument("--new_l",dest="new_label_filename", default=None,
                        help="filename of new cell labels")
    parser.add_argument("--new_c",dest="new_label_color_filename", default=None,
                        help="filename of new cell label colors")
    parser.add_argument("--log2",dest="flag_log2", action="store_true",
                        help="perform log2 transformation")
    parser.add_argument("--norm",dest="flag_norm", action="store_true",
                        help="normalize data based on library size")
    parser.add_argument("--atac",dest="flag_atac", action="store_true",
                        help="indicate scATAC-seq data")
    parser.add_argument("--atac_counts",dest="atac_counts",default = None,
                        help="scATAC-seq counts file name", metavar="FILE")
    parser.add_argument("--atac_regions",dest="atac_regions",default = None,
                        help="scATAC-seq regions file name", metavar="FILE")
    parser.add_argument("--atac_samples",dest="atac_samples",default = None,
                        help="scATAC-seq samples file name", metavar="FILE")    
    parser.add_argument("--atac_k",dest="atac_k",type=int,default=7,
                        help="specify k-mers in scATAC-seq")
    parser.add_argument("--atac_zscore",dest="atac_zscore",default=None,
                        help="precomputed atac zscore pkl file")    
    parser.add_argument("--n_processes",dest = "n_processes",type=int, default=multiprocessing.cpu_count(),
                        help="Specify the number of processes to use. (default, all the available cores)")
    parser.add_argument("--loess_frac",dest = "loess_frac",type=float, default=0.1,
                        help="The fraction of the data used in LOESS regression")
    parser.add_argument("--loess_cutoff",dest="loess_cutoff", type = int, default=95,
                        help="the percentile used in variable gene selection based on LOESS regression")
    parser.add_argument("--pca_first_PC",dest="flag_first_PC", action="store_true",
                        help="keep first PC")
    parser.add_argument("--pca_n_PC",dest="pca_n_PC", type=int,default=15,
                        help="The number of selected PCs,it's 15 by default")
    parser.add_argument("--lle_neighbours",dest="lle_n_nb_percent", type=float,default=0.1,
                        help="LLE neighbour percent ")
    parser.add_argument("--lle_components",dest="lle_n_component", type=int, default=3,
                        help="number of components for LLE space ")
    parser.add_argument("--AP_damping_factor",dest="AP_damping_factor", type=float, default=0.75,
                        help="Affinity Propagation: damping factor")    
    parser.add_argument("--EPG_n_nodes",dest="EPG_n_nodes", type=int, default=50,
                        help=" Number of nodes for elastic principal graph")
    parser.add_argument("--EPG_n_rep",dest="EPG_n_rep", type=int, default=1,
                        help="Number of replica for constructing elastic principal graph")            
    parser.add_argument("--EPG_lambda",dest="EPG_lambda", type=float, default=0.02,
                        help="lambda parameter used to compute the elastic energy")                                                                     
    parser.add_argument("--EPG_mu",dest="EPG_mu", type=float, default=0.1,
                        help="mu parameter used to compute the elastic energy") 
    parser.add_argument("--EPG_trimmingradius",dest="EPG_trimmingradius", type=float, default=np.inf,
                        help="maximal distance of point from a node to affect its embedment") 
    parser.add_argument("--EPG_prob",dest="EPG_prob", type=float, default=1.0,
                        help="probability of including a single point for each computation") 
    parser.add_argument("--EPG_finalenergy",dest="EPG_finalenergy", default='Penalized',
                        help="Indicating the final elastic emergy associated with the configuration. It can be Base or Penalized ")
    parser.add_argument("--EPG_alpha",dest="EPG_alpha", type=float, default=0.02,
                        help="positive numeric, the value of the alpha parameter of the penalized elastic energy") 
    parser.add_argument("--EPG_beta",dest="EPG_beta", type=float, default=0.0,
                        help="positive numeric, the value of the beta parameter of the penalized elastic energy") 
    parser.add_argument("--EPG_collapse",dest="flag_EPG_collapse", action="store_true",
                        help="collapsing small branches")
    parser.add_argument("--EPG_collapse_mode",dest="EPG_collapse_mode", default ="PointNumber",
                        help="the mode used to collapse branches. PointNumber,PointNumber_Extrema, PointNumber_Leaves,EdgesNumber or EdgesLength")
    parser.add_argument("--EPG_collapse_par",dest="EPG_collapse_par", type=float, default=5,
                        help="positive numeric, the cotrol paramter used for collapsing small branches")
    parser.add_argument("--disable_EPG_optimize",dest="flag_disable_EPG_optimize", action="store_true",
                        help="disable optimizing branching")    
    parser.add_argument("--EPG_shift",dest="flag_EPG_shift", action="store_true",
                        help="shift branching point ")  
    parser.add_argument("--EPG_shift_mode",dest="EPG_shift_mode",default = 'NodeDensity',
                        help="the mode to use to shift the branching points NodePoints or NodeDensity")
    parser.add_argument("--EPG_shift_DR",dest="EPG_shift_DR",type=float, default=0.05,
                        help="positive numeric, the radius to be used when computing point density if EPG_shift_mode is NodeDensity")
    parser.add_argument("--EPG_shift_maxshift",dest="EPG_shift_maxshift", type=int, default=5,
                        help="positive integer, the maxium distance (as number of edges) to consider when exploring the branching point neighborhood")
    parser.add_argument("--disable_EPG_ext",dest="flag_disable_EPG_ext", action="store_true",
                        help="disable extending leaves with additional nodes")
    parser.add_argument("--EPG_ext_mode",dest="EPG_ext_mode",default = 'QuantDists',
                        help=" the mode used to extend the graph,QuantDists, QuantCentroid or WeigthedCentroid")
    parser.add_argument("--EPG_ext_par",dest="EPG_ext_par", type=float, default=0.5,
                        help="the control parameter used for contribution of the different data points when extending leaves with nodes")  
    parser.add_argument("--DE_zscore_cutoff",dest="DE_zscore_cutoff", default=2,
                        help="Differentially Expressed Genes z-score cutoff")
    parser.add_argument("--DE_logfc_cutoff",dest="DE_logfc_cutoff", default=0.25,
                        help="Differentially Expressed Genes log fold change cutoff")
    parser.add_argument("--TG_spearman_cutoff",dest="TG_spearman_cutoff", default=0.4,
                        help="Transition Genes Spearman correlation cutoff")
    parser.add_argument("--TG_logfc_cutoff",dest="TG_logfc_cutoff", default=0.25,
                        help="Transition Genes log fold change cutoff")
    parser.add_argument("--LG_zscore_cutoff",dest="LG_zscore_cutoff", default=1.5,
                        help="Leaf Genes z-score cutoff")
    parser.add_argument("--LG_pvalue_cutoff",dest="LG_pvalue_cutoff", default=1e-2,
                        help="Leaf Genes p value cutoff")
    parser.add_argument("--umap",dest="flag_umap", action="store_true",
                        help="whether to use UMAP for visualization (default: No)") 
    parser.add_argument("-r",dest="root" ,default=None,
                        help="root node for subwaymap_plot and stream_plot")    
    parser.add_argument("--stream_log_view",dest="flag_stream_log_view", action="store_true",
                        help="use log2 scale for y axis of stream_plot")
    parser.add_argument("-o","--output_folder",dest="output_folder", default=None,
                        help="Output folder")
    parser.add_argument("--for_web",dest="flag_web", action="store_true",
                        help="Output files for website")


    args = parser.parse_args()
    if args.input_filename is None and args.new_filename is None and args.atac_counts is None and args.atac_zscore is None:
       parser.error("at least one of -m,--atac_counts,--atac_zscore,or --new required") 

    new_filename = args.new_filename
    new_label_filename = args.new_label_filename
    new_label_color_filename = args.new_label_color_filename
    flag_stream_log_view = args.flag_stream_log_view
    flag_gene_TG_detection = args.flag_gene_TG_detection
    flag_gene_DE_detection = args.flag_gene_DE_detection
    flag_gene_LG_detection = args.flag_gene_LG_detection
    flag_web = args.flag_web
    flag_first_PC = args.flag_first_PC
    flag_umap = args.flag_umap
    genes = args.genes
    DE_zscore_cutoff = args.DE_zscore_cutoff
    DE_logfc_cutoff = args.DE_logfc_cutoff
    TG_spearman_cutoff = args.TG_spearman_cutoff
    TG_logfc_cutoff = args.TG_logfc_cutoff
    LG_zscore_cutoff = args.LG_zscore_cutoff
    LG_pvalue_cutoff = args.LG_pvalue_cutoff
    root = args.root

    input_filename = args.input_filename
    cell_label_filename = args.cell_label_filename
    cell_label_color_filename = args.cell_label_color_filename
    s_method = args.s_method
    use_precomputed = args.use_precomputed
    n_processes = args.n_processes
    loess_frac = args.loess_frac
    loess_cutoff = args.loess_cutoff
    pca_n_PC = args.pca_n_PC
    flag_log2 = args.flag_log2
    flag_norm = args.flag_norm
    flag_atac = args.flag_atac
    atac_regions = args.atac_regions
    atac_counts = args.atac_counts
    atac_samples = args.atac_samples
    atac_k = args.atac_k
    atac_zscore = args.atac_zscore
    lle_n_nb_percent = args.lle_n_nb_percent #LLE neighbour percent
    lle_n_component = args.lle_n_component #LLE dimension reduction
    AP_damping_factor = args.AP_damping_factor
    EPG_n_nodes = args.EPG_n_nodes 
    EPG_n_rep = args.EPG_n_rep
    EPG_lambda = args.EPG_lambda
    EPG_mu = args.EPG_mu
    EPG_trimmingradius = args.EPG_trimmingradius
    EPG_prob = args.EPG_prob
    EPG_finalenergy = args.EPG_finalenergy
    EPG_alpha = args.EPG_alpha
    EPG_beta = args.EPG_beta
    flag_EPG_collapse = args.flag_EPG_collapse
    EPG_collapse_mode = args.EPG_collapse_mode
    EPG_collapse_par = args.EPG_collapse_par
    flag_EPG_shift = args.flag_EPG_shift
    EPG_shift_mode = args.EPG_shift_mode
    EPG_shift_DR = args.EPG_shift_DR
    EPG_shift_maxshift = args.EPG_shift_maxshift
    flag_disable_EPG_optimize = args.flag_disable_EPG_optimize
    flag_disable_EPG_ext = args.flag_disable_EPG_ext
    EPG_ext_mode = args.EPG_ext_mode
    EPG_ext_par = args.EPG_ext_par
    output_folder = args.output_folder #work directory
   
    if(flag_web):
        flag_savefig = False
    else:
        flag_savefig = True
    gene_list = []
    if(genes!=None):
        if(os.path.exists(genes)):
            gene_list = pd.read_csv(genes,sep='\t',header=None,index_col=None,compression= 'gzip' if genes.split('.')[-1]=='gz' else None).iloc[:,0].tolist()
            gene_list = list(set(gene_list))
        else:
            gene_list = genes.split(',')
        print('Genes to visualize: ')
        print(gene_list)
    if(new_filename is None):        
        if(output_folder==None):
            workdir = os.path.join(os.getcwd(),'stream_result')
        else:
            workdir = output_folder
        if(use_precomputed):
            print('Importing the precomputed pkl file...')
            adata = st.read(file_name='stream_result.pkl',file_format='pkl',file_path=workdir)
        else:
            if(flag_atac):
                if(atac_zscore is None):
                    print('Reading in atac seq data...')
                    adata = st.read(atac_counts,file_name_sample=atac_samples,file_name_region=atac_regions,experiment='atac-seq',workdir=workdir)
                    adata = st.counts_to_kmers(adata,k=atac_k,n_jobs = n_processes)
                else:
                    print('Reading in atac zscore file...')
                    adata = st.read(file_name=atac_zscore,file_format='pkl',workdir=workdir)
            else:
                adata=st.read(file_name=input_filename,workdir=workdir)
                print('Input: '+ str(adata.obs.shape[0]) + ' cells, ' + str(adata.var.shape[0]) + ' genes')
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
            if(cell_label_filename !=None):
                st.add_cell_labels(adata,file_name=cell_label_filename)
            else:
                st.add_cell_labels(adata)
            if(cell_label_color_filename !=None):    
                st.add_cell_colors(adata,file_name=cell_label_color_filename)
            else:
                st.add_cell_colors(adata)
            if(flag_atac):
                print('Selecting top principal components...')
                st.select_top_principal_components(adata,n_pc = pca_n_PC,first_pc = flag_first_PC,save_fig=True)
                st.dimension_reduction(adata,n_components=lle_n_component,nb_pct=lle_n_nb_percent,n_jobs=n_processes,feature='top_pcs')                
            else:
                if(flag_norm):
                    st.normalize_per_cell(adata)
                if(flag_log2):
                    st.log_transform(adata)
                if(s_method!='all'):
                    print('Filtering genes...')
                    st.filter_genes(adata,min_num_cells=5)
                    print('Removing mitochondrial genes...')
                    st.remove_mt_genes(adata)        
                    if(s_method == 'LOESS'):
                        print('Selecting most variable genes...')
                        st.select_variable_genes(adata,loess_frac=loess_frac,percentile=loess_cutoff,save_fig=True)
                        pd.DataFrame(adata.uns['var_genes']).to_csv(os.path.join(workdir,'selected_variable_genes.tsv'),sep = '\t',index = None,header=False)
                        st.dimension_reduction(adata,n_components=lle_n_component,nb_pct=lle_n_nb_percent,n_jobs=n_processes,feature='var_genes')
                    if(s_method == 'PCA'):
                        print('Selecting top principal components...')
                        st.select_top_principal_components(adata,n_pc = pca_n_PC,first_pc = flag_first_PC,save_fig=True)
                        st.dimension_reduction(adata,n_components=lle_n_component,nb_pct=lle_n_nb_percent,n_jobs=n_processes,feature='top_pcs')
                else:
                    print('Keep all the genes...')
                    st.dimension_reduction(adata,n_components=lle_n_component,nb_pct=lle_n_nb_percent,n_jobs=n_processes,feature='all')
            st.plot_dimension_reduction(adata,save_fig=flag_savefig)
            st.seed_elastic_principal_graph(adata,damping=AP_damping_factor)
            st.plot_branches(adata,save_fig=flag_savefig,fig_name='seed_elastic_principal_graph_skeleton.pdf')
            st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='seed_elastic_principal_graph.pdf')

            st.elastic_principal_graph(adata,epg_n_nodes = EPG_n_nodes,epg_lambda=EPG_lambda,epg_mu=EPG_mu,epg_trimmingradius=EPG_trimmingradius,epg_alpha=EPG_alpha)
            st.plot_branches(adata,save_fig=flag_savefig,fig_name='elastic_principal_graph_skeleton.pdf')
            st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='elastic_principal_graph.pdf')            
            if(not flag_disable_EPG_optimize):
                st.optimize_branching(adata)
                st.plot_branches(adata,save_fig=flag_savefig,fig_name='optimizing_elastic_principal_graph_skeleton.pdf')
                st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='optimizing_elastic_principal_graph.pdf')   
            if(flag_EPG_shift):
                st.shift_branching(adata,epg_shift_mode = EPG_shift_mode,epg_shift_radius = EPG_shift_DR,epg_shift_max=EPG_shift_maxshift)
                st.plot_branches(adata,save_fig=flag_savefig,fig_name='shifting_elastic_principal_graph_skeleton.pdf')
                st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='shifting_elastic_principal_graph.pdf')
            if(flag_EPG_collapse):
                st.prune_elastic_principal_graph(adata,epg_collapse_mode=EPG_collapse_mode, epg_collapse_par = EPG_collapse_par)
                st.plot_branches(adata,save_fig=flag_savefig,fig_name='pruning_elastic_principal_graph_skeleton.pdf')
                st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='pruning_elastic_principal_graph.pdf')
            if(not flag_disable_EPG_ext):
                st.extend_elastic_principal_graph(adata,epg_ext_mode = EPG_ext_mode, epg_ext_par = EPG_ext_par)
                st.plot_branches(adata,save_fig=flag_savefig,fig_name='extending_elastic_principal_graph_skeleton.pdf')
                st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='extending_elastic_principal_graph.pdf')
            st.plot_branches(adata,save_fig=flag_savefig,fig_name='finalized_elastic_principal_graph_skeleton.pdf')
            st.plot_branches_with_cells(adata,save_fig=flag_savefig,fig_name='finalized_elastic_principal_graph.pdf')
            st.plot_flat_tree(adata,save_fig=flag_savefig)
            if(flag_umap):
                print('UMAP visualization based on top MLLE components...')
                st.plot_visualization_2D(adata,save_fig=flag_savefig,fig_name='umap_cells')
                st.plot_visualization_2D(adata,color_by='branch',save_fig=flag_savefig,fig_name='umap_branches')
            if(root is None):
                print('Visualization of subwaymap and stream plots...')
                flat_tree = adata.uns['flat_tree']
                list_node_start = [value for key,value in nx.get_node_attributes(flat_tree,'label').items()]               
                for ns in list_node_start:
                    if(flag_web):
                        st.subwaymap_plot(adata,percentile_dist=100,root=ns,save_fig=flag_savefig)
                        st.stream_plot(adata,root=ns,fig_size=(8,8),save_fig=True,flag_log_view=flag_stream_log_view,fig_name='stream_plot.png')                        
                    else:
                        st.subwaymap_plot(adata,percentile_dist=100,root=ns,save_fig=flag_savefig)
                        st.stream_plot(adata,root=ns,fig_size=(8,8),save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
            else:
                st.subwaymap_plot(adata,percentile_dist=100,root=root,save_fig=flag_savefig)
                st.stream_plot(adata,root=root,fig_size=(8,8),save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
            if(flag_gene_TG_detection):
                print('Identifying transition genes...')
                st.detect_transistion_genes(adata,cutoff_spearman=TG_spearman_cutoff,cutoff_logfc = TG_logfc_cutoff,n_jobs = n_processes)
                st.plot_transition_genes(adata,save_fig=flag_savefig)
            if(flag_gene_DE_detection):
                print('Identifying differentially expressed genes...')
                st.detect_de_genes(adata,cutoff_zscore=DE_logfc_cutoff,cutoff_logfc = DE_logfc_cutoff,n_jobs = n_processes)
                st.plot_de_genes(adata,save_fig=flag_savefig)
            if(flag_gene_LG_detection):
                print('Identifying leaf genes...')
                st.detect_leaf_genes(adata,cutoff_zscore=LG_zscore_cutoff,cutoff_pvalue=LG_pvalue_cutoff,n_jobs = n_processes)
            output_cell_info(adata)
            if(flag_web):
                output_for_website(adata)
            st.write(adata)

        if((genes!=None) and (len(gene_list)>0)):
            print('Visualizing genes...')
            flat_tree = adata.uns['flat_tree']
            list_node_start = [value for key,value in nx.get_node_attributes(flat_tree,'label').items()]
            if(root is None):
                for ns in list_node_start:
                    if(flag_web):
                        output_for_website_subwaymap_gene(adata,gene_list)
                        st.stream_plot_gene(adata,root=ns,fig_size=(8,8),genes=gene_list,save_fig=True,flag_log_view=flag_stream_log_view,fig_format='png')
                    else:
                        st.subwaymap_plot_gene(adata,percentile_dist=100,root=ns,genes=gene_list,save_fig=flag_savefig)
                        st.stream_plot_gene(adata,root=ns,fig_size=(8,8),genes=gene_list,save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
            else:
                if(flag_web):
                    output_for_website_subwaymap_gene(adata,gene_list)
                    st.stream_plot_gene(adata,root=root,fig_size=(8,8),genes=gene_list,save_fig=True,flag_log_view=flag_stream_log_view,fig_format='png')
                else:
                    st.subwaymap_plot_gene(adata,percentile_dist=100,root=root,genes=gene_list,save_fig=flag_savefig)
                    st.stream_plot_gene(adata,root=root,fig_size=(8,8),genes=gene_list,save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
            
    else:
        print('Starting mapping procedure...')
        if(output_folder==None):
            workdir_ref = os.path.join(os.getcwd(),'stream_result')
        else:
            workdir_ref = output_folder
        adata = st.read(file_name='stream_result.pkl',file_format='pkl',file_path=workdir_ref)
        workdir = os.path.join(workdir_ref,os.pardir,'mapping_result')
        adata_new=st.read(file_name=new_filename,workdir=workdir)
        st.add_cell_labels(adata_new,file_name=new_label_filename)
        st.add_cell_colors(adata_new,file_name=new_label_color_filename)    
        if(s_method == 'LOESS'):
            st.map_new_data(adata,adata_new,feature='var_genes')
        if(s_method == 'all'):
            st.map_new_data(adata,adata_new,feature='all')
        if(flag_umap):
            st.plot_visualization_2D(adata,adata_new=adata_new,use_precomputed=False,save_fig=flag_savefig,fig_name='umap_new_cells')
            st.plot_visualization_2D(adata,adata_new=adata_new,show_all_colors=True,save_fig=flag_savefig,fig_name='umap_all_cells')
            st.plot_visualization_2D(adata,adata_new=adata_new,color_by='branch',save_fig=flag_savefig,fig_name='umap_branches')
        if(root is None):
            flat_tree = adata.uns['flat_tree']
            list_node_start = [value for key,value in nx.get_node_attributes(flat_tree,'label').items()]
            for ns in list_node_start:
                st.subwaymap_plot(adata,adata_new=adata_new,percentile_dist=100,show_all_cells=False,root=ns,save_fig=flag_savefig)
                st.stream_plot(adata,adata_new=adata_new,show_all_colors=False,root=ns,fig_size=(8,8),save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
        else:
            st.subwaymap_plot(adata,adata_new=adata_new,percentile_dist=100,show_all_cells=False,root=root,save_fig=flag_savefig)
            st.stream_plot(adata,adata_new=adata_new,show_all_colors=False,root=root,fig_size=(8,8),save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
        if((genes!=None) and (len(gene_list)>0)):
            if(root is None):
                for ns in list_node_start:
                    st.subwaymap_plot_gene(adata,adata_new=adata_new,percentile_dist=100,root=ns,save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
            else:
                st.subwaymap_plot_gene(adata,adata_new=adata_new,percentile_dist=100,root=root,save_fig=flag_savefig,flag_log_view=flag_stream_log_view)
        st.write(adata_new,file_name='stream_mapping_result.pkl')
    print('Finished computation...')

if __name__ == "__main__":
    main()