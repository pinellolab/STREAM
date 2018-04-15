#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__= '0.1.0'
__tool_name__='STREAM'


print'''
   _____ _______ _____  ______          __  __ 
  / ____|__   __|  __ \|  ____|   /\   |  \/  |
 | (___    | |  | |__) | |__     /  \  | \  / |
  \___ \   | |  |  _  /|  __|   / /\ \ | |\/| |
  ____) |  | |  | | \ \| |____ / ____ \| |  | |
 |_____/   |_|  |_|  \_\______/_/    \_\_|  |_|
                                               
'''
print '- Single-cell Trajectory Reconstruction and Mapping -'
print'\n[Luca Pinello & Huidong Chen 2018, send bugs, suggestions or comments to lucapinello AT gmail DOT com]\n\n',

print 'Version %s\n' % __version__

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import os
import itertools
import multiprocessing
import argparse
import math
import string
import cPickle
#import pickle
# import msgpack
#import dill
import unicodedata
import re

import matplotlib as mpl
mpl.use('Agg')
mpl.rc('pdf', fonttype=42)
import pylab as plt
from pylab import *
import seaborn as sns


from sklearn.cluster import SpectralClustering,AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances,pairwise_distances_argmin_min
from sklearn.manifold import locally_linear_embedding,MDS,LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from scipy import stats,interpolate
from scipy.stats import spearmanr,mannwhitneyu,gaussian_kde
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline,UnivariateSpline
from scipy.signal import savgol_filter
from scipy.spatial import distance,cKDTree,KDTree
from scipy.linalg import eigh, svd, qr, solve
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels import robust
from copy import deepcopy
from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.path import Path
import matplotlib.patches as Patches
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from decimal import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely.geometry as geom
import igraph

# from ZIFA import ZIFA
# from ZIFA import block_ZIFA

import logging
logging.basicConfig(level=logging.INFO,
                     format='%(levelname)-5s @ %(asctime)s:\n\t %(message)s \n',
                     datefmt='%a, %d %b %Y %H:%M:%S',
                     stream=sys.stderr,
                     filemode="w"
                     )

error   = logging.critical
warn    = logging.warning
debug   = logging.debug
info    = logging.info



os.environ["QT_QPA_PLATFORM"] = "offscreen"


def Read_In_Data(input_filename,cell_label_filename,cell_label_color_filename,flag_log2,flag_norm):
    input_data = pd.read_csv(input_filename,sep='\t',header=0,index_col=0,compression= 'gzip' if input_filename.split('.')[-1]=='gz' else None)
    # input_data = pd.read_csv(input_filename,sep='\t',header=0,index_col=0)
    print('Input: '+ str(input_data.shape[1]) + ' cells, ' + str(input_data.shape[0]) + ' genes')
    if(flag_norm):
        ### remove libary size factor
        print('Noralizing data...')
        input_data = input_data.div(input_data.sum(axis=0),axis=1)*1e6
    if(flag_log2):
        ### log2 transformation
        print('Log2 transformation...')
        input_data = np.log2(input_data+1)
    input_genes = input_data.index.astype('str').tolist()
    ### remove mitochondrial genes
    r = re.compile("^MT-",flags=re.IGNORECASE)
    mt_genes = filter(r.match, input_genes)
    if(len(mt_genes)>0):
        print('remove mitochondrial genes:')
        print(mt_genes)
        input_data.drop(mt_genes,axis=0,inplace=True)
        input_genes = input_data.index.tolist()
    if(len(input_genes)!=len(set(input_genes))):
        uni_genes,uni_counts = np.unique(input_genes,return_counts=True)
        dup_genes = uni_genes[np.where(uni_counts>1)]
        print('There exist duplicated genes: ' + str(dup_genes))
        print('Average expression for duplicated genes is calculated')
        ### average duplicate genes
        input_data = input_data.groupby(input_data.index).mean()
        input_data.shape
        input_genes = input_data.index.tolist()
    input_cell_id = input_data.columns.astype('str').tolist()
    if(cell_label_filename != None):
        input_cell_label = pd.read_csv(cell_label_filename,sep='\t',header=None,index_col=None,compression= 'gzip' if cell_label_filename.split('.')[-1]=='gz' else None)
        input_cell_label = input_cell_label.iloc[:,0].astype('str').tolist()
        input_cell_label_uni = np.unique(input_cell_label).tolist()
        if(cell_label_color_filename != None):
            # df_label_color = pd.read_csv(cell_label_color_filename,sep='\t',header=None,index_col=0)
            df_label_color = pd.read_csv(cell_label_color_filename,sep='\t',header=None,dtype={0:np.str},compression= 'gzip' if cell_label_color_filename.split('.')[-1]=='gz' else None)
            df_label_color = df_label_color.set_index(0)            
            input_cell_label_uni_color = df_label_color.to_dict()[1]
        else:
            input_cm = sns.color_palette("hls",n_colors=len(input_cell_label_uni)).as_hex()
            input_cell_label_uni_color = {x:input_cm[i] for i,x in enumerate(input_cell_label_uni)}
    else:
        input_cell_label = np.repeat('unknown',len(input_cell_id)).tolist()
        input_cell_label_uni = np.unique(input_cell_label).tolist()
        input_cell_label_uni_color = {x:'gray' for i,x in enumerate(input_cell_label_uni)}

    df_flat_tree = pd.DataFrame(index = range(len(input_cell_id)),columns=['CELL_ID','CELL_LABEL'])
    df_flat_tree['CELL_ID'] = input_cell_id
    df_flat_tree['CELL_LABEL'] = input_cell_label
    # df_flat_tree['is_outlier'] = 'No'

    df_sc = input_data.copy()
    df_sc = df_sc.T
    df_sc.insert(0,'CELL_LABEL',input_cell_label)
    df_sc.index = range(df_sc.shape[0])
    return df_flat_tree,df_sc,input_genes,input_cell_label_uni,input_cell_label_uni_color

def Read_In_New_Data(input_filename,cell_label_filename,cell_label_color_filename,flag_log2,flag_norm):
    input_data = pd.read_csv(input_filename,sep='\t',header=0,index_col=0,compression= 'gzip' if input_filename.split('.')[-1]=='gz' else None)
    # input_data = pd.read_csv(input_filename,sep='\t',header=0,index_col=0)
    print('New Input: '+ str(input_data.shape[1]) + ' cells, ' + str(input_data.shape[0]) + ' genes')
    if(flag_norm):
        ### remove libary size factor
        print('Noralizing data...')
        input_data = input_data.div(input_data.sum(axis=0),axis=1)*1e6
    if(flag_log2):
        ### log2 transformation
        print('Log2 transformation...')
        input_data = np.log2(input_data+1)
    input_genes = input_data.index.astype('str').tolist()
    if(len(input_genes)!=len(set(input_genes))):
        uni_genes,uni_counts = np.unique(input_genes,return_counts=True)
        dup_genes = uni_genes[np.where(uni_counts>1)]
        print('There exist duplicated genes: ' + str(dup_genes))
        print('Average expression for duplicated genes is calculated')
        ### average duplicate genes
        input_data = input_data.groupby(input_data.index).mean()
        print(str(input_data.shape[0]) + ' genes are kept after removing duplicate genes')
        input_genes = input_data.index.tolist()
    input_cell_id = input_data.columns.astype('str').tolist()
    if(cell_label_filename != None):
        input_cell_label = pd.read_csv(cell_label_filename,sep='\t',header=None,index_col=None,compression= 'gzip' if cell_label_filename.split('.')[-1]=='gz' else None)
        input_cell_label = input_cell_label.iloc[:,0].astype('str').tolist()
        input_cell_label_uni = np.unique(input_cell_label).tolist()
        if(cell_label_color_filename != None):
            df_label_color = pd.read_csv(cell_label_color_filename,sep='\t',header=None,index_col=0,compression= 'gzip' if cell_label_color_filename.split('.')[-1]=='gz' else None)
            input_cell_label_uni_color = df_label_color.to_dict()[1]
        else:
            input_cm = plt.get_cmap('jet',len(input_cell_label_uni))
            input_cell_label_uni_color = {x:input_cm(i) for i,x in enumerate(input_cell_label_uni)}
    else:
        input_cell_label = np.repeat('unknown',len(input_cell_id)).tolist()
        input_cell_label_uni = np.unique(input_cell_label).tolist()
        input_cell_label_uni_color = {x:'gray' for i,x in enumerate(input_cell_label_uni)}

    df_flat_tree = pd.DataFrame(index = range(len(input_cell_id)),columns=['CELL_ID','CELL_LABEL'])

    df_flat_tree['CELL_ID'] = input_cell_id
    df_flat_tree['CELL_LABEL'] = input_cell_label

    df_sc = input_data.copy()
    df_sc = df_sc.T
    df_sc.insert(0,'CELL_LABEL',input_cell_label)
    df_sc.index = range(df_sc.shape[0])
    return df_flat_tree,df_sc,input_cell_label_uni,input_cell_label_uni_color


def Filter_Genes(df_sc):
    df_sc_final = df_sc.copy()
    min_num_cells = max(5,int(round(df_sc_final.shape[0]*0.001))) # minimum number of cells in which genes are expressed
    genes_filtered = df_sc_final.iloc[:,1:].columns[(df_sc_final.iloc[:,1:]>1).sum() > min_num_cells].tolist()
    print('After filtering out low-expressed genes: ')
    print(str(df_sc_final.shape[0])+' cells, ' + str(len(genes_filtered))+' genes')
    return genes_filtered

def project_point_to_curve_distance(params):
    XP = params[0]
    p = params[1]
    curve = geom.LineString(XP)
    point = geom.Point(p)
    #distance from point to curve
    dist_p_to_c = point.distance(curve)
    return dist_p_to_c

def project_point_to_line_distance(params):
    #distance from point P to line(A,B)
    A = np.array(params[0],dtype=float)
    B = np.array(params[1],dtype=float)
    P = np.array(params[2],dtype=float)
    PA = P-A
    BA = B-A
    T = np.dot(PA,BA)/np.dot(BA,BA)
    dist_p_to_l = norm(PA-T*BA)
    return dist_p_to_l

def project_point_to_line(params):
    #projecting point P to line(A,B)
    A = np.array(params[0],dtype=float)
    B = np.array(params[1],dtype=float)
    P = np.array(params[2],dtype=float)
    PA = P-A
    BA = B-A
    #ratio of scalar projection to magnitude of BA
    T = np.dot(PA,BA)/np.dot(BA,BA)
    #projection point
    P_proj = A + T*BA
    #distance from P to its projection point
    dist_p_to_l = norm(PA-T*BA)
    return T,P_proj,dist_p_to_l

def project_point_to_line_segment_matrix_distance(params):
    XP = params[0]
    p = params[1]
    XP = np.array(XP,dtype=float)
    p = np.array(p,dtype=float)
    list_dist = list()
    for id_XP in range(XP.shape[0]-1):
        T,P_proj,dist_p_to_l = project_point_to_line((XP[id_XP],XP[id_XP+1],p))
        if((T>=0)&(T<=1)):
            list_dist.append(dist_p_to_l)
            break
        elif(T<0):
            list_dist.append(norm(p-XP[id_XP]))
        elif(T>1):
            list_dist.append(norm(p-XP[id_XP+1]))
    return(min(list_dist))

def project_point_to_line_segment_matrix(XP,p):
    XP = np.array(XP,dtype=float)
    p = np.array(p,dtype=float)
    AA=XP[:-1,:]
    BB=XP[1:,:]
    AB = (BB-AA)
    AB_squared = (AB*AB).sum(1)
    Ap = (p-AA)
    t = (Ap*AB).sum(1)/AB_squared
    t[AB_squared == 0]=0
    Q = AA + AB*np.tile(t,(XP.shape[1],1)).T
    Q[t<=0,:]=AA[t<=0,:]
    Q[t>=1,:]=BB[t>=1,:]
    kdtree=cKDTree(Q)
    d,idx_q=kdtree.query(p)
    dist_p_to_q = np.sqrt(np.inner(p-Q[idx_q,:],p-Q[idx_q,:]))
    XP_p = np.row_stack((XP[:idx_q+1],Q[idx_q,:]))
    lam = np.sum(np.sqrt(np.square((XP_p[1:,:] - XP_p[:-1,:])).sum(1)))
    return list([Q[idx_q,:],idx_q,dist_p_to_q,lam])

def Select_Variable_Genes(df_sc,loess_frac,loess_z_score_cutoff,n_processes,file_path,flag_web):
    df_sc_final = df_sc.copy()
    mean_genes = np.mean(df_sc_final.iloc[:,1:],axis=0)
    std_genes = np.std(df_sc_final.iloc[:,1:],ddof=1,axis=0)
    loess_fitted = lowess(std_genes,mean_genes,return_sorted=False,frac=loess_frac)
    residuals = std_genes - loess_fitted
    XP = np.column_stack((sort(mean_genes),loess_fitted[np.argsort(mean_genes)]))
    mat_p = np.column_stack((mean_genes,std_genes))
    params = [(XP,mat_p[i,]) for i in range(XP.shape[0])]
    pool = multiprocessing.Pool(processes=n_processes)
    dist_point_to_curve = pool.map(project_point_to_curve_distance,params)
    pool.close()
    mat_sign = np.ones(XP.shape[0])
    mat_sign[np.where(residuals<0)[0]] = -1
    dist_point_to_curve = np.array(dist_point_to_curve)*mat_sign
    zscore_dist = stats.zscore(dist_point_to_curve)
    cutoff = loess_z_score_cutoff
    residuals.index.name = None
    df_dist = pd.DataFrame(data = np.column_stack((dist_point_to_curve,zscore_dist)),
                       index=residuals.index,
                       columns=['dist','zscore'])
    df_dist.sort_values(by='dist',ascending = False,inplace=True)
    df_dist_positive = df_dist.iloc[np.where(df_dist.dist>0)[0],:].copy()
    # df_dist = df_dist[(df_dist.dist>0)&(df_dist.zscore>cutoff)]
    # cutoff_dist = np.percentile(df_dist_positive.dist,90)
    cutoff_dist = np.percentile(df_dist.dist,95)
    df_dist = df_dist[(df_dist.dist>cutoff_dist)]
    feature_genes = list(df_dist.index)
    # if(df_dist.shape[0]>500):
    #     feature_genes = feature_genes[:500]
        # cutoff_dist = np.percentile(df_dist_positive.dist,95)
        # df_dist = df_dist[(df_dist.dist>cutoff_dist)]
        # feature_genes = list(df_dist.index)

    # cutoff = robust.mad(residuals[residuals>0])*3.5
    # dist_point_to_curve = pd.Series(dist_point_to_curve,index=residuals.index)
    # cutoff = np.percentile(dist_point_to_curve,95)
    # feature_genes = list(dist_point_to_curve[dist_point_to_curve>cutoff].index)
    # residuals.sort_values(ascending=False,inplace=True)
    # feature_genes = list(residuals[residuals>cutoff].index)
    # if(len(feature_genes)>600):
    #     feature_genes = feature_genes[:600]
    pandas2ri.activate()
    if(flag_web):
        R.png(file_path + '/st_vs_means.png')
    else:
        R.pdf(file_path + '/st_vs_means.pdf')
    R.smoothScatter(mean_genes,std_genes,xlab='means',ylab='std',nrpoints=0,pch=16,asp=1)
    R.lines(sort(mean_genes),loess_fitted[np.argsort(mean_genes)],col='red',lty=4,lwd=5)
    R.points(mean_genes[feature_genes],std_genes[feature_genes],pch=20,col='red')
    R('dev.off()')
    pd.DataFrame(feature_genes).to_csv(file_path+'/output_selected_genes.tsv',sep = '\t',index = None,header=False)
    print(str(len(feature_genes)) + ' variable genes are selected')
    return feature_genes

def Select_Principal_Components(df_sc,pca_max_PC,pca_n_PC,flag_first_PC,file_path,file_path_precomp,flag_web):
    df_sc_final = df_sc.copy()
    sklearn_pca = sklearnPCA(n_components = pca_max_PC,svd_solver='full')
    sklearn_pca = sklearn_pca.fit(df_sc_final.iloc[:,1:].values)
    Save_To_Pickle(sklearn_pca,'sklearn_pca',file_path_precomp)
    sklearn_transf = sklearn_pca.transform(df_sc_final.iloc[:,1:].values)
    if(pca_n_PC == -1):
        scaled_variance_ratio = sklearn_pca.explained_variance_ratio_/sum(sklearn_pca.explained_variance_ratio_)
        pca_n_PC = np.where(np.cumsum(scaled_variance_ratio)>0.75)[0][0]
    Save_To_Pickle(pca_n_PC,'pca_n_PC',file_path_precomp)
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(pca_max_PC),sklearn_pca.explained_variance_ratio_)
    if(flag_first_PC):
        plt.axvline(pca_n_PC,c='red',ls = '--')
    else:
        plt.axvline(1,c='red',ls = '--')
        plt.axvline(pca_n_PC+1,c='red',ls = '--')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    if(flag_web):
        plt.savefig(file_path +'/variance_vs_PC.png')
    else:
        plt.savefig(file_path +'/variance_vs_PC.pdf')
    plt.close(fig)
    if(flag_first_PC):
        df_sklearn_transf = pd.DataFrame(sklearn_transf[:,0:(pca_n_PC)],columns=['PC' + str(x) for x in range(pca_n_PC)])
    else:
        #discard the first Principal Component
        df_sklearn_transf = pd.DataFrame(sklearn_transf[:,1:(pca_n_PC+1)],columns=['PC' + str(x+1) for x in range(pca_n_PC)])
    df_sklearn_transf.insert(0,'CELL_LABEL',df_sc_final['CELL_LABEL'])
    df_sc_final = df_sklearn_transf
    print(str(df_sc_final.shape[1]-1) + ' PCs are selected')
    return df_sc_final


def Dimension_Reduction(df_sc_final,lle_n_component,lle_n_nb_percent,file_path,file_path_precomp,n_processes):
    DR_input_values = df_sc_final.iloc[:,1:].values
    lle_n_neighbour = int(around(df_sc_final.shape[0] * lle_n_nb_percent))
    np.random.seed(2)
    # X, err = locally_linear_embedding(DR_input_values, n_neighbors=lle_n_neighbour, n_components=lle_n_component,
    #                                     method = 'modified',eigen_solver = 'dense',n_jobs = n_processes)
    sklearn_lle = LocallyLinearEmbedding(n_neighbors=lle_n_neighbour, n_components=lle_n_component,
                                        method = 'modified',eigen_solver = 'dense',n_jobs = n_processes,random_state=10,
                                        neighbors_algorithm = 'kd_tree')
    sklearn_lle = sklearn_lle.fit(DR_input_values)
    Save_To_Pickle(sklearn_lle,'sklearn_lle',file_path_precomp)
    # X = sklearn_lle.transform(DR_input_values)
    X = sklearn_lle.embedding_
    return X

# def ZIFA_Dimension_Reduction(df_sc_final,lle_n_component,lle_n_nb_percent):
#     DR_input_values = df_sc_final.iloc[:,1:].values
#     X, model_params = block_ZIFA.fitModel(DR_input_values,lle_n_component)
#     return X


def Plot_Dimension_Reduction(df_sc_final,X,input_cell_label_uni,input_cell_label_uni_color,file_path):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    df_color = pd.DataFrame(columns=['cell_label','color'])
    df_color['cell_label'] = df_sc_final['CELL_LABEL']   
    list_patches = []
    for x in input_cell_label_uni_color.keys():
        id_cells = np.where(df_sc_final['CELL_LABEL']==x)[0]
        df_color.loc[df_color.index[id_cells],'color'] = input_cell_label_uni_color[x]
        list_patches.append(Patches.Patch(color = input_cell_label_uni_color[x],label=x))
        
    X_plot = pd.DataFrame(X).sample(frac=1,random_state=100)
    X_color = df_color.sample(frac=1,random_state=100)['color']
    ax.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1],X_plot.iloc[:, 2], c=X_color,s=50,linewidth=0,alpha=0.8)  

    max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
    mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
    mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
    mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('LLE Component1',labelpad=20)
    ax.set_ylabel('LLE Component2',labelpad=20)
    ax.set_zlabel('LLE Component3',labelpad=20)
    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
              ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)
    plt.savefig(file_path +'/LLE.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)


def dfs_from_leaf(MS_copy,node,degrees_of_nodes,nodes_to_visit,nodes_to_merge):
    nodes_to_visit.remove(node)
    for n2 in MS_copy.neighbors(node):

        if n2 in nodes_to_visit:

            if degrees_of_nodes[n2]==2:  #grow the branch

                if n2 not in nodes_to_merge:
                    nodes_to_merge.append(n2)

                dfs_from_leaf(MS_copy,n2,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            else:
                nodes_to_merge.append(n2)
                return


def Plot_MST(MS,XC,X,df_flat_tree,input_cell_label_uni,input_cell_label_uni_color,file_path,file_name='MST'):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # legend_labels=[]
    # for idx,celllabel in enumerate(input_cell_label_uni):
    #     indices_to_plot = np.where(df_flat_tree.CELL_LABEL == celllabel)[0]
    #     ax.scatter(X[indices_to_plot, 0], X[indices_to_plot, 1],  X[indices_to_plot, 2], \
    #                c=input_cell_label_uni_color[celllabel],s=50,linewidth=0,alpha=0.6,zorder=None)
    #     legend_labels.append(celllabel)
    # ax.legend(legend_labels,loc='center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)

    for n in MS.nodes():
        ax.scatter(XC[n][0],XC[n][1],XC[n][2],color='red',s=80,marker='o',alpha=0.9,zorder=100)
        # ax.text(XC[n][0],XC[n][1],XC[n][2],n,color='black',fontsize = 12)
    for id_edge,MS_edge in enumerate(MS.edges_iter()):
        x_pos = (XC[MS_edge[0]][0],XC[MS_edge[1]][0])
        y_pos = (XC[MS_edge[0]][1],XC[MS_edge[1]][1])
        z_pos = (XC[MS_edge[0]][2],XC[MS_edge[1]][2])
        ax.plot(x_pos,y_pos,z_pos,'b',lw=2,zorder=10)

    max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
    mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
    mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
    mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('LLE Component1',labelpad=20)
    ax.set_ylabel('LLE Component2',labelpad=20)
    ax.set_zlabel('LLE Component3',labelpad=20)
    plt.savefig(file_path + '/'+str(file_name)+'.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)

def Cal_Branch_Length(EPG,dict_branches):
    dict_nodes_pos = nx.get_node_attributes(EPG,'pos')
    if(dict_nodes_pos != {}):
        for br_key,br_value in dict_branches.iteritems():
            nodes = br_value['nodes']
            br_nodes_pos = np.array([dict_nodes_pos[i] for i in nodes]) 
            dict_branches[br_key]['len'] = sum(np.sqrt(((br_nodes_pos[0:-1,:] - br_nodes_pos[1:,:])**2).sum(1)))
    return dict_branches

def Extract_Branches(EPG):
    #record the original degree(before removing nodes) for each node
    EPG_copy = EPG.copy()
    degrees_of_nodes = EPG_copy.degree()
    dict_branches = dict()
    clusters_to_merge=[]
    while EPG_copy.order()>1: #the number of vertices
        leaves=[n for n,d in EPG_copy.degree().items() if d==1]
        nodes_included=EPG_copy.nodes()
        while leaves:
            leave=leaves.pop()
            nodes_included.remove(leave)
            nodes_to_merge=[leave]
            nodes_to_visit=EPG_copy.nodes()
            dfs_from_leaf(EPG_copy,leave,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            clusters_to_merge.append(nodes_to_merge)
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])] = {}
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])]['nodes'] = nodes_to_merge
            nodes_to_delete = nodes_to_merge[0:len(nodes_to_merge)-1]
            if EPG_copy.degree()[nodes_to_merge[-1]] == 1: #avoid the single point
                nodes_to_delete = nodes_to_merge
                leaves = []
            EPG_copy.remove_nodes_from(nodes_to_delete)
    dict_branches = Cal_Branch_Length(EPG,dict_branches)
    # print('Number of branches: ' + str(len(clusters_to_merge)))
    return dict_branches

def Project_Cells_To_Tree(EPG,X,dict_branches):
    df_cells = pd.DataFrame(columns=['cell_id','branch_id','node_id','dist','lam','branch_len','pt','pt_proj'])
    dict_nodes_pos = nx.get_node_attributes(EPG,'pos')
    nodes_pos = np.empty((0,X.shape[1]))
    nodes_label = np.empty((0,1),dtype=int)
    for x in dict_nodes_pos.keys():
        nodes_pos = np.vstack((nodes_pos,dict_nodes_pos[x]))
        nodes_label = np.append(nodes_label,x)
    indices = pairwise_distances_argmin_min(X,nodes_pos,axis=1,metric='euclidean')[0]
    x_label = nodes_label[indices]
    for ix,xp in enumerate(X): 
        list_br_id = [br_key for br_key,br_value in dict_branches.iteritems() if x_label[ix] in br_value['nodes']]
        dict_br_matrix = dict()
        for br_id in list_br_id:
            dict_br_matrix[br_id] = np.array([dict_nodes_pos[i] for i in dict_branches[br_id]['nodes']])            
        dict_results = dict()
        list_dist_xp = list()
        for br_id in list_br_id:
            dict_results[br_id] = project_point_to_line_segment_matrix(dict_br_matrix[br_id],xp)
            list_dist_xp.append(dict_results[br_id][2])
        br_id_assigned = list_br_id[np.argmin(list_dist_xp)]
        br_len = dict_branches[br_id_assigned]['len']
        results = dict_results[br_id_assigned]
        pt_proj = results[0]
        dist_proj = results[2]
        lam_proj = results[3]
        if(lam_proj>br_len):
            lam_proj = br_len 
        df_cells.loc[ix] = [ix,br_id_assigned,x_label[ix],dist_proj,lam_proj,br_len,xp,pt_proj]
    return df_cells


def Plot_Extract_Branches(EPG,X,dict_branches,curves_color,file_path,file_name='EPG_Branches'):
    dict_nodes_pos = nx.get_node_attributes(EPG,'pos')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')   
    for br_id in dict_branches.keys():
        branch = dict_branches[br_id]['nodes']
        EPG_sub = EPG.subgraph(branch)
        branch_color = curves_color[br_id]
        for EPG_sub_edge in EPG_sub.edges_iter():
            x_pos = (dict_nodes_pos[EPG_sub_edge[0]][0],dict_nodes_pos[EPG_sub_edge[1]][0])
            y_pos = (dict_nodes_pos[EPG_sub_edge[0]][1],dict_nodes_pos[EPG_sub_edge[1]][1])
            z_pos = (dict_nodes_pos[EPG_sub_edge[0]][2],dict_nodes_pos[EPG_sub_edge[1]][2])
            ax.plot(x_pos,y_pos,z_pos,c = branch_color,lw=5,zorder=None)
    nodes_pos = np.array(dict_nodes_pos.values())
    ax.scatter(nodes_pos[:,0],nodes_pos[:,1],nodes_pos[:,2],color='grey',s=8,alpha=1,zorder=2)
    for i in dict_nodes_pos.keys():
        ax.text(dict_nodes_pos[i][0],dict_nodes_pos[i][1],dict_nodes_pos[i][2],i,color='black',fontsize = 10)
    max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
    mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
    mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
    mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('LLE Component1',labelpad=20)
    ax.set_ylabel('LLE Component2',labelpad=20)
    ax.set_zlabel('LLE Component3',labelpad=20)
    plt.savefig(file_path + '/'+str(file_name)+'.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)
    
def Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,file_name = 'EPG',dict_node_state = None):
    dict_nodes_pos = nx.get_node_attributes(EPG,'pos')
    X = np.array(df_flat_tree['X'].tolist())
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    df_color = pd.DataFrame(columns=['cell_label','color'])
    df_color['cell_label'] = df_flat_tree['CELL_LABEL']   
    list_patches = []
    for x in input_cell_label_uni_color.keys():
        id_cells = np.where(df_flat_tree['CELL_LABEL']==x)[0]
        df_color.loc[df_color.index[id_cells],'color'] = input_cell_label_uni_color[x]
        list_patches.append(Patches.Patch(color = input_cell_label_uni_color[x],label=x))
        
    X_plot = pd.DataFrame(X).sample(frac=1,random_state=100)
    X_color = df_color.sample(frac=1,random_state=100)['color']
    ax.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1],X_plot.iloc[:, 2], c=X_color,s=50,linewidth=0,alpha=0.8)  
    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
              ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)
    if((flag_web) & (dict_node_state!=None)):
        pd.DataFrame(X_plot.values,index=X_color,columns=['D'+str(x) for x in range(X_plot.shape[1])]).to_csv\
        (file_path + '/coord_cells.csv',sep='\t')

    for br_id in dict_branches.keys():
        br_value = dict_branches[br_id]['nodes']
        br_color = curves_color[br_id]
        br_nodes_pos = np.array([dict_nodes_pos[i] for i in br_value])
        ax.plot(br_nodes_pos[:,0],br_nodes_pos[:,1],br_nodes_pos[:,2],c = br_color,lw=5,zorder=10)
        if((flag_web) & (dict_node_state!=None)):
            pd.DataFrame(br_nodes_pos).to_csv(file_path + '/coord_curve_'+dict_node_state[br_id[0]] + '_' + dict_node_state[br_id[1]]+'.csv',sep='\t',index=False)

    if(dict_node_state!=None):
        for node_id in dict_node_state.keys():
            ax.text(dict_nodes_pos[node_id][0],dict_nodes_pos[node_id][1],dict_nodes_pos[node_id][2],
                    dict_node_state[node_id],color='black',fontsize = 12,zorder=10)
        if(flag_web):
            dict_node_state_pos = {dict_node_state[node_id]:dict_nodes_pos[node_id] for node_id in dict_node_state.keys()}
            pd.DataFrame(dict_node_state_pos).T.to_csv(file_path + '/coord_states.csv',sep='\t')
    max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
    mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
    mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
    mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('LLE Component1',labelpad=20)
    ax.set_ylabel('LLE Component2',labelpad=20)
    ax.set_zlabel('LLE Component3',labelpad=20)    
    plt.savefig(file_path + '/'+str(file_name)+'.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)

#construct tree structure
def Contruct_Tree(dict_branches):
    flat_tree = nx.Graph()
    flat_tree.add_nodes_from(list(set(itertools.chain.from_iterable(dict_branches.keys()))))
    flat_tree.add_edges_from(dict_branches.keys())
    return flat_tree

def Construct_Node_State(flat_tree):
    visited,visited_branches = Breadth_First_Search(flat_tree)
    dict_node_state = dict()
    for i,node in enumerate(visited):
        dict_node_state[node] = 'S'+str(i)
    return dict_node_state

def Structure_Learning(df_flat_tree,AP_damping_factor,n_cluster,lle_n_nb_percent,EPG_n_nodes,EPG_n_rep,EPG_lambda,EPG_mu,EPG_trimmingradius,EPG_prob,EPG_finalenergy,EPG_alpha,EPG_beta,
                        flag_disable_EPG_collapse,EPG_collapse_mode,EPG_collapse_par,
                        flag_EPG_shift,EPG_shift_mode,EPG_shift_DR,EPG_shift_maxshift,
                        flag_disable_EPG_ext,EPG_ext_mode,EPG_ext_par,
                        n_processes,input_cell_label_uni,input_cell_label_uni_color,file_path,file_path_precomp,flag_web):
    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()

    X = np.array(df_flat_tree['X'].tolist())
    #Spectral clustering
    print('Clustering...')
    AF = AffinityPropagation(damping=AP_damping_factor).fit(X)
    cluster_labels = AF.labels_
    XC = AF.cluster_centers_

    # SC = SpectralClustering(n_clusters=n_cluster,affinity='nearest_neighbors',n_neighbors=np.int(X.shape[0]*0.05)).fit(X)
    # cluster_labels = SC.labels_ #cluster centers
    # XC = np.empty((0,X.shape[1]))
    # for x in np.unique(cluster_labels):
    #     indices_cells = np.array(range(X.shape[0]))[cluster_labels==x]
    #     XC = np.vstack((XC,np.median(X[indices_cells,:],axis=0)))

    #Minimum Spanning Tree
    print('Minimum Spanning Tree...')
    D=pairwise_distances(XC)
    G=nx.from_numpy_matrix(D)
    MS=nx.minimum_spanning_tree(G)
    dict_MST_branches = Extract_Branches(MS)
    print('Number of initial branches: ' + str(len(dict_MST_branches))) 
    Plot_MST(MS,XC,X,df_flat_tree,input_cell_label_uni,input_cell_label_uni_color,file_path)


    #Elastic principal graph
    EPG_nodes_pos = XC
    EPG_edges = np.array(MS.edges())
    EPG_n_nodes = max(EPG_n_nodes,XC.shape[0]+30)
    print('Elastic Principal Graph...')

    if(flag_web):
        TreeEPG_obj = ElPiGraph.computeElasticPrincipalTree(X=X,
                                                            NumNodes = EPG_n_nodes, 
                                                            Lambda=EPG_lambda, Mu=EPG_mu,
                                                            TrimmingRadius= EPG_trimmingradius,
                                                            InitNodePositions = EPG_nodes_pos,
                                                            InitEdges=EPG_edges + 1,
                                                            Do_PCA=False,CenterData=False,
                                                            # n_cores = n_processes,
                                                            n_cores = 1,
                                                            nReps=EPG_n_rep,
                                                            ProbPoint=EPG_prob,
                                                            drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                                            FinalEnergy = EPG_finalenergy,
                                                            alpha = EPG_alpha)   
    else:     
        R.pdf(file_path + '/'+'Initial_ElPigraph.pdf')
        TreeEPG_obj = ElPiGraph.computeElasticPrincipalTree(X=X,
                                                            NumNodes = EPG_n_nodes, 
                                                            Lambda=EPG_lambda, Mu=EPG_mu,
                                                            TrimmingRadius= EPG_trimmingradius,
                                                            InitNodePositions = EPG_nodes_pos,
                                                            InitEdges=EPG_edges + 1,
                                                            Do_PCA=False,CenterData=False,
                                                            # n_cores = n_processes,
                                                            n_cores = 1,
                                                            nReps=EPG_n_rep,
                                                            ProbPoint=EPG_prob,
                                                            FinalEnergy = EPG_finalenergy,
                                                            alpha = EPG_alpha)
        R('dev.off()')


    #Initial Tree structure
    if(EPG_n_rep>1):
        EPG_nodes_pos = np.array(TreeEPG_obj[EPG_n_rep].rx2('NodePositions'))
        EPG_edges = np.array((TreeEPG_obj[EPG_n_rep].rx2('Edges')).rx2('Edges'),dtype=int)-1    
    else:
        EPG_nodes_pos = np.array(TreeEPG_obj[0].rx2('NodePositions'))
        EPG_edges = np.array((TreeEPG_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1

    EPG=nx.Graph()
    EPG.add_nodes_from(range(EPG_nodes_pos.shape[0]))
    EPG.add_edges_from(EPG_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(EPG_nodes_pos)}
    nx.set_node_attributes(EPG, 'pos', dict_nodes_pos)

    dict_branches = Extract_Branches(EPG)
    sns_palette = sns.color_palette("hls", len(dict_branches))
    curves_color = {x:sns_palette[i] for i,x in enumerate(dict_branches.keys())}
    print('Number of branches after initial ElPiGraph: ' + str(len(dict_branches)))
    if(not flag_web):
        Plot_Extract_Branches(EPG,X,dict_branches,curves_color,file_path,file_name='Init_EPG_Branches')
        Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,file_name = 'Init_EPG') 


    #Filtering branches with cells fewer than cutoff
    if(not flag_disable_EPG_collapse):
        print('Collasping small branches ...')
        if(EPG_n_rep>1):
            # Collapsed_P = ElPiGraph.CollapseBrances(X = X, TargetPG = Init_TreeEPG[EPG_n_rep], Mode = EPG_collapse_mode, ControlPar = max(5,ceil(X.shape[0]*EPG_collapse_par)))
            Collapsed_obj = ElPiGraph.CollapseBrances(X = X, TargetPG = TreeEPG_obj[EPG_n_rep], Mode = EPG_collapse_mode, ControlPar = EPG_collapse_par)
        else:
            # Collapsed_P = ElPiGraph.CollapseBrances(X = X, TargetPG = Init_TreeEPG[0], Mode = EPG_collapse_mode, ControlPar = max(5,ceil(X.shape[0]*EPG_collapse_par)))
            Collapsed_obj = ElPiGraph.CollapseBrances(X = X, TargetPG = TreeEPG_obj[0], Mode = EPG_collapse_mode, ControlPar = EPG_collapse_par)
        EPG_nodes_pos = np.array(Collapsed_obj.rx2('Nodes'))
        EPG_edges = np.array(Collapsed_obj.rx2('Edges')) - 1 

    #Optimizing braching point
    TreeEPG_obj  = ElPiGraph.fineTuneBR(X = X, 
                                        NumNodes = EPG_nodes_pos.shape[0]+30,
                                        Lambda = EPG_lambda/2.0, Mu = EPG_mu,
                                        TrimmingRadius = EPG_trimmingradius,
                                        InitNodePositions = EPG_nodes_pos,
                                        InitEdges = EPG_edges + 1,
                                        Do_PCA = False, CenterData = False,
                                        # n_cores = n_processes, 
                                        n_cores = 1,
                                        nReps=EPG_n_rep,
                                        ProbPoint=EPG_prob,                                    
                                        MaxSteps = 50, Mode = 2,
                                        drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                        alpha = EPG_alpha,
                                        FinalEnergy = 'base')

    if(flag_EPG_shift):
        print('Shifting branching point to denser area ...')
        if(EPG_n_rep>1):
            Shifted_obj = ElPiGraph.ShiftBranching(X = X, 
                                                   TargetPG = TreeEPG_obj[EPG_n_rep],
                                                   TrimmingRadius = EPG_trimmingradius,             
                                                   SelectionMode = EPG_shift_mode, 
                                                   DensityRadius = EPG_shift_DR,
                                                   MaxShift = EPG_shift_maxshift)
        else:
            Shifted_obj = ElPiGraph.ShiftBranching(X = X, 
                                                   TargetPG = TreeEPG_obj[0], 
                                                   TrimmingRadius = EPG_trimmingradius,                       
                                                   SelectionMode = EPG_shift_mode, 
                                                   DensityRadius = EPG_shift_DR,
                                                   MaxShift = EPG_shift_maxshift)
        EPG_nodes_pos = np.array(Shifted_obj.rx2('NodePositions'))
        EPG_edges = np.array(Shifted_obj.rx2('Edges')) - 1         
        
        TreeEPG_obj = ElPiGraph.computeElasticPrincipalTree(X = X, 
                                                            NumNodes = EPG_nodes_pos.shape[0],
                                                            Lambda = EPG_lambda/2.0, Mu = EPG_mu,
                                                            TrimmingRadius = EPG_trimmingradius,
                                                            InitNodePositions = EPG_nodes_pos,
                                                            InitEdges = EPG_edges + 1,
                                                            Do_PCA = False, CenterData = False,
                                                            # n_cores = n_processes,
                                                            n_cores = 1, 
                                                            nReps=EPG_n_rep,
                                                            ProbPoint=EPG_prob,       
                                                            drawAccuracyComplexity = False, drawEnergy = False,drawPCAView = False,
                                                            FinalEnergy = 'base',
                                                            alpha = EPG_alpha)


    if(EPG_n_rep>1):
        EPG_nodes_pos = np.array(TreeEPG_obj[EPG_n_rep].rx2('NodePositions'))
        EPG_edges = np.array((TreeEPG_obj[EPG_n_rep].rx2('Edges')).rx2('Edges'),dtype=int)-1    
    else:
        EPG_nodes_pos = np.array(TreeEPG_obj[0].rx2('NodePositions'))
        EPG_edges = np.array((TreeEPG_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1
    EPG=nx.Graph()
    EPG.add_nodes_from(range(EPG_nodes_pos.shape[0]))
    EPG.add_edges_from(EPG_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(EPG_nodes_pos)}
    nx.set_node_attributes(EPG, 'pos', dict_nodes_pos)   

    dict_branches = Extract_Branches(EPG)
    sns_palette = sns.color_palette("hls", len(dict_branches))
    curves_color = {x:sns_palette[i] for i,x in enumerate(dict_branches.keys())}
    print('Number of branches after optimization: ' + str(len(dict_branches)))
    if(not flag_web):
        Plot_Extract_Branches(EPG,X,dict_branches,curves_color,file_path)
        Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web)    

    dict_branches_ori = deepcopy(dict_branches)

    #Extend leaves with additional nodes
    if(not flag_disable_EPG_ext):
        print('Extending leaves with additional nodes ...')
        if(EPG_n_rep>1):
            TreeEPG_obj = ElPiGraph.ExtendLeaves(X = X, 
                                                 TargetPG = TreeEPG_obj[nRep], 
                                                 Mode = EPG_ext_mode, 
                                                 ControlPar = EPG_ext_par,
                                                 TrimmingRadius=EPG_trimmingradius, 
                                                 PlotSelected = False)
        else:
            TreeEPG_obj = ElPiGraph.ExtendLeaves(X = X, 
                                                 TargetPG = TreeEPG_obj[0], 
                                                 Mode = EPG_ext_mode, 
                                                 ControlPar = EPG_ext_par,
                                                 TrimmingRadius=EPG_trimmingradius, 
                                                 PlotSelected = False)  

        EPG_nodes_pos = np.array(TreeEPG_obj.rx2('NodePositions'))
        EPG_edges = np.array((TreeEPG_obj.rx2('Edges')).rx2('Edges'),dtype=int)-1    
        EPG=nx.Graph()
        EPG.add_nodes_from(range(EPG_nodes_pos.shape[0]))
        EPG.add_edges_from(EPG_edges)
        dict_nodes_pos = {i:x for i,x in enumerate(EPG_nodes_pos)}
        nx.set_node_attributes(EPG, 'pos', dict_nodes_pos)    

        dict_branches = Extract_Branches(EPG)
        sns_palette = sns.color_palette("hls", len(dict_branches))
        curves_color = {x:sns_palette[i] for i,x in enumerate(dict_branches.keys())}
        print('Number of branches after extension: ' + str(len(dict_branches)))
        if(not flag_web):
            Plot_Extract_Branches(EPG,X,dict_branches,curves_color,file_path,file_name='Ext_EPG_Branches')
            Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,file_name = 'Ext_EPG')
    df_cells = Project_Cells_To_Tree(EPG,X,dict_branches)
    flat_tree = Contruct_Tree(dict_branches)
    dict_node_state = Construct_Node_State(flat_tree)
    Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,
            file_name = 'Final_EPG',dict_node_state=dict_node_state)

    for x_br in dict_branches.keys():
        x_nodes = dict_branches[x_br]['nodes']
        dict_branches[x_br]['len_ori'] = 0
        for y_br in dict_branches_ori.keys():
            if set(y_br).issubset(x_nodes):
                dict_branches[x_br]['len_ori'] = dict_branches[x_br]['len_ori']+dict_branches_ori[y_br]['len']

    nx.set_edge_attributes(flat_tree,'len',{x: dict_branches[x]['len'] for x in dict_branches.keys()})  
    nx.set_edge_attributes(flat_tree,'len_ori',{x: dict_branches[x]['len_ori'] for x in dict_branches.keys()})  

    #Update sample dataframe
    df_flat_tree['X_projected'] = df_cells['pt_proj']
    df_flat_tree['node_id'] = df_cells['node_id']
    df_flat_tree['lam'] = df_cells['lam']
    df_flat_tree['dist'] = df_cells['dist']
    df_flat_tree['branch_len'] = df_cells['branch_len']
    df_flat_tree['branch_id'] = df_cells['branch_id']
    df_flat_tree['branch_len_ori'] = ""
    df_flat_tree['lam_contracted'] = ""
    df_flat_tree['dist_contracted'] = ""
    list_ratio_len = list()
    for x_br in dict_branches.keys():
        id_cells = df_flat_tree[df_flat_tree['branch_id'] == x_br].index
        if(len(id_cells)>0):
            ratio_len = dict_branches[x_br]['len']/float(dict_branches[x_br]['len_ori'])
            df_flat_tree.loc[id_cells,'lam_contracted'] = df_cells.loc[id_cells,'lam']/ratio_len
            df_flat_tree.loc[id_cells,'branch_len_ori'] = dict_branches[x_br]['len_ori']
            list_ratio_len.append(ratio_len)
            # dist_p = np.log2((df_flat_tree.loc[id_cells,'dist']+1).tolist())
            # df_flat_tree.loc[id_cells,'dist_contracted'] = 0.7*min_len_ori*dist_p/max(dist_p)
    df_flat_tree['dist_contracted'] = df_flat_tree['dist']/max(list_ratio_len)
    df_flat_tree = df_flat_tree.astype('object')  
    return df_flat_tree,EPG,flat_tree,dict_branches,dict_node_state,curves_color


def Plot_Flat_Tree(df_flat_tree,EPG,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,n_processes,file_path,file_name='flat_tree_mds'):
    dict_nodes_pos = nx.get_node_attributes(EPG,'pos')
    X = np.array(df_flat_tree['X'].tolist())
    nodes_pos = np.empty((0,X.shape[1]))
    nodes_label = np.empty((0,1),dtype=int)
    for x in dict_nodes_pos.keys():
        nodes_pos = np.vstack((nodes_pos,dict_nodes_pos[x]))
        nodes_label = np.append(nodes_label,x)
    # ##Multidimensional scaling
    # X_and_nodes = np.vstack((X,nodes_pos))
    # mds = MDS(2,n_jobs=n_processes)
    # X_and_nodes_mds = mds.fit_transform(X_and_nodes)
    # X_mds = X_and_nodes_mds[:X.shape[0],:]
    # nodes_mds = X_and_nodes_mds[X.shape[0]:,:]
    # dict_nodes_pos_mds = {}
    # for i,x in enumerate(nodes_label):
    #     dict_nodes_pos_mds[x] = nodes_mds[i,:]

    ##Multidimensional scaling
    # print(n_processes)
    mds = MDS(2,n_jobs=n_processes,random_state=100)
    # print(X.shape)
    X_mds = mds.fit_transform(X)  
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    df_color = pd.DataFrame(columns=['cell_label','color'])
    df_color['cell_label'] = df_flat_tree['CELL_LABEL']   
    list_patches = []
    for x in input_cell_label_uni_color.keys():
        id_cells = np.where(df_flat_tree['CELL_LABEL']==x)[0]
        df_color.loc[df_color.index[id_cells],'color'] = input_cell_label_uni_color[x]
        list_patches.append(Patches.Patch(color = input_cell_label_uni_color[x],label=x))
        
    X_plot = pd.DataFrame(X_mds).sample(frac=1,random_state=100)
    X_color = df_color.sample(frac=1,random_state=100)['color']
    ax.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8)  

    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
              ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)
    plt.tight_layout(h_pad=1.0)
    plt.savefig(file_path + '/'+str(file_name)+'_labels.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)       

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    df_color = pd.DataFrame(columns=['branch_id','color'])
    df_color['branch_id'] = df_flat_tree['branch_id']   
    list_patches = []
    for br_id in dict_branches.keys():
        id_cells = np.where(df_flat_tree['branch_id']==br_id)[0]
        df_color.loc[df_color.index[id_cells],'color'] = curves_color[br_id]
        list_patches.append(Patches.Patch(color = curves_color[br_id],
            label='branch '+dict_node_state[br_id[0]]+'_'+dict_node_state[br_id[1]]))
        
    X_plot = pd.DataFrame(X_mds).sample(frac=1,random_state=100)
    X_color = df_color.sample(frac=1,random_state=100)['color']
    ax.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8)  
    ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
              ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)   
    plt.tight_layout(h_pad=1.0)
    plt.savefig(file_path + '/'+str(file_name)+'_branches.pdf',pad_inches=1,bbox_inches='tight')
    plt.close(fig)       


def Flat_Tree_Plot(df_flat_tree,flat_tree,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,mode = 'normal'):
    g = igraph.Graph()
    g_label = dict_node_state.keys()
    g.add_vertices(len(dict_node_state))
    g.add_edges([(g_label.index(x[0]),g_label.index(x[1])) for x in flat_tree.edges()])
    g.vs["label"] = g_label
    # nodes_pos_array = np.array(g.layout_reingold_tilford_circular())
    np.random.seed(100)
    nodes_pos_array = np.array(g.layout_fruchterman_reingold(seed=np.random.random((len(dict_node_state), 2))))
    id_cluster = dict_node_state.keys()

    dict_nodes_pos = {x:nodes_pos_array[g_label.index(x)] for x in id_cluster}
    visited,visited_branches = Breadth_First_Search(flat_tree)

    dict_nodes_pos_updated = deepcopy(dict_nodes_pos)
    flat_tree_copy = flat_tree.copy()
    flat_tree_copy.remove_node(visited[0])
    for id_br,br in enumerate(visited_branches):
        cur_dist = distance.euclidean(dict_nodes_pos_updated[br[0]],dict_nodes_pos_updated[br[1]])
        if(mode == 'normal'):
            if(br in dict_branches.keys()):
                pc_length = dict_branches[br]['len']
            else:
                pc_length = dict_branches[(br[1],br[0])]['len']
        if(mode == 'contracted'):
            if(br in dict_branches.keys()):
                pc_length = dict_branches[br]['len_ori']
            else:
                pc_length = dict_branches[(br[1],br[0])]['len_ori']
        st_x = dict_nodes_pos_updated[br[0]][0]
        ed_x = dict_nodes_pos_updated[br[1]][0]
        st_y = dict_nodes_pos_updated[br[0]][1]
        ed_y = dict_nodes_pos_updated[br[1]][1]
        p_x = st_x + (ed_x - st_x)*(pc_length/cur_dist)
        p_y = st_y + (ed_y - st_y)*(pc_length/cur_dist)
        dict_nodes_pos_updated[br[1]] = np.array([p_x,p_y])

        con_components = list(nx.connected_components(flat_tree_copy))
        #update other reachable unvisited nodes
        for con_comp in con_components:
            if br[1] in con_comp:
                reachable_unvisited = con_comp - {br[1]}
                flat_tree_copy.remove_node(br[1])
                break
        for nd in reachable_unvisited:
            nd_x = dict_nodes_pos_updated[nd][0] + p_x - ed_x
            nd_y = dict_nodes_pos_updated[nd][1] + p_y - ed_y
            dict_nodes_pos_updated[nd] = np.array([nd_x,nd_y])

    nx.set_node_attributes(flat_tree, 'nodes_pos', dict_nodes_pos_updated)

    if(mode == 'normal'):
        df_flat_tree['pos_ft'] = ''
        for br_i in dict_branches.keys():
            s_pos = dict_nodes_pos_updated[br_i[0]] #start node position
            e_pos = dict_nodes_pos_updated[br_i[1]] #end node position
            dist_se = distance.euclidean(s_pos,e_pos)
            p_x = np.array(df_flat_tree[df_flat_tree['branch_id']==br_i]['lam'].tolist())
            dist_p = np.array(df_flat_tree[df_flat_tree['branch_id']==br_i]['dist'].tolist())
            np.random.seed(100)
            p_y = np.random.choice([1,-1],size=len(p_x))*dist_p
            #rotation matrix
            ro_angle = np.arctan2((e_pos-s_pos)[1],(e_pos-s_pos)[0])#counterclockwise angle
            p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y*math.sin(ro_angle)
            p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y*math.cos(ro_angle)
            p_pos = np.array((p_x_prime,p_y_prime)).T
            df_flat_tree.loc[df_flat_tree[df_flat_tree['branch_id']==br_i].index,'pos_ft'] =[p_pos[i,:].tolist() for i in range(p_pos.shape[0])]
    if(mode == 'contracted'):
        df_flat_tree['pos_cft'] = ''
        for br_i in dict_branches.keys():
            s_pos = dict_nodes_pos_updated[br_i[0]] #start node position
            e_pos = dict_nodes_pos_updated[br_i[1]] #end node position
            dist_se = distance.euclidean(s_pos,e_pos)
            p_x = np.array(df_flat_tree[df_flat_tree['branch_id']==br_i]['lam_contracted'].tolist())
            dist_p = np.array(df_flat_tree[df_flat_tree['branch_id']==br_i]['dist_contracted'].tolist())
            np.random.seed(100)
            p_y = np.random.choice([1,-1],size=len(p_x))*dist_p
            #rotation matrix
            ro_angle = np.arctan2((e_pos-s_pos)[1],(e_pos-s_pos)[0])#counterclockwise angle
            p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y*math.sin(ro_angle)
            p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y*math.cos(ro_angle)
            p_pos = np.array((p_x_prime,p_y_prime)).T
            df_flat_tree.loc[df_flat_tree[df_flat_tree['branch_id']==br_i].index,'pos_cft'] =[p_pos[i,:].tolist() for i in range(p_pos.shape[0])]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
    for br_i in dict_branches.keys():
        flat_tree[br_i[0]][br_i[1]]['color'] = curves_color[br_i]
    edges = flat_tree.edges()
    if(mode == 'normal'):   
        array_edges = deepcopy(np.array(edges))
        array_edges = array_edges.astype(str)
        df_nodes = pd.DataFrame(columns=['D' + str(x) for x in range(2)])
        for x in dict_node_state.keys():
            np.place(array_edges,array_edges==str(x),dict_node_state[x])
            df_nodes.loc[dict_node_state[x]] = dict_nodes_pos_updated[x]
        pd.DataFrame(array_edges).to_csv(file_path+'/edges.tsv',sep = '\t',index = False,header=False)
        df_nodes.sort_index(inplace=True)
        df_nodes.to_csv(file_path+'/nodes.tsv',sep = '\t',index = True,header=True)

    edges_color = [flat_tree[u][v]['color'] for u,v in edges]
    nx.draw_networkx(flat_tree,pos=dict_nodes_pos_updated,labels=dict_node_state,node_color='white',alpha=1,\
                    edges=dict_branches.keys(), edge_color=edges_color, width = 6,node_size=0,font_size=15)
    # nx.draw_networkx(flat_tree,pos=nodes_pos_mds_updated,labels=dict_node_state,node_color='white',alpha=0.2)
    df_color = pd.DataFrame(columns=['cell_label','color'])
    df_color['cell_label'] = df_flat_tree['CELL_LABEL']   
    list_patches = []
    for x in input_cell_label_uni_color.keys():
        id_cells = np.where(df_flat_tree['CELL_LABEL']==x)[0]
        df_color.loc[df_color.index[id_cells],'color'] = input_cell_label_uni_color[x]
        list_patches.append(Patches.Patch(color = input_cell_label_uni_color[x],label=x))  

    if(mode == 'normal'):
        X_plot = np.array(df_flat_tree['pos_ft'].sample(frac=1,random_state=100).tolist())
        X_color = df_color.sample(frac=1,random_state=100)['color']
        if(not flag_web):
            ax.scatter(X_plot[:, 0], X_plot[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8)  
            ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                      ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5) 
            plt.savefig(file_path + '/flat_tree.pdf',pad_inches=1,bbox_inches='tight')
            plt.close(fig)       
        else:
            pd.DataFrame(X_plot,index=X_color,columns=['D'+str(x) for x in range(X_plot.shape[1])]).to_csv\
            (file_path + '/flat_tree_coord_cells.csv',sep='\t')


    if(mode == 'contracted'):
        X_plot = np.array(df_flat_tree['pos_cft'].sample(frac=1,random_state=100).tolist())
        X_color = df_color.sample(frac=1,random_state=100)['color']
        ax.scatter(X_plot[:, 0], X_plot[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8)  
        ax.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)        
        plt.savefig(file_path + '/contracted_flat_tree.pdf',pad_inches=1,bbox_inches='tight')
        plt.close(fig)        


def Breadth_First_Search(flat_tree,s=None):
    leaves=[i for i,x in flat_tree.degree().items() if x==1]
    if(s==None):
        np.random.seed(10)
        s = np.random.choice(leaves)
    visited = list()
    visited_edge = list()
    queue = [s]
    while(queue):
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            unvisited_neighbor = list(set(flat_tree.neighbors(vertex)) - set(visited))
            queue.extend(unvisited_neighbor)
            for id_nb,nb in enumerate(unvisited_neighbor):
                visited_edge.append((vertex,nb))
    return visited,visited_edge


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def Scale_Genes(params):
    df_gene_detection = params[0]
    gene = params[1]
    gene_values = df_gene_detection[gene].copy()
    gene_values = gene_values + max(0,0-min(gene_values))
    max_gene_values = np.percentile(gene_values[gene_values>0],90)
    gene_values[gene_values>max_gene_values] = max_gene_values
    gene_values = gene_values/max_gene_values
    return gene_values

def Cal_Transition_Genes(params):
    edge_i = params[0]
    df_gene_detection = params[1]
    input_genes_expressed = params[2]
    TG_spearman_cutoff = params[3]
    TG_diff_cutoff = params[4]
    dict_node_state = params[5]
    df_cells_edge_i = deepcopy(df_gene_detection[df_gene_detection.branch_id==edge_i])
    df_cells_edge_i_sort = df_cells_edge_i.sort_values(['lam'])
    df_stat_pval_qval = pd.DataFrame(columns = ['stat','diff','pval','qval'],dtype=float)
    for genename in input_genes_expressed:
        id_initial = range(0,int(df_cells_edge_i_sort.shape[0]*0.2))
        id_final = range(int(df_cells_edge_i_sort.shape[0]*0.8),int(df_cells_edge_i_sort.shape[0]*1))
        values_initial = df_cells_edge_i_sort.iloc[id_initial,:][genename]
        values_final = df_cells_edge_i_sort.iloc[id_final,:][genename]
        diff_initial_final = values_final.mean() - values_initial.mean()
        if(abs(diff_initial_final)>TG_diff_cutoff):
            df_stat_pval_qval.loc[genename] = np.nan
            df_stat_pval_qval.loc[genename,['stat','pval']] = spearmanr(df_cells_edge_i_sort.loc[:,genename],\
                                                                        df_cells_edge_i_sort.loc[:,'lam'])
            df_stat_pval_qval.loc[genename,'diff'] = diff_initial_final
    if(df_stat_pval_qval.shape[0]==0):
        print('No Transition genes are detected in branch ' + dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]])
        return pd.DataFrame()
    else:
        p_values = df_stat_pval_qval['pval']
        q_values = multipletests(p_values, method='fdr_bh')[1]
        df_stat_pval_qval['qval'] = q_values
        return df_stat_pval_qval[(abs(df_stat_pval_qval.stat)>=TG_spearman_cutoff)].sort_values(['qval'])

#### TG (Transition Genes) along each branch
def TG_Genes_Detection(df_gene_detection,input_genes_expressed,TG_spearman_cutoff,TG_diff_cutoff,dict_node_state,n_processes,num_displayed_genes,file_path,flag_web):
    file_path_TG = file_path + '/Transition_Genes'
    if(not os.path.exists(file_path_TG)):
        os.makedirs(file_path_TG)
    all_branches = np.unique(df_gene_detection['branch_id']).tolist()
    params = [(x,df_gene_detection,input_genes_expressed,TG_spearman_cutoff,TG_diff_cutoff,dict_node_state) for x in all_branches]
    pool = multiprocessing.Pool(processes=n_processes)
    results = pool.map(Cal_Transition_Genes,params)
    pool.close()
    dict_TG_edges = {edge_i: results[i] for i,edge_i in enumerate(all_branches) if(results[i].shape[0]>0)}
    colors = sns.color_palette("Set1", n_colors=8, desat=0.8)
    # colors = [sns.xkcd_rgb['faded orange'],sns.xkcd_rgb['faded green']]
    for edge_i in dict_TG_edges.keys():
        df_TG_edge_i = deepcopy(dict_TG_edges[edge_i])
        df_TG_edge_i.to_csv(file_path_TG+'/Transition_Genes_'+ dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]] + '.tsv',sep = '\t',index = True)
        df_TG_edge_i = df_TG_edge_i.iloc[:num_displayed_genes,:]

        stat = df_TG_edge_i.stat[::-1]
        qvals = df_TG_edge_i.qval[::-1]

        pos = arange(df_TG_edge_i.shape[0])-1
        bar_colors = tile(colors[4],(len(stat),1))
        # bar_colors = repeat(colors[0],len(stat))
        id_neg = np.arange(len(stat))[np.array(stat<0)]
        bar_colors[id_neg]=colors[2]

        fig = plt.figure(figsize=(12,ceil(0.4*len(stat))))
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
        if(flag_web):
            plt.savefig(file_path_TG+'/Transition_Genes_'+ dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]]+'.png',\
                        pad_inches=1,bbox_inches='tight')
        else:
            plt.savefig(file_path_TG+'/Transition_Genes_'+ dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]]+'.pdf',\
                        pad_inches=1,bbox_inches='tight')
        plt.close(fig)
    return dict_TG_edges


def Cal_DE_Genes(params):
    pair_i = params[0]
    df_gene_detection = params[1]
    input_genes_expressed = params[2]
    DE_z_score_cutoff = params[3]
    DE_diff_cutoff = params[4]
    dict_node_state = params[5]
    df_cells_sub1 = df_gene_detection[df_gene_detection.branch_id==pair_i[0]]
    df_cells_sub2 = df_gene_detection[df_gene_detection.branch_id==pair_i[1]]
    #only use Mann-Whitney U test when the number of observation in each sample is > 20
    if(df_cells_sub1.shape[0]>20 and df_cells_sub2.shape[0]>20):
        df_DE_pval_qval = pd.DataFrame(columns = ['z_score','U','diff','mean_up','mean_down','pval','qval'],dtype=float)
        for genename in input_genes_expressed:
            sub1_values = df_cells_sub1.loc[:,genename].tolist()
            sub2_values = df_cells_sub2.loc[:,genename].tolist()
            diff_mean = mean(sub1_values) - mean(sub2_values)
            if(abs(diff_mean)>DE_diff_cutoff):
                df_DE_pval_qval.loc[genename] = np.nan
                df_DE_pval_qval.loc[genename,['U','pval']] = mannwhitneyu(sub1_values,sub2_values,alternative='two-sided')
                df_DE_pval_qval.loc[genename,'diff'] = diff_mean
                df_DE_pval_qval.loc[genename,'mean_up'] = mean(sub1_values)
                df_DE_pval_qval.loc[genename,'mean_down'] = mean(sub2_values)
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
                df_DE_pval_qval.loc[genename,'z_score'] = (df_DE_pval_qval.loc[genename,'U']-mu_U)/sigma_U
        if(df_DE_pval_qval.shape[0]==0):
            print('No DE genes are detected between branches ' + dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]]+\
                  ' and '+dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]])
            return pd.DataFrame()
        else:
            p_values = df_DE_pval_qval['pval']
            q_values = multipletests(p_values, method='fdr_bh')[1]
            df_DE_pval_qval['qval'] = q_values
            return df_DE_pval_qval[abs(df_DE_pval_qval.z_score)>DE_z_score_cutoff].sort_values(['z_score'])
    else:
        print('There are not sufficient cells (should be greater than 20) between branches '+\
              dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]] +' and '+\
              dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]]+ '. fold_change is calculated')
        df_DE_pval_qval = pd.DataFrame(columns = ['fold_change','diff','mean_up','mean_down'],dtype=float)
        for genename in input_genes_expressed:
            sub1_values = df_cells_sub1.loc[:,genename].tolist()
            sub2_values = df_cells_sub2.loc[:,genename].tolist()
            diff_mean = mean(sub1_values) - mean(sub2_values)
            if(abs(diff_mean)>DE_diff_cutoff):
                df_DE_pval_qval.loc[genename] = np.nan
                #make sure the largest fold change is 5
                df_DE_pval_qval.loc[genename,'fold_change'] = log2((mean(sub1_values)+1/24.0)/(mean(sub2_values)+1/24.0))
                df_DE_pval_qval.loc[genename,'diff'] = diff_mean
                df_DE_pval_qval.loc[genename,'mean_up'] = mean(sub1_values)
                df_DE_pval_qval.loc[genename,'mean_down'] = mean(sub2_values)
        if(df_DE_pval_qval.shape[0]==0):
            print('No DE genes are detected between branches ' + dict_node_state[pair_i[0][0]]+'_'+dict_node_state[pair_i[0][1]]+\
                  ' and '+dict_node_state[pair_i[1][0]]+'_'+dict_node_state[pair_i[1][1]])
            return pd.DataFrame()
        else:
            return df_DE_pval_qval[abs(df_DE_pval_qval.fold_change)>1.5].sort_values(['fold_change'])

#### DE (Differentially expressed genes) between sub-branches
def DE_Genes_Detection(df_gene_detection,input_genes_expressed,DE_z_score_cutoff,DE_diff_cutoff,dict_node_state,n_processes,num_displayed_genes,file_path,flag_web):
    file_path_DE = file_path + '/DE_Genes'
    if(not os.path.exists(file_path_DE)):
        os.makedirs(file_path_DE)
    pairs_branches = list()
    all_branches = np.unique(df_gene_detection['branch_id']).tolist()
    for node_i in dict_node_state.keys():
        neighbor_branches = [x for x in all_branches if node_i in x]
        if(len(neighbor_branches)>1):
            pairs_branches += list(itertools.combinations(neighbor_branches,r=2))
    params = [(x,df_gene_detection,input_genes_expressed,DE_z_score_cutoff,DE_diff_cutoff,dict_node_state) for x in pairs_branches]
    pool = multiprocessing.Pool(processes=n_processes)
    results = pool.map(Cal_DE_Genes,params)
    pool.close()
    dict_DE_greater = {pair_i: results[i][results[i]['diff']>0].sort_values(results[i].columns[0],ascending=False)\
                       for i,pair_i in enumerate(pairs_branches) if(results[i].shape[0]>0)}
    dict_DE_less = {pair_i: results[i][results[i]['diff']<0].sort_values(results[i].columns[0])\
                    for i,pair_i in enumerate(pairs_branches) if(results[i].shape[0]>0)}
    colors = sns.color_palette("Set1", n_colors=8, desat=0.8)
    for sub_edges_i in dict_DE_greater.keys():
        fig = plt.figure(figsize=(20,12))
        gs = gridspec.GridSpec(2,1)
        ax = fig.add_subplot(gs[0],adjustable='box')
        if('z_score' in dict_DE_greater[sub_edges_i].columns):
            if(not dict_DE_greater[sub_edges_i].empty):
                dict_DE_greater[sub_edges_i].to_csv(file_path_DE+'/DE_up_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]] + '.tsv',sep = '\t',index = True)
                val_greater = dict_DE_greater[sub_edges_i].iloc[:num_displayed_genes,:]['z_score'].values  # the bar lengths
                pos_greater = arange(dict_DE_greater[sub_edges_i].iloc[:num_displayed_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_greater = np.repeat(0,num_displayed_genes)
                pos_greater = arange(num_displayed_genes)-1
            ax.bar(pos_greater,val_greater, align='center',color = colors[0])
            ax.plot([pos_greater[0]-1,pos_greater[-1]+1], [DE_z_score_cutoff, DE_z_score_cutoff], "k--",lw=2)
            q_vals = dict_DE_greater[sub_edges_i].iloc[:num_displayed_genes,:]['qval'].values
            for i, q in enumerate(q_vals):
                alignment = {'horizontalalignment': 'center', 'verticalalignment': 'bottom'}
                ax.text(pos_greater[i], val_greater[i]+.1, \
                        "{:.2E}".format(Decimal(str(q))),color='black',fontsize=15,**alignment)
            plt.xticks(pos_greater,dict_DE_greater[sub_edges_i].index,rotation=90)
            ax.set_ylim(0,max(val_greater)+1.5)
            ax.set_ylabel('z_score')
            ax.set_title('DE genes between branches ' + dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and ' + \
                         dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]])
            ax1 = fig.add_subplot(gs[1], adjustable='box')
            if(not dict_DE_less[sub_edges_i].empty):
                dict_DE_less[sub_edges_i].to_csv(file_path_DE+'/DE_down_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]] + '.tsv',sep = '\t',index = True)
                val_less = dict_DE_less[sub_edges_i].iloc[:num_displayed_genes,:]['z_score'].values  # the bar lengths
                pos_less = arange(dict_DE_less[sub_edges_i].iloc[:num_displayed_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_less = np.repeat(0,num_displayed_genes)
                pos_less = arange(num_displayed_genes)-1
            ax1.bar(pos_less,val_less, align='center',color = colors[1])
            ax1.plot([pos_less[0]-1,pos_less[-1]+1], [-DE_z_score_cutoff, -DE_z_score_cutoff], "k--",lw=2)
            q_vals = dict_DE_less[sub_edges_i].iloc[:num_displayed_genes,:]['qval'].values
            for i, q in enumerate(q_vals):
                alignment = {'horizontalalignment': 'center', 'verticalalignment': 'top'}
                ax1.text(pos_less[i], val_less[i]-.1, \
                        "{:.2E}".format(Decimal(str(q))),color='black',fontsize=15,**alignment)
            plt.xticks(pos_less,dict_DE_less[sub_edges_i].index)
            ax1.set_ylim(min(val_less)-1.5,0)
            ax1.set_xticklabels(dict_DE_less[sub_edges_i].index,rotation=90)
            ax1.set_ylabel('z_score')

            ax.set_xlim(-2,14)
            ax1.set_xlim(-2,14)
            ax1.xaxis.tick_top()
            plt.tight_layout()
            if(flag_web):
                plt.savefig(file_path_DE+'/DE_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.png',pad_inches=1,bbox_inches='tight')
            else:
                plt.savefig(file_path_DE+'/DE_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.pdf',pad_inches=1,bbox_inches='tight')
            plt.close(fig)
        else:
            if(not dict_DE_greater[sub_edges_i].empty):
                dict_DE_greater[sub_edges_i].to_csv(file_path_DE+'/DE_up_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]] + '.tsv',sep = '\t',index = True)
                val_greater = dict_DE_greater[sub_edges_i].iloc[:num_displayed_genes,:]['fold_change'].values  # the bar lengths
                pos_greater = arange(dict_DE_greater[sub_edges_i].iloc[:num_displayed_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_greater = np.repeat(0,num_displayed_genes)
                pos_greater = arange(num_displayed_genes)-1
            ax.bar(pos_greater,val_greater, align='center',color = colors[0])
            ax.plot([pos_greater[0]-1,pos_greater[-1]+1], [1.5, 1.5], "k--",lw=2)
            plt.xticks(pos_greater,dict_DE_greater[sub_edges_i].index,rotation=90)
            ax.set_ylim(0,max(val_greater)+1.5)
            ax.set_ylabel('fold_change')
            ax.set_title('DE genes between branches ' + dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and ' + \
                         dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]])
            ax1 = fig.add_subplot(gs[1], adjustable='box')
            if(not dict_DE_less[sub_edges_i].empty):
                dict_DE_less[sub_edges_i].to_csv(file_path_DE+'/DE_down_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]] + '.tsv',sep = '\t',index = True)
                val_less = dict_DE_less[sub_edges_i].iloc[:num_displayed_genes,:]['fold_change'].values  # the bar lengths
                pos_less = arange(dict_DE_less[sub_edges_i].iloc[:num_displayed_genes,:].shape[0])-1    # the bar centers on the y axis
            else:
                val_less = np.repeat(0,num_displayed_genes)
                pos_less = arange(num_displayed_genes)-1
            ax1.bar(pos_less,val_less, align='center',color = colors[1])
            ax1.plot([pos_less[0]-1,pos_less[-1]+1], [-1.5, -1.5], "k--",lw=2)
            plt.xticks(pos_less,dict_DE_less[sub_edges_i].index)
            ax1.set_ylim(min(val_less)-1.5,0)
            ax1.set_xticklabels(dict_DE_less[sub_edges_i].index,rotation=90)
            ax1.set_ylabel('fold_change')

            ax.set_xlim(-2,14)
            ax1.set_xlim(-2,14)
            ax1.xaxis.tick_top()
            plt.tight_layout()
            if(flag_web):
                plt.savefig(file_path_DE+'/DE_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.png',pad_inches=1,bbox_inches='tight')
            else:
                plt.savefig(file_path_DE+'/DE_genes_'+dict_node_state[sub_edges_i[0][0]]+'_'+dict_node_state[sub_edges_i[0][1]] + ' and '\
                            + dict_node_state[sub_edges_i[1][0]]+'_'+dict_node_state[sub_edges_i[1][1]]+'.pdf',pad_inches=1,bbox_inches='tight')
            plt.close(fig)
    return dict_DE_greater,dict_DE_less


### Find transition genes along each branch 
def Genes_Detection_For_Transition(df_flat_tree,df_sc,input_genes,TG_spearman_cutoff,TG_diff_cutoff,dict_node_state,n_processes,file_path,file_path_precomp,flag_web):
    df_gene_detection = df_flat_tree[['CELL_LABEL','branch_id','lam']].copy()
    #exclude genes that are expressed in fewer than min_num_cells cells
    min_num_cells = max(5,int(round(df_flat_tree.shape[0]*0.001)))
    print('Minimum number of cells in one cluster: '+ str(min_num_cells))
    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()

    # id_non_outlier = df_flat_tree[df_flat_tree.is_outlier == 'No'].index
    # df_gene_detection = df_gene_detection.loc[id_non_outlier]
    df_gene_detection.index = range(df_gene_detection.shape[0])

    num_displayed_genes = 15
    num_top_genes = 3

    # if(os.path.exists(file_path_precomp+'/results_scaled_genes.pickle')):
    #     print('Importing precomputed scaled genes ...')
    #     results = Read_From_Pickle('results_scaled_genes',file_path_precomp)
    if(os.path.exists(file_path_precomp+'/results_scaled_genes.msg')):
        print('Importing precomputed scaled genes ...')
        results = pd.read_msgpack(file_path_precomp+'/results_scaled_genes.msg')
    else:
        params = [(df_gene_detection,x) for x in input_genes_expressed]
        pool = multiprocessing.Pool(processes=n_processes)
        results = pool.map(Scale_Genes,params)
        pool.close()
        pd.DataFrame(results).to_msgpack(file_path_precomp + '/results_scaled_genes.msg',compress='zlib')
        # Save_To_Pickle(results,'results_scaled_genes',file_path_precomp)

    df_gene_detection[input_genes_expressed] = pd.DataFrame(results).T
    #### TG (Transition Genes)
    dict_TG_edges = TG_Genes_Detection(df_gene_detection,input_genes_expressed,TG_spearman_cutoff,TG_diff_cutoff,\
                                       dict_node_state,n_processes,num_displayed_genes,file_path,flag_web)
    top_genes =list()
    for i in dict_TG_edges.keys():
        top_genes = top_genes + dict_TG_edges[i].index.tolist()[0:num_top_genes]
    gene_list_TG = list(set(top_genes))
    return gene_list_TG

### Find differentially expressed genes between different sub-branches
def Genes_Detection_For_DE(df_flat_tree,df_sc,input_genes,DE_z_score_cutoff,DE_diff_cutoff,dict_node_state,n_processes,file_path,file_path_precomp,flag_web):
    df_gene_detection = df_flat_tree[['CELL_LABEL','branch_id','lam']].copy()
    #exclude genes that are expressed in fewer than min_num_cells cells
    min_num_cells = max(5,int(round(df_flat_tree.shape[0]*0.001)))
    print('Minimum number of cells in one cluster: '+ str(min_num_cells))
    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()

    # id_non_outlier = df_flat_tree[df_flat_tree.is_outlier == 'No'].index
    # df_gene_detection = df_gene_detection.loc[id_non_outlier]
    df_gene_detection.index = range(df_gene_detection.shape[0])

    num_displayed_genes = 15
    num_top_genes = 3

    # if(os.path.exists(file_path_precomp+'/results_scaled_genes.pickle')):
    #     print('Importing precomputed scaled genes ...')
    #     results = Read_From_Pickle('results_scaled_genes',file_path_precomp)
    if(os.path.exists(file_path_precomp+'/results_scaled_genes.msg')):
        print('Importing precomputed scaled genes ...')
        results = pd.read_msgpack(file_path_precomp+'/results_scaled_genes.msg')
    else:
        params = [(df_gene_detection,x) for x in input_genes_expressed]
        pool = multiprocessing.Pool(processes=n_processes)
        results = pool.map(Scale_Genes,params)
        pool.close()
        pd.DataFrame(results).to_msgpack(file_path_precomp + '/results_scaled_genes.msg',compress='zlib')
        # Save_To_Pickle(results,'results_scaled_genes',file_path_precomp)

    df_gene_detection[input_genes_expressed] = pd.DataFrame(results).T
    #### DE (Differentially expressed genes) between sub-branches
    dict_DE_greater,dict_DE_less = DE_Genes_Detection(df_gene_detection,input_genes_expressed,DE_z_score_cutoff,DE_diff_cutoff,\
                                                      dict_node_state,n_processes,num_displayed_genes,file_path,flag_web)

    top_genes =list()
    for i in dict_DE_greater.keys():
        top_genes = top_genes + dict_DE_greater[i].index.tolist()[0:num_top_genes]
    for i in dict_DE_less.keys():
        top_genes = top_genes + dict_DE_less[i].index.tolist()[0:num_top_genes]
    gene_list_DE = list(set(top_genes))
    return gene_list_DE

def Subway_Map_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,mode='normal'):
    file_path_S = file_path + '/'+dict_node_state[node_start]
    if(not os.path.exists(file_path_S)):
        os.makedirs(file_path_S)
    bfs_edges = list(nx.bfs_edges(flat_tree,node_start)) 
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])           

    dict_tree = {}
    bfs_prev = nx.bfs_predecessors(flat_tree,node_start)
    bfs_next = nx.bfs_successors(flat_tree,node_start)
    for x in bfs_nodes:
        dict_tree[x] = {'prev':"",'next':[]}
        if(x in bfs_prev.keys()):
            dict_tree[x]['prev'] = bfs_prev[x]
        if(x in bfs_next.keys()):
            dict_tree[x]['next'] = bfs_next[x]
            
    if(mode == 'normal'):
        df_rooted_tree['edge'] = ""
        df_rooted_tree['lam_ordered'] = ""
        df_rooted_tree['pos_ori'] = ""    
        dict_len = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                dict_len[x] = dict_branches[x]['len']
            else:
                dict_len[x] = dict_branches[(x[1],x[0])]['len']
    if(mode == 'contracted'):
        df_rooted_tree['edge'] = ""
        df_rooted_tree['lam_orderd_contracted'] = ""
        df_rooted_tree['pos_contracted'] = ""    
        dict_len = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                dict_len[x] = dict_branches[x]['len_ori']
            else:
                dict_len[x] = dict_branches[(x[1],x[0])]['len_ori']
    ##shift distance of each branch
    dict_shift_dist = dict()
    #depth first search
    dfs_nodes = list(nx.dfs_preorder_nodes(flat_tree,node_start))
    leaves=[n for n,d in flat_tree.degree().items() if d==1]
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

    dict_shift_dist = {x: dict_shift_dist[x]*np.percentile(df_rooted_tree['dist'],90)*5 for x in dict_shift_dist.keys()}

    if(mode == 'normal'):
        dict_path_len = nx.shortest_path_length(flat_tree,source=node_start,weight='len')
    if(mode == 'contracted'):
        dict_path_len = nx.shortest_path_length(flat_tree,source=node_start,weight='len_ori')
    dict_edges_pos = {}
    dict_nodes_pos = {}
    for edge_id in bfs_edges:
        node_pos_st = np.array([dict_path_len[edge_id[0]],dict_shift_dist[edge_id]])
        node_pos_ed = np.array([dict_path_len[edge_id[1]],dict_shift_dist[edge_id]])    
        if(edge_id in dict_branches.keys()):
            id_cells = np.where(df_rooted_tree['branch_id']==edge_id)[0]
            if(mode == 'normal'):
                cells_pos_x = node_pos_st[0] + df_rooted_tree.iloc[id_cells,]['lam']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist']
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] =  df_rooted_tree.iloc[id_cells,]['lam']
            if(mode == 'contracted'):
                cells_pos_x = node_pos_st[0] + df_rooted_tree.iloc[id_cells,]['lam_contracted']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist_contracted']
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_orderd_contracted'] =  df_rooted_tree.iloc[id_cells,]['lam_contracted']         
        else:
            id_cells = np.where(df_rooted_tree['branch_id']==(edge_id[1],edge_id[0]))[0]    
            if(mode == 'normal'):
                cells_pos_x = node_pos_st[0] + dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist']   
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_ordered'] = dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam']   
            if(mode == 'contracted'):
                cells_pos_x = node_pos_st[0] + dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam_contracted']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist_contracted'] 
                df_rooted_tree.loc[df_rooted_tree.index[id_cells],'lam_orderd_contracted'] =  dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam_contracted']
        cells_pos = np.array((cells_pos_x,cells_pos_y)).T
        df_rooted_tree.loc[df_rooted_tree.index[id_cells],'edge'] =  [edge_id]
        if(mode == 'normal'):
            df_rooted_tree.loc[df_rooted_tree.index[id_cells],'pos_ori'] =[cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
        if(mode == 'contracted'):
            df_rooted_tree.loc[df_rooted_tree.index[id_cells],'pos_contracted'] =[cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
        dict_edges_pos[edge_id] = np.array([node_pos_st,node_pos_ed])    
        if(edge_id[0] not in dict_nodes_pos.keys()):
            dict_nodes_pos[edge_id[0]] = node_pos_st
        if(edge_id[1] not in dict_nodes_pos.keys()):
            dict_nodes_pos[edge_id[1]] = node_pos_ed
        
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
    legend_labels = []
    legend_handles = []
    for edge_id in bfs_edges:  
        edge_pos = dict_edges_pos[edge_id]
        if(edge_id in dict_branches.keys()):
            edge_color = curves_color[edge_id]
        else:
            edge_color = curves_color[(edge_id[1],edge_id[0])]
        ax1.plot(dict_edges_pos[edge_id][:,0],dict_edges_pos[edge_id][:,1],c=edge_color,alpha=1,lw=5,zorder=None)
        prev_node = dict_tree[edge_id[0]]['prev']
        if(prev_node!=''):
            link_edge_pos = np.array([dict_edges_pos[(prev_node,edge_id[0])][1,],dict_edges_pos[edge_id][0,]])
            ax1.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
            edge_pos = np.vstack((link_edge_pos,edge_pos))
        if(flag_web):
            pd.DataFrame(edge_pos).to_csv(file_path_S + '/subway_coord_line_'+dict_node_state[edge_id[0]] + '_' + dict_node_state[edge_id[1]]+'.csv',sep='\t',index=False)
    if(flat_tree.degree(node_start)>1):
        multi_nodes = dict_tree[node_start]['next']
        multi_edges = [(node_start,x) for x in multi_nodes]
        max_y_pos = max([dict_edges_pos[x][0,1] for x in multi_edges])
        min_y_pos = min([dict_edges_pos[x][0,1] for x in multi_edges])
        median_y_pos = np.median([dict_edges_pos[x][0,1] for x in multi_edges])
        x_pos = dict_edges_pos[multi_edges[0]][0,0]
        link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
        ax1.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
        dict_nodes_pos[node_start] = np.array([x_pos,median_y_pos])
        if(flag_web):
            pd.DataFrame(link_edge_pos).to_csv(file_path_S + '/subway_coord_line_'+dict_node_state[node_start] + '_' + dict_node_state[node_start]+'.csv',sep='\t',index=False)


    for x in dict_node_state.keys():
        ax1.text(dict_nodes_pos[x][0],dict_nodes_pos[x][1],dict_node_state[x],color='black',fontsize = 15,horizontalalignment='center',verticalalignment='center',zorder=20)

    df_color = pd.DataFrame(columns=['cell_label','color'])
    df_color['cell_label'] = df_rooted_tree['CELL_LABEL']   
    list_patches = []
    for x in input_cell_label_uni_color.keys():
        id_cells = np.where(df_rooted_tree['CELL_LABEL']==x)[0]
        df_color.loc[df_color.index[id_cells],'color'] = input_cell_label_uni_color[x]
        list_patches.append(Patches.Patch(color = input_cell_label_uni_color[x],label=x))  

    if(mode == 'normal'):
        X_plot = np.array(df_rooted_tree['pos_ori'].sample(frac=1,random_state=100).tolist())
        X_color = df_color.sample(frac=1,random_state=100)['color']
        ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8,zorder=10)  
        ax1.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)  
        if(flag_web):
            pd.DataFrame(X_plot,X_color).to_csv(file_path_S + '/subway_coord_cells.csv',sep='\t')    
        plt.savefig(file_path_S + '/subway_map.pdf',pad_inches=1,bbox_inches='tight')
        plt.close(fig)
        
    if(mode == 'contracted'):
        X_plot = np.array(df_rooted_tree['pos_contracted'].sample(frac=1,random_state=100).tolist())
        X_color = df_color.sample(frac=1,random_state=100)['color']
        ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=X_color,s=50,linewidth=0,alpha=0.8,zorder=10)  
        ax1.legend(handles = list_patches,loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=int(ceil(len(input_cell_label_uni)/2.0)), fancybox=True, shadow=True,markerscale=2.5)  
        plt.savefig(file_path_S + '/contracted_subway_map.pdf',pad_inches=1,bbox_inches='tight')
        plt.close(fig)  


def find_root_to_leaf_paths(flat_tree, root): 
    list_paths = list()
    for x in flat_tree.nodes():
        if((x!=root)&(flat_tree.degree(x)==1)):
            path = list(nx.all_simple_paths(flat_tree,root,x))[0]
            list_edges = list()
            for ft,sd in zip(path,path[1:]):
                list_edges.append((ft,sd))
            list_paths.append(list_edges)
    return list_paths

def find_longest_path(list_paths,len_ori):
    list_lengths = list()
    for x in list_paths:
        list_lengths.append(sum([len_ori[x_i] for x_i in x]))
    return max(list_lengths)


def Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web,mode='normal'):
    file_path_S = file_path + '/'+dict_node_state[node_start]
    if(not os.path.exists(file_path_S)):
        os.makedirs(file_path_S)
    bfs_edges = list(nx.bfs_edges(flat_tree,node_start)) 
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])           
    if(mode == 'normal'):
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_ordered']].copy()
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len']
    if(mode == 'contracted'):
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_orderd_contracted']].copy()
        df_stream.rename(columns={'lam_orderd_contracted': 'lam_ordered'},inplace=True)
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len_ori']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len_ori']

    dict_tree = {}
    bfs_prev = nx.bfs_predecessors(flat_tree,node_start)
    bfs_next = nx.bfs_successors(flat_tree,node_start)
    for x in bfs_nodes:
        dict_tree[x] = {'prev':"",'next':[]}
        if(x in bfs_prev.keys()):
            dict_tree[x]['prev'] = bfs_prev[x]
        if(x in bfs_next.keys()):
            dict_tree[x]['next'] = bfs_next[x]

    ##shift distance of each branch
    dict_shift_dist = dict()
    #depth first search
    dfs_nodes = list(nx.dfs_preorder_nodes(flat_tree,node_start))
    leaves=[n for n,d in flat_tree.degree().items() if d==1]
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
    size_w = max_path_len/10.0
    if(size_w>min(len_ori.values())/2.0):
        size_w = min(len_ori.values())/2.0

    step_w = size_w/2 #step of sliding window (the divisor should be even)    
    
    max_width = (max_path_len/2.5)/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
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
            nb_nodes = flat_tree.neighbors(edge_i[0])
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

        max_binnum = around((len_ori[edge_i]/4.0-size_w)/step_w) # the maximal number of merging bins
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
                    df_bins.loc['center',"win"+str(total_bins+i_win)] = mean(mat_w[i_win,:])
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
                        df_bins.loc['center',"win"+str(total_bins)] = mean(bd_bins)
                    total_bins = total_bins + 1
                    id_stack = []

        if(degree_end>1):
            #matrix of windows appearing on multiple edges
            mat_w_common = np.vstack([np.arange(len_ori[edge_i]-size_w+step_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w),\
                                      np.arange(step_w,size_w+(len_ori[edge_i]/10**6),step_w)]).T
            #neighbor nodes
            nb_nodes = flat_tree.neighbors(edge_i[1])
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
        if(flatnonzero(df_bins.loc[cellname,]).size==0):
            print('Cell '+cellname+' does not exist')
            break
        else:
            id_nonzero.append(flatnonzero(df_bins.loc[cellname,])[0])
    cell_list_sorted = cell_list[argsort(id_nonzero)].tolist()
    #original count
    df_bins_ori = df_bins.reindex(cell_list_sorted+['boundary','center','edge'])
    df_bins_cumsum = df_bins_ori.copy()
    df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

    if(flag_stream_log_view):
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
            nb_nodes = flat_tree.neighbors(node_i)
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
                    dict_forest[cellname][node_i]['div'] = cumsum(np.repeat(1.0/len(next_nodes),len(next_nodes))).tolist()
                else:
                    dict_forest[cellname][node_i]['div'] = (cumsum(pro_next_edges)/sum(pro_next_edges)).tolist()

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
        mean_shift_dist = mean([dict_shift_dist[(node_start,x)] \
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

    #find all paths
    def find_paths(dict_tree,bfs_nodes):
        dict_paths_top = dict()
        dict_paths_base = dict()
        for node_i in bfs_nodes:
            prev_node = dict_tree[node_i]['prev']
            next_nodes = dict_tree[node_i]['next']
            if(prev_node == '') or (len(next_nodes)>1):
                if(prev_node == ''):
                    cur_node_top = node_i
                    cur_node_base = node_i
                    stack_top = [cur_node_top]
                    stack_base = [cur_node_base]
                    while(len(dict_tree[cur_node_top]['next'])>0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i,next_nodes[0])] = stack_top
                    while(len(dict_tree[cur_node_base]['next'])>0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i,next_nodes[-1])] = stack_base
                for i_mid in range(len(next_nodes)-1):
                    cur_node_base = next_nodes[i_mid]
                    cur_node_top = next_nodes[i_mid+1]
                    stack_base = [node_i,cur_node_base]
                    stack_top = [node_i,cur_node_top]
                    while(len(dict_tree[cur_node_base]['next'])>0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i,next_nodes[i_mid])] = stack_base
                    while(len(dict_tree[cur_node_top]['next'])>0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i,next_nodes[i_mid+1])] = stack_top
        return dict_paths_top,dict_paths_base

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


    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
    patches = []
    legend_labels = []
    for cellname in cell_list_sorted:
        legend_labels.append(cellname)
        verts_cell = verts[cellname]
        polygon = Polygon(verts_cell,closed=True,color=input_cell_label_uni_color[cellname],alpha=0.8,lw=0)
        ax.add_patch(polygon)

    if(input_cell_label_uni[0] == 'unknown'):
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                linear_top = dict_smooth_linear[cellname]['top'][edge_i].loc['y']
                linear_base = dict_smooth_linear[cellname]['base'][edge_i].loc['y']
                if(sum(np.abs(linear_base - linear_top))>1e-10):
                    ax.plot(dict_smooth_new[cellname]['top'][edge_i].loc['x'],dict_smooth_new[cellname]['top'][edge_i].loc['y'],\
                            c = 'grey',ls = 'solid',lw=2)
                    ax.plot(dict_smooth_new[cellname]['base'][edge_i].loc['x'],dict_smooth_new[cellname]['base'][edge_i].loc['y'],\
                            c = 'grey',ls = 'solid',lw=2)
    plt.xticks(fontsize=35)
    # plt.xticks([])
    plt.yticks([])
    plt.xlabel('Pseudotime',fontsize=40)
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
    if(mode == 'normal'):
        if(flag_web):
            plt.savefig(file_path_S + '/stream_plot.png',pad_inches=1,bbox_inches='tight',dpi=120)
        else:
            plt.legend(legend_labels,prop={'size':45},loc='center', bbox_to_anchor=(0.5, 1.20),ncol=int(ceil(len(legend_labels)/2.0)), \
                       fancybox=True, shadow=True)
            plt.savefig(file_path_S + '/stream_plot.pdf',pad_inches=1,bbox_inches='tight',dpi=120)
    if(mode == 'contracted'):
        plt.legend(legend_labels,prop={'size':45},loc='center', bbox_to_anchor=(0.5, 1.20),ncol=int(ceil(len(legend_labels)/2.0)), \
                   fancybox=True, shadow=True)
        plt.savefig(file_path_S + '/contracted_stream_plot.pdf',pad_inches=1,bbox_inches='tight',dpi=120)        
    plt.close(fig)


def Output_Ordered_Info(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,file_path):
    bfs_edges = list(nx.bfs_edges(flat_tree,node_start)) 
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])  
    dict_tree = {}
    bfs_prev = nx.bfs_predecessors(flat_tree,node_start)
    bfs_next = nx.bfs_successors(flat_tree,node_start)
    for x in bfs_nodes:
        dict_tree[x] = {'prev':"",'next':[]}
        if(x in bfs_prev.keys()):
            dict_tree[x]['prev'] = bfs_prev[x]
        if(x in bfs_next.keys()):
            dict_tree[x]['next'] = bfs_next[x]
    len_ori = {}
    for x in bfs_edges:
        if(x in dict_branches.keys()):
            len_ori[x] = dict_branches[x]['len']
        else:
            len_ori[x] = dict_branches[(x[1],x[0])]['len']
    len_prev = dict()
    for edge_i in bfs_edges:
        cur_node = edge_i[0]
        prev_node = dict_tree[edge_i[0]]['prev']
        len_prev[edge_i] = 0
        while(prev_node !=''):
            len_prev[edge_i] = len_prev[edge_i] + len_ori[(prev_node,cur_node)]
            cur_node = prev_node
            prev_node = dict_tree[cur_node]['prev']
    df_out_ordered_info = df_rooted_tree[['CELL_ID','edge','lam_ordered','dist']].copy()
    df_out_ordered_info['Branch'] = ''
    df_out_ordered_info['pseudo_time'] = ''
    for edge_i in bfs_edges:
        index_cells = df_out_ordered_info[df_out_ordered_info['edge'] == edge_i].index
        df_out_ordered_info.loc[index_cells,'Branch'] = dict_node_state[edge_i[0]]+'_'+ dict_node_state[edge_i[1]]
        df_out_ordered_info.loc[index_cells,'pseudo_time'] = df_out_ordered_info.loc[index_cells,'lam_ordered'] + len_prev[edge_i]
    file_path_S = file_path + '/'+dict_node_state[node_start]
    out_ordered_info = df_out_ordered_info[['CELL_ID','Branch','pseudo_time','dist']].copy()
    out_ordered_info.sort_values(by=["pseudo_time"], ascending=[True],inplace=True)
    out_ordered_info.to_csv(file_path_S+'/'+dict_node_state[node_start]+'_pseudotime_info.tsv',sep = '\t',index = True)

def Subway_Map_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_atac,file_path,flag_web,mode = 'normal'):
    file_path_S = file_path + '/'+dict_node_state[node_start]
    if(not os.path.exists(file_path_S)):
        os.makedirs(file_path_S)
    bfs_edges = list(nx.bfs_edges(flat_tree,node_start)) 
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])           

    dict_tree = {}
    bfs_prev = nx.bfs_predecessors(flat_tree,node_start)
    bfs_next = nx.bfs_successors(flat_tree,node_start)
    for x in bfs_nodes:
        dict_tree[x] = {'prev':"",'next':[]}
        if(x in bfs_prev.keys()):
            dict_tree[x]['prev'] = bfs_prev[x]
        if(x in bfs_next.keys()):
            dict_tree[x]['next'] = bfs_next[x]

    if(mode == 'normal'):
        dict_len = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                dict_len[x] = dict_branches[x]['len']
            else:
                dict_len[x] = dict_branches[(x[1],x[0])]['len']
    if(mode == 'contracted'):   
        dict_len = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                dict_len[x] = dict_branches[x]['len_ori']
            else:
                dict_len[x] = dict_branches[(x[1],x[0])]['len_ori']
    ##shift distance of each branch
    dict_shift_dist = dict()
    #depth first search
    dfs_nodes = list(nx.dfs_preorder_nodes(flat_tree,node_start))
    leaves=[n for n,d in flat_tree.degree().items() if d==1]
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

    dict_shift_dist = {x: dict_shift_dist[x]*np.percentile(df_rooted_tree['dist'],90)*5 for x in dict_shift_dist.keys()}

    if(mode == 'normal'):
        dict_path_len = nx.shortest_path_length(flat_tree,source=node_start,weight='len')
    if(mode == 'contracted'):
        dict_path_len = nx.shortest_path_length(flat_tree,source=node_start,weight='len_ori')
    dict_edges_pos = {}
    dict_nodes_pos = {}
    for edge_id in bfs_edges:
        node_pos_st = np.array([dict_path_len[edge_id[0]],dict_shift_dist[edge_id]])
        node_pos_ed = np.array([dict_path_len[edge_id[1]],dict_shift_dist[edge_id]])    
        if(edge_id in dict_branches.keys()):
            id_cells = np.where(df_rooted_tree['branch_id']==edge_id)[0]
            if(mode == 'normal'):
                cells_pos_x = node_pos_st[0] + df_rooted_tree.iloc[id_cells,]['lam']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist']
            if(mode == 'contracted'):
                cells_pos_x = node_pos_st[0] + df_rooted_tree.iloc[id_cells,]['lam_contracted']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist_contracted']             
        else:
            id_cells = np.where(df_rooted_tree['branch_id']==(edge_id[1],edge_id[0]))[0]    
            if(mode == 'normal'):
                cells_pos_x = node_pos_st[0] + dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist']        
            if(mode == 'contracted'):
                cells_pos_x = node_pos_st[0] + dict_len[edge_id] - df_rooted_tree.iloc[id_cells,]['lam_contracted']
                np.random.seed(100)
                cells_pos_y = node_pos_st[1] + np.random.choice([1,-1],size=id_cells.shape[0])*df_rooted_tree.iloc[id_cells,]['dist_contracted'] 
        cells_pos = np.array((cells_pos_x,cells_pos_y)).T
        dict_edges_pos[edge_id] = np.array([node_pos_st,node_pos_ed])    
        if(edge_id[0] not in dict_nodes_pos.keys()):
            dict_nodes_pos[edge_id[0]] = node_pos_st
        if(edge_id[1] not in dict_nodes_pos.keys()):
            dict_nodes_pos[edge_id[1]] = node_pos_ed

    for idx in range(len(gene_list)):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1, adjustable='box',aspect=1)       
        for edge_id in bfs_edges:  
            ax.plot(dict_edges_pos[edge_id][:,0],dict_edges_pos[edge_id][:,1],c='gray',alpha=0.8,lw=5,zorder=None)
            prev_node = dict_tree[edge_id[0]]['prev']
            if(prev_node!=''):
                link_edge_pos = np.array([dict_edges_pos[(prev_node,edge_id[0])][1,],dict_edges_pos[edge_id][0,]])
                ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5,zorder=None)
        if(flat_tree.degree(node_start)>1):
            multi_nodes = dict_tree[node_start]['next']
            multi_edges = [(node_start,x) for x in multi_nodes]
            max_y_pos = max([dict_edges_pos[x][0,1] for x in multi_edges])
            min_y_pos = min([dict_edges_pos[x][0,1] for x in multi_edges])
            median_y_pos = np.median([dict_edges_pos[x][0,1] for x in multi_edges])
            x_pos = dict_edges_pos[multi_edges[0]][0,0]
            link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
            ax.plot(link_edge_pos[:,0],link_edge_pos[:,1],c='gray',alpha=0.5,lw=5)
            dict_nodes_pos[node_start] = np.array([x_pos,median_y_pos])

        for x in dict_node_state.keys():
            ax.text(dict_nodes_pos[x][0],dict_nodes_pos[x][1],dict_node_state[x],
                     color='black',fontsize = 15,horizontalalignment='center',verticalalignment='center',zorder=20)        

        if(mode == 'normal'):
            df_subway_gene = pd.DataFrame(df_rooted_tree['pos_ori'].tolist(),index=df_rooted_tree['CELL_ID'].values,columns=['D'+str(x) for x in range(2)])
            pos = np.array(df_rooted_tree['pos_ori'].tolist())
        if(mode == 'contracted'):
            df_subway_gene = pd.DataFrame(df_rooted_tree['pos_contracted'].tolist(),index=df_rooted_tree['CELL_ID'].values,columns=['D'+str(x) for x in range(2)])
            pos = np.array(df_rooted_tree['pos_contracted'].tolist())
        gene_values = df_sc[gene_list[idx]].copy()
        df_subway_gene['Ori_Expr'] = gene_values.values
        cm = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))
        if(not flag_atac):
            max_gene_values = np.percentile(gene_values[gene_values>0],90)
            gene_values[gene_values>max_gene_values] = max_gene_values
            df_subway_gene['Scaled_Expr'] = ((gene_values/max_gene_values)**2).values
            # sizes = 50*((gene_values/max_gene_values)**2)
            X_plot = pd.DataFrame(pos).sample(frac=1,random_state=100)
            gene_values = gene_values.sample(frac=1,random_state=100)  
            if(flag_web):    
                X_plot_for_web = deepcopy(X_plot)
                X_plot_for_web[gene_list[idx]] = gene_values
                X_plot_for_web.to_csv(file_path_S + '/subway_coord_' + slugify(gene_list[idx]) + '.csv',sep='\t')    
                plt.close(fig)
            else:
                sc = ax.scatter(X_plot.iloc[:,0],X_plot.iloc[:,1], c=gene_values, vmin=0, vmax=max_gene_values, s=50, cmap=cm, linewidths=0,alpha=0.5,zorder=10)
                cbar=plt.colorbar(sc)
                cbar.ax.tick_params(labelsize=20)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.set_alpha(1)
                cbar.draw_all()
                ax.set_title(gene_list[idx],size=15)
                if(mode == 'normal'):
                    plt.savefig(file_path_S + '/subway_map_' + slugify(gene_list[idx]) + '.pdf',pad_inches=1,bbox_inches='tight')
                    plt.close(fig)
                if(mode == 'contracted'):
                    plt.savefig(file_path_S + '/contracted_subway_map_' + slugify(gene_list[idx]) + '.pdf',pad_inches=1,bbox_inches='tight')
                    plt.close(fig)     

        else:
            min_gene_values = np.percentile(gene_values[gene_values<0],10)
            max_gene_values = np.percentile(gene_values[gene_values>0],90)
            gene_values[gene_values<min_gene_values] = min_gene_values
            gene_values[gene_values>max_gene_values] = max_gene_values
            df_subway_gene['Scaled_Expr'] = ((gene_values/max_gene_values)**2).values
            # sizes = 50*((gene_values/max_gene_values)**2)
            v_limit = max(abs(min_gene_values),max_gene_values)
            X_plot = pd.DataFrame(pos).sample(frac=1,random_state=100)
            gene_values = gene_values.sample(frac=1,random_state=100)    
            if(flag_web): 
                X_plot_for_web = deepcopy(X_plot)
                X_plot_for_web[gene_list[idx]] = gene_values
                X_plot_for_web.to_csv(file_path_S + '/subway_coord_' + slugify(gene_list[idx]) + '.csv',sep='\t')
                plt.close(fig)
            else:                                       
                sc = ax.scatter(X_plot.iloc[:,0],X_plot.iloc[:,1], c=gene_values, vmin=-v_limit, vmax=max_gene_values, s=50, cmap=cm, linewidths=0,alpha=0.5,zorder=10)
                cbar=plt.colorbar(sc)
                cbar.ax.tick_params(labelsize=20)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.set_alpha(1)
                cbar.draw_all()
                ax.set_title(gene_list[idx],size=15)
                if(mode == 'normal'):
                    plt.savefig(file_path_S + '/subway_map_' + slugify(gene_list[idx]) + '.pdf',pad_inches=1,bbox_inches='tight')
                    plt.close(fig)
                if(mode == 'contracted'):
                    plt.savefig(file_path_S + '/contracted_subway_map_' + slugify(gene_list[idx]) + '.pdf',pad_inches=1,bbox_inches='tight')
                    plt.close(fig)                

            

def Stream_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_stream_log_view,flag_atac,file_path,flag_web,mode='normal'):
    file_path_S = file_path + '/'+dict_node_state[node_start]
    if(not os.path.exists(file_path_S)):
        os.makedirs(file_path_S)
    bfs_edges = list(nx.bfs_edges(flat_tree,node_start)) 
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])           
    if(mode == 'normal'):
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_ordered']].copy()
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len']
    if(mode == 'contracted'):
        df_stream = df_rooted_tree[['CELL_LABEL','edge','lam_orderd_contracted']].copy()
        df_stream.rename(columns={'lam_orderd_contracted': 'lam_ordered'},inplace=True)
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len_ori']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len_ori']
    df_stream.CELL_LABEL = 'unknown'
    df_stream[gene_list] = df_sc[gene_list]

    dict_tree = {}
    bfs_prev = nx.bfs_predecessors(flat_tree,node_start)
    bfs_next = nx.bfs_successors(flat_tree,node_start)
    for x in bfs_nodes:
        dict_tree[x] = {'prev':"",'next':[]}
        if(x in bfs_prev.keys()):
            dict_tree[x]['prev'] = bfs_prev[x]
        if(x in bfs_next.keys()):
            dict_tree[x]['next'] = bfs_next[x]

    ##shift distance of each branch
    dict_shift_dist = dict()
    #depth first search
    dfs_nodes = list(nx.dfs_preorder_nodes(flat_tree,node_start))
    leaves=[n for n,d in flat_tree.degree().items() if d==1]
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
    dict_genes = {gene: pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique())) for gene in gene_list}
    dict_merge_num = {gene:[] for gene in gene_list} #number of merged sliding windows
    list_paths = find_root_to_leaf_paths(flat_tree, node_start)
    max_path_len = find_longest_path(list_paths,len_ori)
    size_w = max_path_len/10.0
    if(size_w>min(len_ori.values())/2.0):
        size_w = min(len_ori.values())/2.0

    step_w = size_w/2 #step of sliding window (the divisor should be even)    
    
    max_width = (max_path_len/2.5)/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
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
            nb_nodes = flat_tree.neighbors(edge_i[0])
            index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
            nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
            #matrix of windows appearing on multiple edges
            total_bins = df_bins.shape[1] # current total number of bins
            for i_win in range(mat_w_common.shape[0]):
                df_bins["win"+str(total_bins+i_win)] = ""
                df_bins.loc[df_bins.index[:-3],"win"+str(total_bins+i_win)] = 0
                df_bins.loc['edge',"win"+str(total_bins+i_win)] = [(node_start,node_start)]
                dict_df_genes_common = dict()
                for gene in gene_list:
                    dict_df_genes_common[gene] = list()
                for j in range(degree_st):
                    df_edge_j = dict_edge_filter[(edge_i[0],nb_nodes[j])]
                    cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                    df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                    df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                    for gene in gene_list:
                        dict_df_genes_common[gene].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
    #                     gene_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
    #                                                             df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[gene].mean()
    #                     dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] = \
    #                     dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] + gene_values_common2
                    df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[0],nb_nodes[j]))
                for gene in gene_list:
                    gene_values_common = pd.concat(dict_df_genes_common[gene]).groupby(['CELL_LABEL'])[gene].mean()
                    dict_genes[gene].loc[gene_values_common.index,"win"+str(total_bins+i_win)] = gene_values_common
                df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                if(i_win == 0):
                    df_bins.loc['center',"win"+str(total_bins+i_win)] = 0
                else:
                    df_bins.loc['center',"win"+str(total_bins+i_win)] = size_w/2

        max_binnum = around((len_ori[edge_i]/4.0-size_w)/step_w) # the maximal number of merging bins
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
                for gene in gene_list:
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
                    df_bins.loc['center',"win"+str(total_bins+i_win)] = mean(mat_w[i_win,:])
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
                    for gene in gene_list:
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
                        df_bins.loc['center',"win"+str(total_bins)] = mean(bd_bins)
                    total_bins = total_bins + 1
                    id_stack = []

        if(degree_end>1):
            #matrix of windows appearing on multiple edges
            mat_w_common = np.vstack([np.arange(len_ori[edge_i]-size_w+step_w,len_ori[edge_i]+(len_ori[edge_i]/10**6),step_w),\
                                      np.arange(step_w,size_w+(len_ori[edge_i]/10**6),step_w)]).T
            #neighbor nodes
            nb_nodes = flat_tree.neighbors(edge_i[1])
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
                    for gene in gene_list:
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
                        for gene in gene_list:
                            dict_df_genes_common[gene].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
    #                         gene_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
    #                                                                 df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[gene].mean()
    #                         dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] = \
    #                         dict_genes[gene].ix[gene_values_common2.index,"win"+str(total_bins+i_win)] + gene_values_common2
                        if abs(((sum(mat_w_common[i_win,:])+len_ori[edge_i])/2)-(len_ori[edge_i]+size_w/2.0))< step_w/100.0:
                            df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[1],nb_nodes[j]))
                    for gene in gene_list:
                        gene_values_common = pd.concat(dict_df_genes_common[gene]).groupby(['CELL_LABEL'])[gene].mean()
                        dict_genes[gene].loc[gene_values_common.index,"win"+str(total_bins+i_win)] = gene_values_common
                    df_bins.loc['boundary',"win"+str(total_bins+i_win)] = mat_w_common[i_win,:]
                    df_bins.loc['center',"win"+str(total_bins+i_win)] = (sum(mat_w_common[i_win,:])+len_ori[edge_i])/2


    #order cell names by the index of first non-zero
    cell_list = df_bins.index[:-3]
    id_nonzero = []
    for i_cn,cellname in enumerate(cell_list):
        if(flatnonzero(df_bins.loc[cellname,]).size==0):
            print('Cell '+cellname+' does not exist')
            break
        else:
            id_nonzero.append(flatnonzero(df_bins.loc[cellname,])[0])
    cell_list_sorted = cell_list[argsort(id_nonzero)].tolist()
    #original count
    df_bins_ori = df_bins.reindex(cell_list_sorted+['boundary','center','edge'])
    df_bins_cumsum = df_bins_ori.copy()
    df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

    if(flag_stream_log_view):
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
    if(not flag_atac):
        for gene in gene_list:
            gene_values = dict_genes[gene].iloc[0,].values
            max_gene_values = np.percentile(gene_values[gene_values>0],90)
            dict_genes_norm[gene] = dict_genes[gene].reindex(cell_list_sorted)
            dict_genes_norm[gene][dict_genes_norm[gene]>max_gene_values] = max_gene_values
    else:
        for gene in gene_list:
            gene_values = dict_genes[gene].iloc[0,].values
            min_gene_values = np.percentile(gene_values[gene_values<0],10)
            max_gene_values = np.percentile(gene_values[gene_values>0],90)
            dict_genes_norm[gene] = dict_genes[gene].reindex(cell_list_sorted)
            dict_genes_norm[gene][dict_genes_norm[gene]<min_gene_values] = min_gene_values
            dict_genes_norm[gene][dict_genes_norm[gene]>max_gene_values] = max_gene_values        

    dict_forest = {cellname: {nodename:{'prev':"",'next':"",'div':""} for nodename in bfs_nodes}\
                   for cellname in df_edge_cellnum.index}
    for cellname in cell_list_sorted:
        for node_i in bfs_nodes:
            nb_nodes = flat_tree.neighbors(node_i)
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
                    dict_forest[cellname][node_i]['div'] = cumsum(np.repeat(1.0/len(next_nodes),len(next_nodes))).tolist()
                else:
                    dict_forest[cellname][node_i]['div'] = (cumsum(pro_next_edges)/sum(pro_next_edges)).tolist()

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
        mean_shift_dist = mean([dict_shift_dist[(node_start,x)] \
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
    #                 ax.plot(px_top_prime_st,py_top_prime_st,'go',alpha=0.2)
    #                 ax.plot(px_base_prime_st,py_base_prime_st,'ro',alpha=0.2)

    #find all paths
    def find_paths(dict_tree,bfs_nodes):
        dict_paths_top = dict()
        dict_paths_base = dict()
        for node_i in bfs_nodes:
            prev_node = dict_tree[node_i]['prev']
            next_nodes = dict_tree[node_i]['next']
            if(prev_node == '') or (len(next_nodes)>1):
                if(prev_node == ''):
                    cur_node_top = node_i
                    cur_node_base = node_i
                    stack_top = [cur_node_top]
                    stack_base = [cur_node_base]
                    while(len(dict_tree[cur_node_top]['next'])>0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i,next_nodes[0])] = stack_top
                    while(len(dict_tree[cur_node_base]['next'])>0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i,next_nodes[-1])] = stack_base
                for i_mid in range(len(next_nodes)-1):
                    cur_node_base = next_nodes[i_mid]
                    cur_node_top = next_nodes[i_mid+1]
                    stack_base = [node_i,cur_node_base]
                    stack_top = [node_i,cur_node_top]
                    while(len(dict_tree[cur_node_base]['next'])>0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i,next_nodes[i_mid])] = stack_base
                    while(len(dict_tree[cur_node_top]['next'])>0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i,next_nodes[i_mid+1])] = stack_top
        return dict_paths_top,dict_paths_base

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

    def fill_im_array(dict_im_array,id_wins,edge_i,id_wins_prev,prev_edge):
        pad_ratio = 0.008
        xmin_edge = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].min()
        xmax_edge = df_base_x.loc[cellname,map(lambda x: 'win' + str(x), id_wins)].max()
        id_st_x = int(floor(((xmin_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
        id_ed_x =  int(floor(((xmax_edge - xmin)/(xmax - xmin))*(im_ncol-1)))
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

    for Gene_Name in gene_list:
        #calculate gradient image
        #image array
        im_nrow = 100
        im_ncol = 400
        xmin = dict_extent['xmin']
        xmax = dict_extent['xmax']
        ymin = dict_extent['ymin'] - (dict_extent['ymax'] - dict_extent['ymin'])*0.1
        ymax = dict_extent['ymax'] + (dict_extent['ymax'] - dict_extent['ymin'])*0.1
        dict_im_array = {cellname: np.zeros((im_nrow,im_ncol)) for cellname in cell_list_sorted}
        df_bins_gene = dict_genes_norm[Gene_Name]
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                id_wins_all = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==edge_i]
                prev_edge = ''
                id_wins_prev = []
                if(flat_tree.degree(node_start)>1):
                    if(edge_i == bfs_edges[0]):
                        id_wins = [0,1]
                        dict_im_array = fill_im_array(dict_im_array,id_wins,edge_i,id_wins_prev,prev_edge)
                    id_wins = id_wins_all
                    if(edge_i[0] == node_start):
                        prev_edge = (node_start,node_start)
                        id_wins_prev = [0,1]
                    else:
                        prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                        id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                    dict_im_array = fill_im_array(dict_im_array,id_wins,edge_i,id_wins_prev,prev_edge)
                else:
                    id_wins = id_wins_all
                    if(edge_i[0]!=node_start):
                        prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                        id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                    dict_im_array = fill_im_array(dict_im_array,id_wins,edge_i,id_wins_prev,prev_edge)

        #clip parts according to determined polygon
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1)
        ax.set_title(Gene_Name,size=20)
        patches = []

        dict_imshow = dict()
        cmap1 = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 256))
        # cmap1 = mpl.colors.ListedColormap(sns.diverging_palette(250, 10,s=90,l=35, n=256))
        for cellname in cell_list_sorted:
            if(not flag_atac):
                im = ax.imshow(dict_im_array[cellname], cmap=cmap1,interpolation='bicubic',\
                               extent=[xmin,xmax,ymin,ymax],vmin=0,vmax=df_bins_gene.values.max())
            else:
                v_limit = max(abs(df_bins_gene.values.min()),df_bins_gene.values.max())
                im = ax.imshow(dict_im_array[cellname], cmap=cmap1,interpolation='bicubic',\
                               extent=[xmin,xmax,ymin,ymax],vmin=-v_limit,vmax=v_limit)                
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

        plt.xticks(fontsize=20)
        plt.yticks([])
        plt.xlabel('Pseudotime',fontsize=25)

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
                            c = 'grey',ls = 'solid',lw=1)
                    ax.plot(dict_smooth_new[cellname]['base'][edge_i].loc['x'],dict_smooth_new[cellname]['base'][edge_i].loc['y'],\
                            c = 'grey',ls = 'solid',lw=1)

        fig_xmin, fig_xmax = ax.get_xlim()
        fig_ymin, fig_ymax = ax.get_ylim()
        # manual arrowhead width and length
        fig_hw = 1./20.*(fig_ymax-fig_ymin)
        fig_hl = 1./20.*(fig_xmax-fig_xmin)
        ax.arrow(fig_xmin, fig_ymin, fig_xmax-fig_xmin, 0., fc='k', ec='k', lw = 1.0,
                 head_width=fig_hw, head_length=fig_hl, overhang = 0.3,
                 length_includes_head= True, clip_on = False)
        if(flag_web):
            plt.savefig(file_path_S+'/stream_plot_' + slugify(Gene_Name) + '.png',dpi=120)
        else:
            if(mode == 'normal'):
                plt.savefig(file_path_S+'/stream_plot_' + slugify(Gene_Name) + '.pdf',dpi=120)
            if(mode == 'contracted'):
                plt.savefig(file_path_S+'/contracted_stream_plot_' + slugify(Gene_Name) + '.pdf',dpi=120)
        plt.close(fig)


def Output_Info(df_flat_tree,flat_tree,dict_node_state,file_path):
    edges = flat_tree.edges()
    dict_nodes_pos = nx.get_node_attributes(flat_tree,'nodes_pos')
    array_edges = deepcopy(np.array(edges))
    array_edges = array_edges.astype(str)
    df_nodes = pd.DataFrame(columns=['D' + str(x) for x in range(2)])
    for x in dict_node_state.keys():
        np.place(array_edges,array_edges==str(x),dict_node_state[x])
        df_nodes.loc[dict_node_state[x]] = dict_nodes_pos[x]
    pd.DataFrame(array_edges).to_csv(file_path+'/edges.tsv',sep = '\t',index = False,header=False)
    df_nodes.sort_index(inplace=True)
    df_nodes.to_csv(file_path+'/nodes.tsv',sep = '\t',index = True,header=True)

    df_out_info = df_flat_tree[['CELL_ID','branch_id','lam','dist']].copy()
    df_out_info['Branch'] = ''
    for edge_i in np.unique(df_out_info['branch_id']):
        index_cells = df_out_info[df_out_info['branch_id'] == edge_i].index
        df_out_info.loc[index_cells,'Branch'] = dict_node_state[edge_i[0]]+'_'+dict_node_state[edge_i[1]]
    out_info = df_out_info[['CELL_ID','Branch','lam','dist']].copy()
    out_info.sort_values(by=["Branch","lam"], ascending=[True, True],inplace=True)
    out_info.to_csv(file_path+'/cell_info.tsv',sep = '\t',index = True)


def slugify(value): #adapted from the Django project
    value = unicodedata.normalize('NFKD', unicode(value)).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\.\s-]', '-', value).strip())
    value = unicode(re.sub('[-\s]+', '-', value))
    return str(value)


def barycenter_weights_modified(X, Z, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (1, n_dim)
    Z : array-like, shape (n_neighbors, n_dim)
    reg : float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (1, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
#     X = check_array(X, dtype=FLOAT_DTYPES)
#     Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)

    n_samples, n_neighbors = 1, Z.shape[0]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    C = Z - X  # broadcasting
    G = np.dot(C, C.T)
    trace = np.trace(G)
    if trace > 0:
        R = reg * trace
    else:
        R = reg
    G.flat[::Z.shape[0] + 1] += R
    w = solve(G, v, sym_pos=True)
    B = w / np.sum(w)
    return B


def Map_New_Data_To_LLE(new_df_sc_final,sklearn_lle,file_path_precomp): #map new data to known tree structure
    new_data = new_df_sc_final.iloc[:,1:].values
    # ind = sklearn_lle.nbrs_.kneighbors(new_data, n_neighbors=int(sklearn_lle.embedding_.shape[0]),
    #                         return_distance=False)
    # weights = barycenter_weights_modified(new_data, sklearn_lle.nbrs_._fit_X[ind],
    #                              reg=sklearn_lle.reg)
    # new_X = np.empty((new_data.shape[0], sklearn_lle.n_components))
    # for i in range(new_data.shape[0]):
    #     new_X[i] = np.dot(sklearn_lle.embedding_[ind[i]].T, weights[i])
    
    dist_nb = sklearn_lle.nbrs_.kneighbors(new_data, n_neighbors=sklearn_lle.n_neighbors,
                            return_distance=True)[0]
    ind = sklearn_lle.nbrs_.radius_neighbors(new_data, radius = dist_nb.max(),
                            return_distance=False)
    new_X = np.empty((new_data.shape[0], sklearn_lle.n_components))
    for i in range(new_data.shape[0]):
        weights = barycenter_weights_modified(new_data[i], sklearn_lle.nbrs_._fit_X[ind[i]],
                                 reg=sklearn_lle.reg)
        new_X[i] = np.dot(sklearn_lle.embedding_[ind[i]].T, weights)

    # new_X = sklearn_lle.transform(new_data)
    return new_X

def Map_New_Data_To_Tree(new_df_flat_tree,df_cells,dict_branches):
    new_df_flat_tree['X_projected'] = df_cells['pt_proj']
    new_df_flat_tree['node_id'] = df_cells['node_id']
    new_df_flat_tree['lam'] = df_cells['lam']
    new_df_flat_tree['dist'] = df_cells['dist']
    new_df_flat_tree['branch_len'] = df_cells['branch_len']
    new_df_flat_tree['branch_id'] = df_cells['branch_id']
    new_df_flat_tree['branch_len_ori'] = ""
    new_df_flat_tree['lam_contracted'] = ""
    new_df_flat_tree['dist_contracted'] = ""
    min_len_ori = min([dict_branches[x_br]['len_ori'] for x_br in dict_branches.keys()])
    for x_br in dict_branches.keys():
        id_cells = new_df_flat_tree[new_df_flat_tree['branch_id'] == x_br].index
        if(len(id_cells)>0):
            ratio_len = dict_branches[x_br]['len']/float(dict_branches[x_br]['len_ori'])
            new_df_flat_tree.loc[id_cells,'lam_contracted'] = df_cells.loc[id_cells,'lam']/ratio_len
            new_df_flat_tree.loc[id_cells,'branch_len_ori'] = dict_branches[x_br]['len_ori']
            dist_p = np.log2((new_df_flat_tree.loc[id_cells,'dist']+1).tolist())
            new_df_flat_tree.loc[id_cells,'dist_contracted'] = 0.7*min_len_ori*dist_p/max(dist_p)
    new_df_flat_tree = new_df_flat_tree.astype('object')
    return new_df_flat_tree


def Save_To_Pickle(variable,file_name,file_path_precomp):
    f = open(file_path_precomp + '/' + file_name + '.pickle', 'wb')
    cPickle.dump(variable, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def Read_From_Pickle(file_name,file_path_precomp):
    f = open(file_path_precomp+'/'+ file_name +'.pickle', 'rb')
    variable = cPickle.load(f)
    f.close()
    return variable


def counts_to_kmers(counts_file,regions_file,samples_file,k,file_path):
    chromVAR = importr('chromVAR')
    GenomicRanges = importr('GenomicRanges')
    SummarizedExperiment = importr('SummarizedExperiment')
    BSgenome_Hsapiens_UCSC_hg19 = importr('BSgenome.Hsapiens.UCSC.hg19')
    r_Matrix = importr('Matrix')
    BiocParallel = importr('BiocParallel')
    BiocParallel.register(BiocParallel.MulticoreParam(2))
    pandas2ri.activate()
    
    df_regions = pd.read_csv(regions_file,sep='\t',header=None,compression= 'gzip' if regions_file.split('.')[-1]=='gz' else None)
    df_regions = df_regions.iloc[:,:3]
    df_regions.columns = ['seqnames','start','end']
    pandas2ri.activate()
    r_regions_dataframe = pandas2ri.py2ri(df_regions)
    regions = GenomicRanges.makeGRangesFromDataFrame(r_regions_dataframe)
    df_counts = pd.read_csv(counts_file,sep='\t',header=None,names=['i','j','x'],compression= 'gzip' if counts_file.split('.')[-1]=='gz' else None)
    counts = r_Matrix.sparseMatrix(i = df_counts['i'], j = df_counts['j'], x=df_counts['x'])
    df_samples = pd.read_csv(samples_file,sep='\t',header=None,names=['cell_id'],compression= 'gzip' if samples_file.split('.')[-1]=='gz' else None)
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
    df_zscores.to_csv(file_path+'/df_zscores.tsv',sep='\t')
    df_zscores_scaled = preprocessing.scale(df_zscores,axis=1)
    df_zscores_scaled = pd.DataFrame(df_zscores_scaled,index=df_zscores.index,columns=df_zscores.columns)
    df_zscores_scaled.to_csv(file_path+'/df_zscores_scaled.tsv',sep='\t')


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
    parser.add_argument("-f","--feature_genes",dest="feature_genes", default=None,
                        help="specified feature genes ")
    parser.add_argument("-t","--detect_TG_genes",dest="flag_gene_TG_detection", action="store_true",
                        help="detect transition genes automatically")
    parser.add_argument("-d","--detect_DE_genes",dest="flag_gene_DE_detection", action="store_true",
                        help="detect DE genes automatically")
    parser.add_argument("-g","--gene_list",dest="gene_list", default=None,
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
    parser.add_argument("--n_processes",dest = "n_processes",type=int, default=multiprocessing.cpu_count(),
                        help="Specify the number of processes to use. (default, all the available cores)")
    parser.add_argument("--loess_frac",dest = "loess_frac",type=float, default=0.1,
                        help="The fraction of the data used in LOESS regression")
    parser.add_argument("--loess_z_score_cutoff",dest="loess_z_score_cutoff", type = float, default=1.0,
                        help="z-score cutoff in gene selection based on LOESS regression")
    parser.add_argument("--pca_max_PC",dest="pca_max_PC", type=int,default=100,
                        help="allowed maximal principal components in PCA")
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
    parser.add_argument("--SC_n_cluster",dest="n_cluster", type=int, default=20,
                        help="Number of clusters for spectral clustering")
    parser.add_argument("--EPG_n_nodes",dest="EPG_n_nodes", type=int, default=50,
                        help=" Number of nodes for elastic principal graph")
    parser.add_argument("--EPG_n_rep",dest="EPG_n_rep", type=int, default=1,
                        help="Number of replica for constructing elastic principal graph")            
    parser.add_argument("--EPG_lambda",dest="EPG_lambda", type=float, default=0.02,
                        help="lambda parameter used to compute the elastic energy")                                                                     
    parser.add_argument("--EPG_mu",dest="EPG_mu", type=float, default=0.1,
                        help="mu parameter used to compute the elastic energy") 
    parser.add_argument("--EPG_trimmingradius",dest="EPG_trimmingradius", type=float, default=Inf,
                        help="maximal distance of point from a node to affect its embedment") 
    parser.add_argument("--EPG_prob",dest="EPG_prob", type=float, default=1.0,
                        help="probability of including a single point for each computation") 
    parser.add_argument("--EPG_finalenergy",dest="EPG_finalenergy", default='Penalized',
                        help="Indicating the final elastic emergy associated with the configuration. It can be Base or Penalized ")
    parser.add_argument("--EPG_alpha",dest="EPG_alpha", type=float, default=0.02,
                        help="positive numeric, the value of the alpha parameter of the penalized elastic energy") 
    parser.add_argument("--EPG_beta",dest="EPG_beta", type=float, default=0.0,
                        help="positive numeric, the value of the beta parameter of the penalized elastic energy") 
    parser.add_argument("--disable_EPG_collapse",dest="flag_disable_EPG_collapse", action="store_true",
                        help="disable collapsing small branches")
    parser.add_argument("--EPG_collapse_mode",dest="EPG_collapse_mode", default ="PointNumber",
                        help="the mode used to collapse branches. PointNumber,PointNumber_Extrema, PointNumber_Leaves,EdgesNumber or EdgesLength")
    parser.add_argument("--EPG_collapse_par",dest="EPG_collapse_par", type=float, default=5,
                        help="positive numeric, the cotrol paramter used for collapsing small branches")
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
    parser.add_argument("--DE_z_score_cutoff",dest="DE_z_score_cutoff", default=2,
                        help="Differentially Expressed Genes z-score cutoff")
    parser.add_argument("--DE_diff_cutoff",dest="DE_diff_cutoff", default=0.2,
                        help="Differentially Expressed Genes difference cutoff")
    parser.add_argument("--TG_spearman_cutoff",dest="TG_spearman_cutoff", default=0.4,
                        help="Transition Genes Spearman correlation cutoff")
    parser.add_argument("--TG_diff_cutoff",dest="TG_diff_cutoff", default=0.2,
                        help="Transition Genes difference cutoff")
    parser.add_argument("--mds",dest="flag_mds", action="store_true",
                        help="whether to use MDS for visualization (default: No)")    
    parser.add_argument("--stream_log_view",dest="flag_stream_log_view", action="store_true",
                        help="use log2 scale for y axis of stream_plot")
    parser.add_argument("-o","--output_folder",dest="output_folder", default=None,
                        help="Output folder")
    parser.add_argument("--for_web",dest="flag_web", action="store_true",
                        help="Output files for website")

    args = parser.parse_args()
    new_filename = args.new_filename
    flag_stream_log_view = args.flag_stream_log_view
    flag_gene_TG_detection = args.flag_gene_TG_detection
    flag_gene_DE_detection = args.flag_gene_DE_detection
    flag_web = args.flag_web
    flag_first_PC = args.flag_first_PC
    flag_mds = args.flag_mds
    gene_list_filename = args.gene_list
    DE_z_score_cutoff = args.DE_z_score_cutoff
    DE_diff_cutoff = args.DE_diff_cutoff
    TG_spearman_cutoff = args.TG_spearman_cutoff
    TG_diff_cutoff = args.TG_diff_cutoff

    if(new_filename!=None):
        new_label_filename = args.new_label_filename
        new_label_color_filename = args.new_label_color_filename
        output_folder = args.output_folder #work directory

        file_path = os.getcwd() + '/' + output_folder + '/Mapping_Result'
        if(not os.path.exists(file_path)):
            os.makedirs(file_path)
        file_path_precomp = os.getcwd() + '/'+ output_folder +'/Precomputed'

        flag_log2 = Read_From_Pickle('flag_log2',file_path_precomp)
        flag_norm = Read_From_Pickle('flag_norm',file_path_precomp)
        s_method = Read_From_Pickle('s_method',file_path_precomp)
        feature_genes_filename = Read_From_Pickle('feature_genes_filename',file_path_precomp)

        print('Loading new data...')
        new_df_flat_tree,new_df_sc,new_cell_label_uni,new_cell_label_uni_color = \
        Read_In_New_Data(new_filename,new_label_filename,new_label_color_filename,flag_log2,flag_norm)
        new_df_sc_final = new_df_sc.copy()

        if(feature_genes_filename!=None):
            print('Loading feature genes...')
            feature_genes= Read_From_Pickle('feature_genes',file_path_precomp)
            new_df_sc_final = new_df_sc_final[['CELL_LABEL'] + feature_genes]
        else:
            print('Loading filtered genes...')
            genes_filtered = Read_From_Pickle('genes_filtered',file_path_precomp)
            new_df_sc_final = new_df_sc_final[['CELL_LABEL'] + genes_filtered]
            if(s_method == 'LOESS'):
                print('Loading feature genes...')
                feature_genes= Read_From_Pickle('feature_genes',file_path_precomp)
                new_df_sc_final = new_df_sc_final[['CELL_LABEL'] + feature_genes]
            if(s_method == 'PCA'):
                print('Loading feature genes...')
                pca_n_PC = Read_From_Pickle('pca_n_PC',file_path_precomp)
                sklearn_pca = Read_From_Pickle('sklearn_pca',file_path_precomp)
                sklearn_transf = sklearn_pca.transform(new_df_sc_final.iloc[:,1:].values)
                df_sklearn_transf = pd.DataFrame(sklearn_transf[:,0:pca_n_PC],columns=['PC' + str(x) for x in range(pca_n_PC)])
                df_sklearn_transf.insert(0,'CELL_LABEL',new_df_sc_final['CELL_LABEL'])
                new_df_sc_final = df_sklearn_transf

        print('Mapping new cells to LLE space...')
        sklearn_lle = Read_From_Pickle('sklearn_lle',file_path_precomp)
        new_X = Map_New_Data_To_LLE(new_df_sc_final,sklearn_lle,file_path_precomp)
        new_df_flat_tree['X'] = [new_X[i,:].tolist() for i in range(new_X.shape[0])]

        print('Projecting new cells to tree structure...')
        EPG = Read_From_Pickle('EPG',file_path_precomp)
        dict_branches = Read_From_Pickle('dict_branches',file_path_precomp)
        df_cells = Project_Cells_To_Tree(EPG,new_X,dict_branches)
        new_df_flat_tree = Map_New_Data_To_Tree(new_df_flat_tree,df_cells,dict_branches)

        #Reading saved variables 
        df_sc = pd.read_msgpack(file_path_precomp+'/df_sc.msg')
        df_sc_final = pd.read_msgpack(file_path_precomp+'/df_sc_final.msg')
        df_flat_tree = Read_From_Pickle('df_flat_tree',file_path_precomp)
        X = np.array(df_flat_tree['X'].tolist())
        input_cell_label_uni = np.unique(df_flat_tree.CELL_LABEL).tolist()
        input_cell_label_uni_color = Read_From_Pickle('input_cell_label_uni_color',file_path_precomp)

        #merge new data into existing data
        X = np.vstack((X,new_X))
        df_flat_tree = pd.concat([df_flat_tree,new_df_flat_tree])
        df_flat_tree.index = range(df_flat_tree.shape[0])
        df_sc = pd.concat([df_sc,new_df_sc])
        df_sc.index = range(df_sc.shape[0])
        df_sc_final = pd.concat([df_sc_final,new_df_sc_final])
        df_sc_final.index = range(df_sc_final.shape[0])
        input_cell_label_uni = input_cell_label_uni + new_cell_label_uni
        input_cell_label_uni = np.unique(input_cell_label_uni).tolist()
        input_cell_label_uni_color.update(new_cell_label_uni_color)
 
        print('Plotting cells in LLE space...')
        Plot_Dimension_Reduction(df_sc_final,X,input_cell_label_uni,input_cell_label_uni_color,file_path)

        print('Plotting Elastic Principal Graph...')
        curves_color = Read_From_Pickle('curves_color',file_path_precomp)
        dict_node_state = Read_From_Pickle('dict_node_state',file_path_precomp)
        Plot_EPG(EPG,df_flat_tree,dict_branches,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,file_name = 'EPG',dict_node_state=dict_node_state)
        
        flat_tree = Read_From_Pickle('flat_tree',file_path_precomp)
        Flat_Tree_Plot(df_flat_tree,flat_tree,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web)
        # print('MDS...')
        # n_processes = multiprocessing.cpu_count()
        # Plot_Flat_Tree(df_flat_tree,EPG,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,n_processes,file_path)    
        print('Subway map and Stream plots...')
        list_node_start = flat_tree.nodes()
        dict_df_rooted_tree = dict()
        for node_start in list_node_start:
            df_rooted_tree = df_flat_tree.copy()
            Subway_Map_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,\
                            input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web)
            Subway_Map_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,\
                            input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,mode='contracted')    
            Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web)    
            Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web,mode='contracted')    
            dict_df_rooted_tree[node_start] = df_rooted_tree.copy()
            Output_Ordered_Info(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,file_path)
        Output_Info(df_flat_tree,flat_tree,dict_node_state,file_path)

        gene_list = []
        if(flag_gene_TG_detection):
            if(os.path.exists(file_path_precomp+'/gene_list_TG.pickle')):
                print('importing saved transition genes...')
                gene_list_TG = Read_From_Pickle('gene_list_TG',file_path_precomp)
            else:
                print('Detecting transition genes...')
                gene_list_TG = Genes_Detection_For_Transition(df_flat_tree,df_sc,input_genes,TG_spearman_cutoff,TG_diff_cutoff,\
                                    dict_node_state,n_processes,file_path,file_path_precomp,flag_web)
            gene_list = gene_list + gene_list_TG
            gene_list = list(set(gene_list))
        if(flag_gene_DE_detection):
            if(os.path.exists(file_path_precomp+'/gene_list_DE.pickle')):
                print('importing saved DE genes...')
                gene_list_DE = Read_From_Pickle('gene_list_DE',file_path_precomp)
            else:
                print('Detecting DE genes...')
                gene_list_DE = Genes_Detection_For_DE(df_flat_tree,df_sc,input_genes,DE_z_score_cutoff,DE_diff_cutoff,\
                                    dict_node_state,n_processes,file_path,file_path_precomp,flag_web)
            gene_list = gene_list + gene_list_DE
            gene_list = list(set(gene_list))
        
        if(gene_list_filename!=None):
            if(os.path.exists(gene_list_filename)):
                gene_list = pd.read_csv(gene_list_filename,sep='\t',header=None,index_col=None,compression= 'gzip' if gene_list_filename.split('.')[-1]=='gz' else None).iloc[:,0].tolist()
                gene_list = list(set(gene_list))
            else:
                gene_list = gene_list_filename.split(',')
                print gene_list
        if(len(gene_list)>0):
            print('Visulizing genes...')
            list_node_start = flat_tree.nodes()
            for node_start in list_node_start:
                df_rooted_tree = dict_df_rooted_tree[node_start]
                Subway_Map_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_atac,file_path,flag_web)
                Subway_Map_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_atac,file_path,flag_web,mode = 'contracted')
                Stream_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_stream_log_view,flag_atac,file_path,flag_web)
                Stream_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_stream_log_view,flag_atac,file_path,flag_web,mode='contracted')

    else:
        input_filename = args.input_filename
        cell_label_filename = args.cell_label_filename
        cell_label_color_filename = args.cell_label_color_filename
        s_method = args.s_method
        feature_genes_filename = args.feature_genes
        use_precomputed = args.use_precomputed
        n_processes = args.n_processes
        loess_frac = args.loess_frac
        loess_z_score_cutoff = args.loess_z_score_cutoff
        pca_max_PC = args.pca_max_PC
        pca_n_PC = args.pca_n_PC
        flag_log2 = args.flag_log2
        flag_norm = args.flag_norm
        flag_atac = args.flag_atac
        atac_regions = args.atac_regions
        atac_counts = args.atac_counts
        atac_samples = args.atac_samples
        atac_k = args.atac_k
        lle_n_nb_percent = args.lle_n_nb_percent #LLE neighbour percent
        lle_n_component = args.lle_n_component #LLE dimension reduction
        AP_damping_factor = args.AP_damping_factor
        n_cluster = args.n_cluster #number of clusters in spectral clustering
        EPG_n_nodes = args.EPG_n_nodes 
        EPG_n_rep = args.EPG_n_rep
        EPG_lambda = args.EPG_lambda
        EPG_mu = args.EPG_mu
        EPG_trimmingradius = args.EPG_trimmingradius
        EPG_prob = args.EPG_prob
        EPG_finalenergy = args.EPG_finalenergy
        EPG_alpha = args.EPG_alpha
        EPG_beta = args.EPG_beta
        flag_disable_EPG_collapse = args.flag_disable_EPG_collapse
        EPG_collapse_mode = args.EPG_collapse_mode
        EPG_collapse_par = args.EPG_collapse_par
        flag_EPG_shift = args.flag_EPG_shift
        EPG_shift_mode = args.EPG_shift_mode
        EPG_shift_DR = args.EPG_shift_DR
        EPG_shift_maxshift = args.EPG_shift_maxshift
        flag_disable_EPG_ext = args.flag_disable_EPG_ext
        EPG_ext_mode = args.EPG_ext_mode
        EPG_ext_par = args.EPG_ext_par
        output_folder = args.output_folder #work directory

        if(output_folder==None):
            file_path = os.getcwd() + '/STREAM_result'
        else:
            file_path = output_folder
        if(not os.path.exists(file_path)):
            os.makedirs(file_path)
            
        file_path_precomp = file_path +'/Precomputed'
        if(not os.path.exists(file_path_precomp)):
            os.makedirs(file_path_precomp)

        if(use_precomputed):
            input_genes = Read_From_Pickle('input_genes',file_path_precomp)
            input_cell_label_uni_color = Read_From_Pickle('input_cell_label_uni_color',file_path_precomp)
            input_cell_label_uni = Read_From_Pickle('input_cell_label_uni',file_path_precomp)
            dict_branches = Read_From_Pickle('dict_branches',file_path_precomp)
            df_sc = pd.read_msgpack(file_path_precomp+'/df_sc.msg')
            df_sc_final = pd.read_msgpack(file_path_precomp+'/df_sc_final.msg')
            df_flat_tree = Read_From_Pickle('df_flat_tree',file_path_precomp)
            flat_tree = Read_From_Pickle('flat_tree',file_path_precomp)
            dict_node_state = Read_From_Pickle('dict_node_state',file_path_precomp)
            dict_df_rooted_tree = Read_From_Pickle('dict_df_rooted_tree',file_path_precomp)
            flag_stream_log_view_saved = Read_From_Pickle('flag_stream_log_view',file_path_precomp)
            Save_To_Pickle(flag_stream_log_view,'flag_stream_log_view',file_path_precomp)
            if(flag_stream_log_view != flag_stream_log_view_saved):
                list_node_start = flat_tree.nodes()
                print('Visualizing cells...')
                for node_start in list_node_start:
                    df_rooted_tree = dict_df_rooted_tree[node_start]
                    Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web)    
                    Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web,mode='contracted')    
        else:
            if(flag_atac):
                if((atac_samples!=None) and (atac_regions!=None) and (atac_counts!=None)):
                    counts_to_kmers(atac_counts,atac_regions,atac_samples,atac_k,file_path)
                    input_filename = file_path + '/df_zscores_scaled.tsv'
            if(input_filename==None):
                print('Input file must be provided')
                sys.exit()
            Save_To_Pickle(flag_log2,'flag_log2',file_path_precomp)
            Save_To_Pickle(flag_norm,'flag_norm',file_path_precomp)
            Save_To_Pickle(s_method,'s_method',file_path_precomp)
            Save_To_Pickle(feature_genes_filename,'feature_genes_filename',file_path_precomp)
            Save_To_Pickle(flag_stream_log_view,'flag_stream_log_view',file_path_precomp)

            print('Loading input data...')
            df_flat_tree,df_sc,input_genes,input_cell_label_uni,input_cell_label_uni_color = \
            Read_In_Data(input_filename,cell_label_filename,cell_label_color_filename,flag_log2,flag_norm)
            df_sc.to_msgpack(file_path_precomp + '/df_sc.msg',compress='zlib')
            Save_To_Pickle(input_genes,'input_genes',file_path_precomp)
            Save_To_Pickle(input_cell_label_uni,'input_cell_label_uni',file_path_precomp)
            Save_To_Pickle(input_cell_label_uni_color,'input_cell_label_uni_color',file_path_precomp)

            df_sc_final = df_sc.copy()
            if(feature_genes_filename!=None):
                print('Loading specified feature genes...')
                feature_genes = pd.read_csv(feature_genes_filename,sep='\t',header=None,index_col=None,compression= 'gzip' if feature_genes_filename.split('.')[-1]=='gz' else None).iloc[:,0].tolist()
                print(str(len(feature_genes)) + ' feature genes are specified')
                Save_To_Pickle(feature_genes,'feature_genes',file_path_precomp)
                df_sc_final = df_sc_final[['CELL_LABEL'] + feature_genes]
            else:
                if(not flag_atac):
                    print('Filtering genes...')
                    genes_filtered = Filter_Genes(df_sc_final)
                    Save_To_Pickle(genes_filtered,'genes_filtered',file_path_precomp)
                    df_sc_final = df_sc_final[['CELL_LABEL'] + genes_filtered]
                if(s_method!='all'):
                    print('Selecting features...')
                    if(s_method == 'LOESS'):
                        feature_genes = Select_Variable_Genes(df_sc_final,loess_frac,loess_z_score_cutoff,n_processes,file_path,flag_web)
                        Save_To_Pickle(feature_genes,'feature_genes',file_path_precomp)
                        df_sc_final = df_sc_final[['CELL_LABEL'] + feature_genes]
                    elif(s_method == 'PCA'):
                        df_sc_final = Select_Principal_Components(df_sc_final,pca_max_PC,pca_n_PC,flag_first_PC,file_path,file_path_precomp,flag_web)
                    else:
                        print('\'s\' must be \'LOESS\' or \'PCA\'')
                        sys.exit()

            df_sc_final.to_msgpack(file_path_precomp + '/df_sc_final.msg',compress='zlib')

            print('Number of CPUs being used: ' + str(n_processes))
            print('Reducing dimension...')
            X = Dimension_Reduction(df_sc_final,lle_n_component,lle_n_nb_percent,file_path,file_path_precomp,n_processes)
            if(not flag_web):
                Plot_Dimension_Reduction(df_sc_final,X,input_cell_label_uni,input_cell_label_uni_color,file_path)
            df_flat_tree['X'] = [X[i,:].tolist() for i in range(X.shape[0])]
    
            print('Structure Learning...')
            df_flat_tree,EPG,flat_tree,dict_branches,dict_node_state,curves_color = Structure_Learning(df_flat_tree,AP_damping_factor,n_cluster,lle_n_nb_percent,EPG_n_nodes,EPG_n_rep,EPG_lambda,EPG_mu,EPG_trimmingradius,EPG_prob,EPG_finalenergy,EPG_alpha,EPG_beta,
                                                        flag_disable_EPG_collapse,EPG_collapse_mode,EPG_collapse_par,
                                                        flag_EPG_shift,EPG_shift_mode,EPG_shift_DR,EPG_shift_maxshift,
                                                        flag_disable_EPG_ext,EPG_ext_mode,EPG_ext_par,
                                                        n_processes,input_cell_label_uni,input_cell_label_uni_color,file_path,file_path_precomp,flag_web)

            Save_To_Pickle(EPG,'EPG',file_path_precomp)
            Save_To_Pickle(flat_tree,'flat_tree',file_path_precomp)
            Save_To_Pickle(dict_branches,'dict_branches',file_path_precomp)
            Save_To_Pickle(dict_node_state,'dict_node_state',file_path_precomp)
            Save_To_Pickle(curves_color,'curves_color',file_path_precomp)            

            print('Visualizing cells...')
            print('Flat tree plots...')
            Flat_Tree_Plot(df_flat_tree,flat_tree,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web)
            # Flat_Tree_Plot(df_flat_tree,flat_tree,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,mode='contracted')
            if(flag_mds):
                print('MDS...')
                Plot_Flat_Tree(df_flat_tree,EPG,dict_branches,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,curves_color,n_processes,file_path)
            Save_To_Pickle(df_flat_tree,'df_flat_tree',file_path_precomp)
           
            print('Subway map and Stream plots...')
            list_node_start = flat_tree.nodes()
            dict_df_rooted_tree = dict()
            for node_start in list_node_start:
                df_rooted_tree = df_flat_tree.copy()
                Subway_Map_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,\
                               input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web)
                # Subway_Map_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,\
                #                input_cell_label_uni,input_cell_label_uni_color,curves_color,file_path,flag_web,mode='contracted')    
                Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web)    
                # Stream_Plot(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,input_cell_label_uni,input_cell_label_uni_color,flag_stream_log_view,file_path,flag_web,mode='contracted')    
                dict_df_rooted_tree[node_start] = df_rooted_tree.copy()
                Output_Ordered_Info(df_rooted_tree,flat_tree,dict_branches,node_start,dict_node_state,file_path)    
            Save_To_Pickle(dict_df_rooted_tree,'dict_df_rooted_tree',file_path_precomp)
            Output_Info(df_flat_tree,flat_tree,dict_node_state,file_path)

        gene_list = []
        if(flag_gene_TG_detection):
            if(os.path.exists(file_path_precomp+'/gene_list_TG.pickle')):
                print('importing saved transition genes...')
                gene_list_TG = Read_From_Pickle('gene_list_TG',file_path_precomp)
            else:
                print('Detecting transition genes...')
                gene_list_TG = Genes_Detection_For_Transition(df_flat_tree,df_sc,input_genes,TG_spearman_cutoff,TG_diff_cutoff,\
                                    dict_node_state,n_processes,file_path,file_path_precomp,flag_web)
                Save_To_Pickle(gene_list_TG,'gene_list_TG',file_path_precomp)
            gene_list = gene_list + gene_list_TG
            gene_list = list(set(gene_list))
        if(flag_gene_DE_detection):
            if(os.path.exists(file_path_precomp+'/gene_list_DE.pickle')):
                print('importing saved DE genes...')
                gene_list_DE = Read_From_Pickle('gene_list_DE',file_path_precomp)
            else:
                print('Detecting DE genes...')
                gene_list_DE = Genes_Detection_For_DE(df_flat_tree,df_sc,input_genes,DE_z_score_cutoff,DE_diff_cutoff,\
                                    dict_node_state,n_processes,file_path,file_path_precomp,flag_web)
                Save_To_Pickle(gene_list_DE,'gene_list_DE',file_path_precomp)
            gene_list = gene_list + gene_list_DE
            gene_list = list(set(gene_list))
        
    
    if(gene_list_filename!=None):
        if(os.path.exists(gene_list_filename)):
            gene_list = pd.read_csv(gene_list_filename,sep='\t',header=None,index_col=None,compression= 'gzip' if gene_list_filename.split('.')[-1]=='gz' else None).iloc[:,0].tolist()
            gene_list = list(set(gene_list))
        else:
            gene_list = gene_list_filename.split(',')
            print gene_list
    if(len(gene_list)>0):
        print('Visulizing genes...')
        list_node_start = flat_tree.nodes()
        for node_start in list_node_start:
            df_rooted_tree = dict_df_rooted_tree[node_start]
            Subway_Map_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_atac,file_path,flag_web)
            # Subway_Map_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_atac,file_path,flag_web,mode = 'contracted')
            Stream_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_stream_log_view,flag_atac,file_path,flag_web)
            # Stream_Plot_Gene(df_rooted_tree,df_sc,flat_tree,dict_branches,node_start,dict_node_state,gene_list,flag_stream_log_view,flag_atac,file_path,flag_web,mode='contracted')
    print('Finished computation...')
if __name__ == "__main__":
    main()