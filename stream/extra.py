import numpy as np
import pandas as pd
import os
from pandas.api.types import is_string_dtype,is_numeric_dtype
import networkx as nx
import seaborn as sns
import pylab as plt
import matplotlib as mpl
import shapely.geometry as geom
from copy import deepcopy
import itertools
from scipy.spatial import distance,cKDTree,KDTree
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline,UnivariateSpline
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import coo_matrix,diags
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import math
from decimal import *


def get_extension(filename):
    fn,ext = os.path.splitext(filename)
    while(ext[1:] in ['gz','bz2','zip','xz']):
        fn,ext = os.path.splitext(fn)
    return ext[1:]

def project_point_to_curve_distance(XP,p):
    curve = geom.LineString(XP)
    point = geom.Point(p)
    #distance from point to curve
    dist_p_to_c = point.distance(curve)
    return dist_p_to_c    

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def cal_tf_idf(mat):
    mat = coo_matrix(mat)
    tf = coo_matrix(mat/(mat.sum(axis=0)))
    idf = np.array(np.log(1 + mat.shape[1] / mat.sum(axis=1))).flatten()
    tf_idf = np.dot(diags(idf),tf)
    tf_idf = tf_idf.toarray()
    return tf_idf

def dfs_from_leaf(epg_copy,node,degrees_of_nodes,nodes_to_visit,nodes_to_merge):
    nodes_to_visit.remove(node)
    for n2 in epg_copy.neighbors(node):
        if n2 in nodes_to_visit:
            if degrees_of_nodes[n2]==2:  #grow the branch
                if n2 not in nodes_to_merge:
                    nodes_to_merge.append(n2)
                dfs_from_leaf(epg_copy,n2,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            else:
                nodes_to_merge.append(n2)
                return


def add_branch_info(epg,dict_branches):
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    sns_palette = sns.color_palette("hls", len(dict_branches)).as_hex()
    if(dict_nodes_pos != {}):
        for i,(br_key,br_value) in enumerate(dict_branches.items()):
            nodes = br_value['nodes']
            dict_branches[br_key]['id'] = (nodes[0],nodes[-1]) #the direction of nodes for each branch
            br_nodes_pos = np.array([dict_nodes_pos[i] for i in nodes]) 
            dict_branches[br_key]['len'] = sum(np.sqrt(((br_nodes_pos[0:-1,:] - br_nodes_pos[1:,:])**2).sum(1)))
            dict_branches[br_key]['color'] = sns_palette[i]
    return dict_branches


def extract_branches(epg):
    #record the original degree(before removing nodes) for each node
    degrees_of_nodes = epg.degree()
    epg_copy = epg.copy()
    dict_branches = dict()
    clusters_to_merge=[]
    while epg_copy.order()>1: #the number of vertices
        leaves=[n for n,d in epg_copy.degree() if d==1]
        nodes_included=list(epg_copy.nodes())
        while leaves:
            leave=leaves.pop()
            nodes_included.remove(leave)
            nodes_to_merge=[leave]
            nodes_to_visit=list(epg_copy.nodes())
            dfs_from_leaf(epg_copy,leave,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            clusters_to_merge.append(nodes_to_merge)
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])] = {}
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])]['nodes'] = nodes_to_merge
            nodes_to_delete = nodes_to_merge[0:len(nodes_to_merge)-1]
            if epg_copy.degree()[nodes_to_merge[-1]] == 1: #avoid the single point
                nodes_to_delete = nodes_to_merge
                leaves = []
            epg_copy.remove_nodes_from(nodes_to_delete)
    dict_branches = add_branch_info(epg,dict_branches)
    # print('Number of branches: ' + str(len(clusters_to_merge)))
    return dict_branches

# def plot_mst(adata,n_components = 3,**kwargs):
#     options = {
#             'save_fig' : False,
#             'fig_size':(8,8),
#             'fig_path' :  adata.uns['workdir'],
#             'fig_name' : 'dimension_reduction.pdf',}
#     options.update(kwargs)
#     save_fig = options['save_fig']
#     fig_size = options['fig_size']
#     fig_path = options['fig_path']
#     fig_name = options['fig_name']

#     mst = adata.uns['mst']
#     XC = nx.get_node_attributes(mst,'pos')
#     df_coord = pd.DataFrame(adata.obsm['X_dr'],index=adata.obs_names)
#     coord = df_coord.sample(frac=1,random_state=100)
    
#     if(n_components>=3): 
#         fig = plt.figure(figsize=fig_size)
#         ax = fig.add_subplot(111, projection='3d')
#         for n in mst.nodes():
#             ax.scatter(XC[n][0],XC[n][1],XC[n][2],color='#EC4E4E',s=80,marker='o',alpha=0.9,zorder=100)
#         for edge in mst.edges():
#             x_pos = (XC[edge[0]][0],XC[edge[1]][0])
#             y_pos = (XC[edge[0]][1],XC[edge[1]][1])
#             z_pos = (XC[edge[0]][2],XC[edge[1]][2])
#             ax.plot(x_pos,y_pos,z_pos,'#3182bd',lw=2,zorder=10)
#         max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min(), coord[2].max()-coord[2].min()]).max() / 1.9
#         mid_x = (coord[0].max()+coord[0].min()) * 0.5
#         mid_y = (coord[1].max()+coord[1].min()) * 0.5
#         mid_z = (coord[2].max()+coord[2].min()) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)
#         ax.set_xlabel('Component1',labelpad=20)
#         ax.set_ylabel('Component2',labelpad=20)
#         ax.set_zlabel('Component3',labelpad=20)
#         if(save_fig):
#             plt.savefig(fig_path + fig_name,pad_inches=1,bbox_inches='tight')
#             plt.close(fig)
#     if(n_components==2): 
#         fig = plt.figure(figsize=fig_size)
#         ax = fig.add_subplot(111)
#         for n in mst.nodes():
#             ax.scatter(XC[n][0],XC[n][1],color='#EC4E4E',s=80,marker='o',alpha=0.9,zorder=100)
#         for edge in mst.edges():
#             x_pos = (XC[edge[0]][0],XC[edge[1]][0])
#             y_pos = (XC[edge[0]][1],XC[edge[1]][1])
#             ax.plot(x_pos,y_pos,'#3182bd',lw=2,zorder=10)    
#         max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min()]).max() / 1.9
#         mid_x = (coord[0].max()+coord[0].min()) * 0.5
#         mid_y = (coord[1].max()+coord[1].min()) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_xlabel('Component1',labelpad=20)
#         ax.set_ylabel('Component2',labelpad=20)
#         if(save_fig):
#             plt.savefig(fig_path + fig_name,pad_inches=1,bbox_inches='tight')
#             plt.close(fig)      


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



def project_cells_to_epg(adata):
    input_data = adata.obsm['X_dr']
    epg = adata.uns['epg']
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    nodes_pos = np.empty((0,input_data.shape[1]))
    nodes = np.empty((0,1),dtype=int)
    for key in dict_nodes_pos.keys():
        nodes_pos = np.vstack((nodes_pos,dict_nodes_pos[key]))
        nodes = np.append(nodes,key)    
    indices = pairwise_distances_argmin_min(input_data,nodes_pos,axis=1,metric='euclidean')[0]
    x_node = nodes[indices]
    adata.obs['node'] = x_node
    #update the projection info for each cell
    flat_tree = adata.uns['flat_tree']
    dict_branches_nodes = nx.get_edge_attributes(flat_tree,'nodes')
    dict_branches_id = nx.get_edge_attributes(flat_tree,'id')
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    list_x_br_id = list()
    list_x_br_id_alias = list()
    list_x_lam = list()
    list_x_dist = list()
    for ix,xp in enumerate(input_data): 
        list_br_id = [flat_tree.edges[br_key]['id'] for br_key,br_value in dict_branches_nodes.items() if x_node[ix] in br_value]
        dict_br_matrix = dict()
        for br_id in list_br_id:
            dict_br_matrix[br_id] = np.array([dict_nodes_pos[i] for i in flat_tree.edges[br_id]['nodes']])            
        dict_results = dict()
        list_dist_xp = list()
        for br_id in list_br_id:
            dict_results[br_id] = project_point_to_line_segment_matrix(dict_br_matrix[br_id],xp)
            list_dist_xp.append(dict_results[br_id][2])
        x_br_id = list_br_id[np.argmin(list_dist_xp)]
        x_br_id_alias = dict_node_state[x_br_id[0]],dict_node_state[x_br_id[1]]
        br_len = flat_tree.edges[x_br_id]['len']
        results = dict_results[x_br_id]
        x_dist = results[2]
        x_lam = results[3]
        if(x_lam>br_len):
            x_lam = br_len 
        list_x_br_id.append(x_br_id)
        list_x_br_id_alias.append(x_br_id_alias)
        list_x_lam.append(x_lam)
        list_x_dist.append(x_dist)
    adata.obs['branch_id'] = list_x_br_id
    adata.obs['branch_id_alias'] = list_x_br_id_alias
#     adata.uns['branch_id'] = list(set(adata.obs['branch_id'].tolist()))
    adata.obs['branch_lam'] = list_x_lam
    adata.obs['branch_dist'] = list_x_dist
    return None


def calculate_pseudotime(adata):
    flat_tree = adata.uns['flat_tree']
    dict_edge_len = nx.get_edge_attributes(flat_tree,'len')
    adata.obs = adata.obs[adata.obs.columns.drop(list(adata.obs.filter(regex='_pseudotime')))].copy()
    # dict_nodes_pseudotime = dict()
    for root_node in flat_tree.nodes():
        df_pseudotime = pd.Series(index=adata.obs.index)
        list_bfs_edges = list(nx.bfs_edges(flat_tree,source=root_node))
        dict_bfs_predecessors = dict(nx.bfs_predecessors(flat_tree,source=root_node))
        for edge in list_bfs_edges:
            list_pre_edges = list()
            pre_node = edge[0]
            while(pre_node in dict_bfs_predecessors.keys()):
                pre_edge = (dict_bfs_predecessors[pre_node],pre_node)
                list_pre_edges.append(pre_edge)
                pre_node = dict_bfs_predecessors[pre_node]
            len_pre_edges = sum([flat_tree.edges[x]['len'] for x in list_pre_edges]) 
            indices = adata.obs[(adata.obs['branch_id'] == edge) | (adata.obs['branch_id'] == (edge[1],edge[0]))].index
            if(edge==flat_tree.edges[edge]['id']):
                df_pseudotime[indices] = len_pre_edges + adata.obs.loc[indices,'branch_lam']
            else:
                df_pseudotime[indices] = len_pre_edges + (flat_tree.edges[edge]['len']-adata.obs.loc[indices,'branch_lam'])
        adata.obs[flat_tree.node[root_node]['label']+'_pseudotime'] = df_pseudotime
        # dict_nodes_pseudotime[root_node] = df_pseudotime
    # nx.set_node_attributes(flat_tree,values=dict_nodes_pseudotime,name='pseudotime')
    # adata.uns['flat_tree'] = flat_tree
    return None

def construct_flat_tree(dict_branches):
    flat_tree = nx.Graph()
    flat_tree.add_nodes_from(list(set(itertools.chain.from_iterable(dict_branches.keys()))))
    flat_tree.add_edges_from(dict_branches.keys())
    root = list(flat_tree.nodes())[0]
    edges = nx.bfs_edges(flat_tree, root)
    nodes = [root] + [v for u, v in edges]  
    dict_nodes_label = dict()
    for i,node in enumerate(nodes):
        dict_nodes_label[node] = 'S'+str(i) 
    nx.set_node_attributes(flat_tree,values=dict_nodes_label,name='label')
    dict_branches_color = dict()
    dict_branches_len = dict()
    dict_branches_nodes = dict()
    dict_branches_id = dict() #the direction of nodes for each edge
    for x in dict_branches.keys():
        dict_branches_color[x]=dict_branches[x]['color']
        dict_branches_len[x]=dict_branches[x]['len']
        dict_branches_nodes[x]=dict_branches[x]['nodes']
        dict_branches_id[x]=dict_branches[x]['id'] 
    nx.set_edge_attributes(flat_tree,values=dict_branches_nodes,name='nodes')
    nx.set_edge_attributes(flat_tree,values=dict_branches_id,name='id')
    nx.set_edge_attributes(flat_tree,values=dict_branches_color,name='color')
    nx.set_edge_attributes(flat_tree,values=dict_branches_len,name='len')    
    return flat_tree

def add_flat_tree_node_pos(adata):
    ## add the positions of flat tree's nodes
    flat_tree = adata.uns['flat_tree']
    ft_node_pos = nx.spring_layout(flat_tree,random_state=10)
    bfs_root = list(flat_tree.nodes())[0]
    bfs_edges = list(nx.bfs_edges(flat_tree, bfs_root))
    bfs_nodes = [bfs_root] + [v for u, v in bfs_edges]    
    
    ft_node_pos_updated = deepcopy(ft_node_pos)
    flat_tree_copy = deepcopy(flat_tree)
    flat_tree_copy.remove_node(bfs_root)
    for i,edge_i in enumerate(bfs_edges):
        dist_nodes = distance.euclidean(ft_node_pos_updated[edge_i[0]],ft_node_pos_updated[edge_i[1]])
        len_edge = flat_tree.edges[edge_i]['len']
        st_x = ft_node_pos_updated[edge_i[0]][0]
        ed_x = ft_node_pos_updated[edge_i[1]][0]
        st_y = ft_node_pos_updated[edge_i[0]][1]
        ed_y = ft_node_pos_updated[edge_i[1]][1]
        p_x = st_x + (ed_x - st_x)*(len_edge/dist_nodes)
        p_y = st_y + (ed_y - st_y)*(len_edge/dist_nodes)
        ft_node_pos_updated[edge_i[1]] = np.array([p_x,p_y])

        con_components = list(nx.connected_components(flat_tree_copy))
        #update other reachable unvisited nodes
        for con_comp in con_components:
            if edge_i[1] in con_comp:
                reachable_unvisited = con_comp - {edge_i[1]}
                flat_tree_copy.remove_node(edge_i[1])
                break
        for nd in reachable_unvisited:
            nd_x = ft_node_pos_updated[nd][0] + p_x - ed_x
            nd_y = ft_node_pos_updated[nd][1] + p_y - ed_y
            ft_node_pos_updated[nd] = np.array([nd_x,nd_y])

    nx.set_node_attributes(flat_tree, values=ft_node_pos_updated,name='pos_spring')

def add_flat_tree_cell_pos(adata,dist_scale):
    ## Update the positions of cells on flat tree
    cells_pos = np.empty([adata.shape[0],2])
    flat_tree = adata.uns['flat_tree']
    ft_node_pos = nx.get_node_attributes(flat_tree,'pos_spring')
    
    list_branch_id = nx.get_edge_attributes(flat_tree,'id').values()   
    for br_id in list_branch_id:
        s_pos = ft_node_pos[br_id[0]] #start node position
        e_pos = ft_node_pos[br_id[1]] #end node position
        dist_se = distance.euclidean(s_pos,e_pos)
        p_x = np.array(adata.obs[adata.obs['branch_id'] == br_id]['branch_lam'].tolist())
        dist_p = dist_scale*np.array(adata.obs[adata.obs['branch_id'] == br_id]['branch_dist'].tolist())
        np.random.seed(100)
        p_y = np.random.choice([1,-1],size=len(p_x))*dist_p
        #rotation matrix
        ro_angle = np.arctan2((e_pos-s_pos)[1],(e_pos-s_pos)[0])#counterclockwise angle
        p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y*math.sin(ro_angle)
        p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y*math.cos(ro_angle)
        p_pos = np.array((p_x_prime,p_y_prime)).T
        cells_pos[np.where(adata.obs['branch_id'] == br_id)[0],:] =[p_pos[i,:].tolist() for i in range(p_pos.shape[0])]
    adata.obsm['X_spring'] = cells_pos    

def calculate_shift_distance(adata,root='S0',dist_pctl=95,preference=None):
    flat_tree = adata.uns['flat_tree']
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}  
    root_node = dict_label_node[root]
    ##shift distance for each branch
    dict_edge_shift_dist = dict()
    max_dist = np.percentile(adata.obs['branch_dist'],dist_pctl) ## maximum distance from cells to branch
    leaves = [k for k,v in flat_tree.degree() if v==1]
    n_nonroot_leaves = len(list(set(leaves) - set([root_node])))
    dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree,root_node))
    dict_bfs_suc = dict(nx.bfs_successors(flat_tree,root_node))
    #depth first search
    if(preference != None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    dfs_nodes = dfs_nodes_modified(flat_tree,root_node,preference=preference_nodes)
    dfs_nodes_copy = deepcopy(dfs_nodes)
    id_leaf = 0
    while(len(dfs_nodes_copy)>1):
        node = dfs_nodes_copy.pop()
        pre_node = dict_bfs_pre[node]
        if(node in leaves):
            dict_edge_shift_dist[(pre_node,node)] = 2*max_dist*(id_leaf-(n_nonroot_leaves/2.0))
            id_leaf = id_leaf+1
        else:
            suc_nodes = dict_bfs_suc[node]
            dict_edge_shift_dist[(pre_node,node)] = (sum([dict_edge_shift_dist[(node,sn)] for sn in suc_nodes]))/float(len(suc_nodes))            
    return dict_edge_shift_dist

def add_stream_sc_pos(adata,root='S0',dist_scale=1,dist_pctl=95,preference=None):
    flat_tree = adata.uns['flat_tree']
    label_to_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    
    root_node = label_to_node[root]
    dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree,root_node))
    dict_bfs_suc = dict(nx.bfs_successors(flat_tree,root_node))
    dict_edge_shift_dist = calculate_shift_distance(adata,root=root,dist_pctl=dist_pctl,preference=preference)
    dict_path_len = nx.shortest_path_length(flat_tree,source=root_node,weight='len')
    df_cells_pos = pd.DataFrame(index=adata.obs.index,columns=['cells_pos'])
    dict_edge_pos = {}
    dict_node_pos = {}
    for edge in dict_edge_shift_dist.keys():
        node_pos_st = np.array([dict_path_len[edge[0]],dict_edge_shift_dist[edge]])
        node_pos_ed = np.array([dict_path_len[edge[1]],dict_edge_shift_dist[edge]])  
        br_id = flat_tree.edges[edge]['id']
        id_cells = np.where(adata.obs['branch_id']==br_id)[0]
        # cells_pos_x = flat_tree.nodes[root_node]['pseudotime'].iloc[id_cells]
        cells_pos_x = adata.obs[flat_tree.node[root_node]['label']+'_pseudotime'].iloc[id_cells]
        np.random.seed(100)
        cells_pos_y = node_pos_st[1] + dist_scale*adata.obs.iloc[id_cells,]['branch_dist']*np.random.choice([1,-1],size=id_cells.shape[0])
        cells_pos = np.array((cells_pos_x,cells_pos_y)).T
        df_cells_pos.iloc[id_cells,0] = [cells_pos[i,:].tolist() for i in range(cells_pos.shape[0])]
        dict_edge_pos[edge] = np.array([node_pos_st,node_pos_ed])    
        if(edge[0] not in dict_bfs_pre.keys()):
            dict_node_pos[edge[0]] = node_pos_st
        dict_node_pos[edge[1]] = node_pos_ed
    adata.obsm['X_stream_'+root] = np.array(df_cells_pos['cells_pos'].tolist())
    
    if(flat_tree.degree(root_node)>1):
        suc_nodes = dict_bfs_suc[root_node]
        edges = [(root_node,sn) for sn in suc_nodes]
        max_y_pos = max([dict_edge_pos[x][0,1] for x in edges])
        min_y_pos = min([dict_edge_pos[x][0,1] for x in edges])
        median_y_pos = np.median([dict_edge_pos[x][0,1] for x in edges])
        x_pos = dict_edge_pos[edges[0]][0,0]
        dict_node_pos[root_node] = np.array([x_pos,median_y_pos])

    adata.uns['stream_'+root] = dict()
    adata.uns['stream_'+root]['nodes'] = dict_node_pos
    adata.uns['stream_'+root]['edges'] = dict()
     
    for edge in dict_edge_pos.keys():  
        edge_pos = dict_edge_pos[edge]
        edge_color = flat_tree.edges[edge]['color']
        if(edge[0] in dict_bfs_pre.keys()):
            pre_node = dict_bfs_pre[edge[0]]
            link_edge_pos = np.array([dict_edge_pos[(pre_node,edge[0])][1,],dict_edge_pos[edge][0,]])
            edge_pos = np.vstack((link_edge_pos,edge_pos))
        adata.uns['stream_'+root]['edges'][edge]=edge_pos
    if(flat_tree.degree(root_node)>1):
        suc_nodes = dict_bfs_suc[root_node]
        edges = [(root_node,sn) for sn in suc_nodes]
        max_y_pos = max([dict_edge_pos[x][0,1] for x in edges])
        min_y_pos = min([dict_edge_pos[x][0,1] for x in edges])
        x_pos = dict_node_pos[root_node][0]
        link_edge_pos = np.array([[x_pos,min_y_pos],[x_pos,max_y_pos]])
        adata.uns['stream_'+root]['edges'][(root_node,root_node)]=link_edge_pos

def arrowed_spines(
        ax,
        x_width_fraction=0.03,
        x_height_fraction=0.02,
        lw=None,
        ohg=0.2,
        locations=('bottom right', 'left up'),
        **arrow_kwargs
):
    """
    Add arrows to the requested spines
    Code originally sourced here: https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. If not supplied, default behaviour is to use the value on the current bottom spine.
               ('width' in ax.arrow() is actually controling the line width)
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". These must be orthogonal
    (e.g. "left left" will result in an error). Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
#         lw = ax.spines['bottom'].get_linewidth()
        lw = ax.spines['bottom'].get_linewidth()*1e-4

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == 'left' else xmax

            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax


#         annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw = lw,
#                  head_width=head_width, head_length=head_length, **arrow_kwargs)
        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', width = lw,
                 head_width=head_width, head_length=head_length, **arrow_kwargs)

    return annots    


## modified depth first search
def dfs_nodes_modified(tree, source, preference=None):
    visited, stack = [], [source]
    bfs_tree = nx.bfs_tree(tree,source=source)
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            unvisited = set(tree[vertex]) - set(visited)
            if(preference != None):
                weights = list()
                for x in unvisited:
                    successors = dict(nx.bfs_successors(bfs_tree,source=x))
                    successors_nodes = list(itertools.chain.from_iterable(successors.values()))
                    weights.append(min([preference.index(s) if s in preference else len(preference) for s in successors_nodes+[x]]))
                unvisited = [x for _,x in sorted(zip(weights,unvisited),reverse=True,key=lambda x: x[0])]
            stack.extend(unvisited)
    return visited

def bfs_edges_modified(tree, source, preference=None):
    visited, queue = [], [source]
    bfs_tree = nx.bfs_tree(tree,source=source)
    predecessors = dict(nx.bfs_predecessors(bfs_tree,source))
    edges = []
    while queue:
        vertex = queue.pop()
        if vertex not in visited:
            visited.append(vertex)
            if(vertex in predecessors.keys()):
                edges.append((predecessors[vertex],vertex))
            unvisited = set(tree[vertex]) - set(visited)
            if(preference != None):
                weights = list()
                for x in unvisited:
                    successors = dict(nx.bfs_successors(bfs_tree,source=x))
                    successors_nodes = list(itertools.chain.from_iterable(successors.values()))
                    weights.append(min([preference.index(s) if s in preference else len(preference) for s in successors_nodes+[x]]))
                unvisited = [x for _,x in sorted(zip(weights,unvisited),reverse=True,key=lambda x: x[0])]
            queue.extend(unvisited)
    return edges


def count_cell_for_each_window(adata,s_win=None):
    flat_tree = adata.uns['flat_tree']
    if(s_win is None):
        s_win = min(nx.get_edge_attributes(flat_tree,'len').values())/3.0
    min_edge_len = min(nx.get_edge_attributes(flat_tree,'len').values())
    joint_len = min(s_win/2.0,min_edge_len/3.0) #length for joint area for each edge
    dict_win_ncells = dict()
    list_cell_labels = np.unique(adata.obs['label']).tolist()
    ##backbone points including endpoints, joint points, midpoint in each edge
    for edge_i in flat_tree.edges():
        br_id = flat_tree.edges[edge_i]['id']
        #all neighbor edges connectd with start node
        list_edges_st = [(br_id[0],x) for x in set(flat_tree.neighbors(br_id[0]))-set([br_id[1]])] 
        #all neighbor edges connectd with end node
        list_edges_ed = [(br_id[1],x) for x in set(flat_tree.neighbors(br_id[1]))-set([br_id[0]])]   
        len_edge_i = flat_tree.edges[edge_i]['len'] 
        pt_core = np.array([0,joint_len,0.5*len_edge_i,len_edge_i-joint_len,len_edge_i]) 
        #optional points
        if(s_win<len_edge_i):
            pt_opt = np.arange(start=0,stop=len_edge_i, step=s_win)
            if(abs(int(flat_tree.edges[edge_i]['len']/s_win) - flat_tree.edges[edge_i]['len']/s_win)>1e-3):
                pt_opt = np.concatenate((pt_opt,np.arange(start=len_edge_i,stop=0, step=-s_win)))
            pt_opt = np.unique(pt_opt[(pt_opt>pt_core[1])&(pt_opt<pt_core[3])])
        pt = np.unique(np.concatenate((pt_core,pt_opt)))
        df_win_ncells_edge_i = pd.DataFrame(pt,index=range(len(pt)),columns=['pt_lam'])
        df_cells_edge_i = adata.obs[adata.obs['branch_id'] == br_id].copy()
        for cl in list_cell_labels:
            df_cells_edge_i_cl = df_cells_edge_i[df_cells_edge_i['label']==cl].copy()
            lam_edge_i_cl = df_cells_edge_i_cl['branch_lam']
            list_n_cells_pt = list()
            for pt_i in pt:
                if(((pt_i-s_win/2.0)>=0) & ((pt_i+s_win/2.0)<=len_edge_i)):
                    n_cells_pt_i = len(lam_edge_i_cl[(lam_edge_i_cl>=(pt_i-s_win/2.0))&
                                                     (lam_edge_i_cl<=(pt_i+s_win/2.0))])
                    list_n_cells_pt.append(n_cells_pt_i)
                elif(((pt_i-s_win/2.0)<0) & ((pt_i+s_win/2.0)<=len_edge_i)):
                    n_cells_pt_i = len(lam_edge_i_cl[(lam_edge_i_cl>=0)&(lam_edge_i_cl<=(pt_i+s_win/2.0))])
                    n_cells_on_es = 0
                    for es in list_edges_st:
                        br_id_es = flat_tree.edges[es]['id']
                        len_es = flat_tree.edges[es]['len'] 
                        df_cells_es = adata.obs[(adata.obs['branch_id'] == br_id_es) &
                                                (adata.obs['label'] == cl)].copy()
                        lam_es = df_cells_es['branch_lam']
                        if(es == br_id_es):
                            n_cells_on_es = n_cells_on_es + len(lam_es[(lam_es>0)&(lam_es<=abs(pt_i-s_win/2.0))])
                        else:
                            n_cells_on_es = n_cells_on_es + len(lam_es[(lam_es>=(len_es-abs(pt_i-s_win/2.0)))&(lam_es<len_es)])
                    list_n_cells_pt.append(n_cells_pt_i+n_cells_on_es)                                             
                elif(((pt_i-s_win/2.0)>=0) & ((pt_i+s_win/2.0)>len_edge_i)):
                    n_cells_pt_i = len(lam_edge_i_cl[(lam_edge_i_cl>=(pt_i-s_win/2.0))&(lam_edge_i_cl<=len_edge_i)])     
                    n_cells_on_ed = 0
                    for ed in list_edges_ed:
                        br_id_ed = flat_tree.edges[ed]['id']
                        len_ed = flat_tree.edges[ed]['len'] 
                        df_cells_ed = adata.obs[(adata.obs['branch_id'] == br_id_ed) &
                                                (adata.obs['label'] == cl)].copy()
                        lam_ed = df_cells_ed['branch_lam']
                        if(ed == br_id_ed):
                            n_cells_on_ed = n_cells_on_ed + len(lam_ed[(lam_ed>0)&(lam_ed<=(pt_i+s_win/2.0-len_edge_i))])
                        else:
                            n_cells_on_ed = n_cells_on_ed + len(lam_ed[(lam_ed>=(len_ed-(pt_i+s_win/2.0-len_edge_i)))&(lam_ed<len_ed)])
                    list_n_cells_pt.append(n_cells_pt_i+n_cells_on_ed)       
                elif(((pt_i-s_win/2.0)<0) & ((pt_i+s_win/2.0)>len_edge_i)):
                    n_cells_pt_i = len(lam_edge_i_cl[(lam_edge_i_cl>=0)&
                                                     (lam_edge_i_cl<=len_edge_i)])
                    n_cells_on_es = 0
                    for es in list_edges_st:
                        br_id_es = flat_tree.edges[es]['id']
                        len_es = flat_tree.edges[es]['len'] 
                        df_cells_es = adata.obs[(adata.obs['branch_id'] == br_id_es) &
                                                (adata.obs['label'] == cl)].copy()
                        lam_es = df_cells_es['branch_lam']
                        if(es == br_id_es):
                            n_cells_on_es = n_cells_on_es + len(lam_es[(lam_es>0)&(lam_es<=abs(pt_i-s_win/2.0))])
                        else:
                            n_cells_on_es = n_cells_on_es + len(lam_es[(lam_es>=(len_es-abs(pt_i-s_win/2.0)))&(lam_es<len_es)])
                    n_cells_on_ed = 0
                    for ed in list_edges_ed:
                        br_id_ed = flat_tree.edges[ed]['id']
                        len_ed = flat_tree.edges[ed]['len'] 
                        df_cells_ed = adata.obs[(adata.obs['branch_id'] == br_id_ed) &
                                                (adata.obs['label'] == cl)].copy()
                        lam_ed = df_cells_ed['branch_lam']
                        if(ed == br_id_ed):
                            n_cells_on_ed = n_cells_on_ed + len(lam_ed[(lam_ed>0)&(lam_ed<=(pt_i+s_win/2.0-len_edge_i))])
                        else:
                            n_cells_on_ed = n_cells_on_ed + len(lam_ed[(lam_ed>=(len_ed-(pt_i+s_win/2.0-len_edge_i)))&(lam_ed<len_ed)])

                    list_n_cells_pt.append(n_cells_pt_i+n_cells_on_es+n_cells_on_ed) 
            df_win_ncells_edge_i[cl] = pd.Series(list_n_cells_pt)
        dict_win_ncells[br_id] = df_win_ncells_edge_i    
    return dict_win_ncells

def order_cell_label(adata,dict_win_ncells,root_node=None):
    flat_tree = adata.uns['flat_tree']
    if(root_node is None):
        root_node = list(flat_tree.nodes())[0]
    list_bfs_edges = list(nx.bfs_edges(flat_tree,root_node))
    br_id = flat_tree.edges[list_bfs_edges[0]]['id']
    if(br_id == list_bfs_edges[0]):
        df_ncells_edge_root = dict_win_ncells[br_id] 
    else:
        df_ncells_edge_root = dict_win_ncells[br_id].copy()
        df_ncells_edge_root = df_ncells_edge_root.iloc[::-1]
    #order cell names by the index of first non-zero
    cell_labels = df_ncells_edge_root.columns[1:]
    index_nonzero = []
    for i_cl,cl in enumerate(cell_labels):
        indices_nonzero_cl = np.flatnonzero(df_ncells_edge_root[cl])
        if(indices_nonzero_cl.size==0):
            index_nonzero.append(df_ncells_edge_root.shape[0])
        else:
            index_nonzero.append(indices_nonzero_cl[0])
    cell_labels_orderd = cell_labels[np.argsort(index_nonzero)[::-1]].tolist()    
    return cell_labels_orderd 


def construct_bfs_flat_tree(adata,dict_win_ncells,dict_edge_shift_dist,max_width,cell_labels_ordered,root_node=None):
    flat_tree = adata.uns['flat_tree']
    if(root_node is None):
        root_node = list(flat_tree.nodes())[0]
    dict_path_len = nx.shortest_path_length(flat_tree,source=root_node,weight='len')    
    list_bfs_edges = list(nx.bfs_edges(flat_tree,root_node))  
    bfs_flat_tree = nx.bfs_tree(flat_tree,root_node)
    
    dict_edges_ncells = dict()
    dict_edges_ncells_cumsum = dict()
    max_ncells_cumsum = 0
    for edge_i in list_bfs_edges:
        br_id = flat_tree.edges[edge_i]['id']
        if(br_id == edge_i):
            df_ncells_edge_i = dict_win_ncells[br_id] 
        else:
            df_ncells_edge_i = dict_win_ncells[br_id].copy()
            df_ncells_edge_i = df_ncells_edge_i.iloc[::-1]
            df_ncells_edge_i['pt_lam'] = df_ncells_edge_i['pt_lam'].max() - df_ncells_edge_i['pt_lam']
            df_ncells_edge_i.index = range(df_ncells_edge_i.shape[0])
        df_ncells_edge_i = df_ncells_edge_i[['pt_lam'] + cell_labels_ordered]
        dict_edges_ncells[edge_i] = df_ncells_edge_i
        #sum up cell count
        df_ncells_edge_i_cumsum = df_ncells_edge_i.copy()
        df_ncells_edge_i_cumsum.iloc[:,1:] = df_ncells_edge_i.iloc[:,1:].cumsum(axis=1) 
        dict_edges_ncells_cumsum[edge_i] = df_ncells_edge_i_cumsum
        max_ncells_cumsum = max(max_ncells_cumsum,df_ncells_edge_i_cumsum.iloc[:,1:].values.max()) 
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_ncells,name='ncells')
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_ncells_cumsum,name='ncells_cumsum')
    
    dict_edges_ncells_cumsum_norm = dict()
    dict_edges_x = dict()
    dict_edges_y_up = dict()
    dict_edges_y_down = dict()
    for edge_i in list_bfs_edges:
        df_ncells_edge_i_cumsum = bfs_flat_tree.edges[edge_i]['ncells_cumsum']
        #normalize cell count  
        df_ncells_edge_i_cumsum_norm = df_ncells_edge_i_cumsum.copy()
        df_ncells_edge_i_cumsum_norm.iloc[:,1:] = max_width*df_ncells_edge_i_cumsum.iloc[:,1:]/np.float(max_ncells_cumsum)
        dict_edges_ncells_cumsum_norm[edge_i] = df_ncells_edge_i_cumsum_norm
        node_pos_st = np.array([dict_path_len[edge_i[0]],dict_edge_shift_dist[edge_i]]) 
        dict_edges_x[edge_i] = dict_path_len[edge_i[0]] + df_ncells_edge_i_cumsum_norm['pt_lam']
        df_coord_y_up = df_ncells_edge_i_cumsum_norm.iloc[:,1:].subtract(df_ncells_edge_i_cumsum_norm.iloc[:,-1]/2.0,axis=0)
        dict_edges_y_up[edge_i] = dict_edge_shift_dist[edge_i] + df_coord_y_up
        df_coord_y_down = df_coord_y_up.copy()
        df_coord_y_down.iloc[:,1:] =  df_coord_y_up.iloc[:,:-1].values
        df_coord_y_down.iloc[:,0] = 0 - df_ncells_edge_i_cumsum_norm.iloc[:,-1]/2.0
        dict_edges_y_down[edge_i] = dict_edge_shift_dist[edge_i] + df_coord_y_down
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_ncells_cumsum_norm,name='ncells_cumsum_norm')
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_x,name='x')
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_y_up,name='y_up')
    nx.set_edge_attributes(bfs_flat_tree,values=dict_edges_y_down,name='y_down')    
    return bfs_flat_tree

#find both top and bottom outline edges starting from a specified node
def find_outline_edges(bfs_flat_tree,start_node):
    list_up_edges = list()
    list_down_edges = list()
    suc_nodes = list(bfs_flat_tree.successors(start_node))
    num_suc_nodes = len(suc_nodes)
    while(num_suc_nodes>1):
        suc_node_up = suc_nodes[0]
        list_up_edges.append((start_node,suc_node_up))
        suc_nodes = list(bfs_flat_tree.successors(suc_node_up))
        num_suc_nodes = len(suc_nodes)
    suc_nodes = list(bfs_flat_tree.successors(start_node))
    num_suc_nodes = len(suc_nodes)    
    while(num_suc_nodes>1):
        suc_node_down = suc_nodes[-1]
        list_down_edges.append((start_node,suc_node_down))
        suc_nodes = list(bfs_flat_tree.successors(suc_node_down))
        num_suc_nodes = len(suc_nodes)
    return list_up_edges,list_down_edges


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


def fill_im_array(dict_im_array,df_bins_gene,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge):
    pad_ratio = 0.008
    xmin_edge = df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].min()
    xmax_edge = df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].max()
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
        gene_color = df_bins_gene.loc[cellname,list(map(lambda x: 'win' + str(x), [id_wins_prev[-1]] + id_wins[1:]))].tolist()
    else:
        gene_color = df_bins_gene.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].tolist()
    x_axis = df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].tolist()
    x_base = np.linspace(x_axis[0],x_axis[-1],id_ed_x-id_st_x+1)
    gene_color_new = np.interp(x_base,x_axis,gene_color)
    y_axis_base = df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].tolist()
    y_axis_top = df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))].tolist()
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


def cal_stream_polygon_string(adata,dict_ann,root='S0',preference=None,dist_scale=0.9,
                              factor_num_win=10,factor_min_win=2.0,factor_width=2.5,
                              log_scale=False,factor_zoomin=100.0):
    list_ann_string = [k for k,v in dict_ann.items() if is_string_dtype(v)]
    
    flat_tree = adata.uns['flat_tree']
    label_to_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}    
    if(preference!=None):
        preference_nodes = [label_to_node[x] for x in preference]
    else:
        preference_nodes = None
    dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
    dict_node_state = nx.get_node_attributes(flat_tree,'label')

    root_node = label_to_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1]) 

    dict_verts = dict() ### coordinates of all vertices
    dict_extent = dict() ### the extent of plot
    
    df_stream = adata.obs[['branch_id','branch_lam']].copy()
    df_stream = df_stream.astype('object')
    df_stream['edge'] = ''
    df_stream['lam_ordered'] = ''
    for x in bfs_edges:
        if x in nx.get_edge_attributes(flat_tree,'id').values():
            id_cells = np.where(df_stream['branch_id']==x)[0]
            df_stream.loc[df_stream.index[id_cells],'edge'] = [x]
            df_stream.loc[df_stream.index[id_cells],'lam_ordered'] = df_stream.loc[df_stream.index[id_cells],'branch_lam']
        else:
            id_cells = np.where(df_stream['branch_id']==(x[1],x[0]))[0]
            df_stream.loc[df_stream.index[id_cells],'edge'] = [x]
            df_stream.loc[df_stream.index[id_cells],'lam_ordered'] = flat_tree.edges[x]['len'] - df_stream.loc[df_stream.index[id_cells],'branch_lam']  
    for ann in list_ann_string:
        df_stream['CELL_LABEL'] = dict_ann[ann]
        len_ori = {}
        for x in bfs_edges:
            if(x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1],x[0])]['len']        

        dict_tree = {}
        bfs_prev = dict(nx.bfs_predecessors(flat_tree,root_node))
        bfs_next = dict(nx.bfs_successors(flat_tree,root_node))
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
        dfs_nodes = dfs_nodes_modified(flat_tree,root_node,preference=preference_nodes)
        leaves=[n for n,d in flat_tree.degree() if d==1]
        id_leaf = 0
        dfs_nodes_copy = deepcopy(dfs_nodes)
        num_nonroot_leaf = len(list(set(leaves) - set([root_node])))
        while len(dfs_nodes_copy)>1:
            node = dfs_nodes_copy.pop()
            prev_node = dict_tree[node]['prev']
            if(node in leaves):
                dict_shift_dist[(prev_node,node)] = -(np.float(1)/dist_scale)*(num_nonroot_leaf-1)/2.0 + id_leaf*(np.float(1)/dist_scale)
                id_leaf = id_leaf+1
            else:
                next_nodes = dict_tree[node]['next']
                dict_shift_dist[(prev_node,node)] = (sum([dict_shift_dist[(node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))
        if (flat_tree.degree(root_node))>1:
            next_nodes = dict_tree[root_node]['next']
            dict_shift_dist[(root_node,root_node)] = (sum([dict_shift_dist[(root_node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))


        #dataframe of bins
        df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary','center','edge'])    
        list_paths = find_root_to_leaf_paths(flat_tree, root_node)
        max_path_len = find_longest_path(list_paths,len_ori)
        size_w = max_path_len/np.float(factor_num_win)
        if(size_w>min(len_ori.values())/np.float(factor_min_win)):
            size_w = min(len_ori.values())/np.float(factor_min_win)

        step_w = size_w/2 #step of sliding window (the divisor should be even)

        if(len(dict_shift_dist)>1):
            max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
        else:
            max_width = max_path_len/np.float(factor_width)
        # max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
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
                    df_bins.loc['edge',"win"+str(total_bins+i_win)] = [(root_node,root_node)]
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
        if(log_scale):
            df_n_cells= df_bins_ori.iloc[:-3,:].sum()
            df_n_cells = df_n_cells/df_n_cells.max()*factor_zoomin
            df_bins_ori.iloc[:-3,:] = df_bins_ori.iloc[:-3,:]*np.log2(df_n_cells+1)/(df_n_cells+1) 

        df_bins_cumsum = df_bins_ori.copy()
        df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

        #normalization  
        df_bins_cumsum_norm = df_bins_cumsum.copy()
        df_bins_cumsum_norm.iloc[:-3,:] = min_width + max_width*(df_bins_cumsum.iloc[:-3,:])/(df_bins_cumsum.iloc[:-3,:]).values.max()

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
                if node_i == root_node:
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
                if(flat_tree.degree(root_node)>1):
                    id_wins = id_wins[1:]
            else:
                id_wins = id_wins[1:] # remove the overlapped window
                x_st = dict_ep_center[(prev_node,edge_i[0])][0] - step_w
            y_st = dict_shift_dist[edge_i]
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',list(map(lambda x: 'win' + str(x), id_wins))]
                py_top = df_bins_top.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))]
                px_top_prime = x_st  + px_top
                py_top_prime = y_st  + py_top
                dict_ep_top[cellname][edge_i] = [px_top_prime[-1],py_top_prime[-1]]
                df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = px_top_prime
                df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',list(map(lambda x: 'win' + str(x), id_wins))]
                py_base = df_bins_base.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))]
                px_base_prime = x_st + px_base
                py_base_prime = y_st + py_base
                dict_ep_base[cellname][edge_i] = [px_base_prime[-1],py_base_prime[-1]]
                df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = px_base_prime
                df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = py_base_prime
            dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

        id_wins_start = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==(root_node,root_node)]
        if(len(id_wins_start)>0):
            mean_shift_dist = np.mean([dict_shift_dist[(root_node,x)] \
                                    for x in dict_forest[cell_list_sorted[0]][root_node]['next']])
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center',list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_top = df_bins_top.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_top_prime = 0  + px_top
                py_top_prime = mean_shift_dist  + py_top
                df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = px_top_prime
                df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center',list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_base = df_bins_base.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_base_prime = 0 + px_base
                py_base_prime = mean_shift_dist + py_base
                df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = px_base_prime
                df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = py_base_prime

        #determine joints points
        dict_joint_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        dict_joint_base = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
        if(flat_tree.degree(root_node)==1):
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1]
        else:
            id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1 and x[0]!=(root_node,root_node)]
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
            if(flat_tree.degree(root_node)>1 and \
               edge_i_top==(root_node,dict_forest[cell_list_sorted[0]][root_node]['next'][0])):
                id_wins_top.insert(0,1)
                id_wins_top.insert(0,0)
            for cellname in cell_list_sorted:
                x_top = df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
                y_top = df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
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
            if(flat_tree.degree(root_node)>1 and \
               edge_i_base==(root_node,dict_forest[cell_list_sorted[0]][root_node]['next'][-1])):
                id_wins_base.insert(0,1)
                id_wins_base.insert(0,0)
            for cellname in cell_list_sorted:
                x_base = df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
                y_base = df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
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
        cur_node = root_node
        next_node = dict_tree_copy[cur_node]['next'][0]
        dict_tree_copy[cur_node]['next'].pop(0)
        poly_edges.append((cur_node,next_node))
        cur_node = next_node
        while(not(next_node==root_node and cur_node == dict_tree[root_node]['next'][-1])):
            while(len(dict_tree_copy[cur_node]['next'])!=0):
                next_node = dict_tree_copy[cur_node]['next'][0]
                dict_tree_copy[cur_node]['next'].pop(0)
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[root_node]['next'][-1] and next_node==root_node):
                    break
                cur_node = next_node
            while(len(dict_tree_copy[cur_node]['next'])==0):
                next_node = dict_tree_copy[cur_node]['prev']
                poly_edges.append((cur_node,next_node))
                if(cur_node == dict_tree[root_node]['next'][-1] and next_node==root_node):
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
        dict_verts[ann] = verts
        
        extent = {'xmin':"",'xmax':"",'ymin':"",'ymax':""}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
                xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
                ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
                ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
                if(extent['xmin']==""):
                    extent['xmin'] = xmin
                else:
                    if(xmin < extent['xmin']) :
                        extent['xmin'] = xmin

                if(extent['xmax']==""):
                    extent['xmax'] = xmax
                else:
                    if(xmax > extent['xmax']):
                        extent['xmax'] = xmax

                if(extent['ymin']==""):
                    extent['ymin'] = ymin
                else:
                    if(ymin < extent['ymin']):
                        extent['ymin'] = ymin

                if(extent['ymax']==""):
                    extent['ymax'] = ymax
                else:
                    if(ymax > extent['ymax']):
                        extent['ymax'] = ymax
        dict_extent[ann] = extent
    return dict_verts,dict_extent

def cal_stream_polygon_numeric(adata,dict_ann,root='S0',preference=None, dist_scale=0.9,
                               factor_num_win=10,factor_min_win=2.0,factor_width=2.5,
                               factor_nrow=200,factor_ncol=400,
                               log_scale=False,factor_zoomin=100.0):
    list_ann_numeric = [k for k,v in dict_ann.items() if is_numeric_dtype(v)]

    flat_tree = adata.uns['flat_tree']
    label_to_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}    
    if(preference!=None):
        preference_nodes = [label_to_node[x] for x in preference]
    else:
        preference_nodes = None
    dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
    dict_node_state = nx.get_node_attributes(flat_tree,'label')

    root_node = label_to_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
    bfs_nodes = []
    for x in bfs_edges:
        if x[0] not in bfs_nodes:
            bfs_nodes.append(x[0])
        if x[1] not in bfs_nodes:
            bfs_nodes.append(x[1])    
    
    df_stream = adata.obs[['branch_id','branch_lam']].copy()
    df_stream = df_stream.astype('object')
    df_stream['edge'] = ''
    df_stream['lam_ordered'] = ''
    for x in bfs_edges:
        if x in nx.get_edge_attributes(flat_tree,'id').values():
            id_cells = np.where(df_stream['branch_id']==x)[0]
            df_stream.loc[df_stream.index[id_cells],'edge'] = [x]
            df_stream.loc[df_stream.index[id_cells],'lam_ordered'] = df_stream.loc[df_stream.index[id_cells],'branch_lam']
        else:
            id_cells = np.where(df_stream['branch_id']==(x[1],x[0]))[0]
            df_stream.loc[df_stream.index[id_cells],'edge'] = [x]
            df_stream.loc[df_stream.index[id_cells],'lam_ordered'] = flat_tree.edges[x]['len'] - df_stream.loc[df_stream.index[id_cells],'branch_lam']  
    
    df_stream['CELL_LABEL'] = 'unknown'
    for ann in list_ann_numeric:
        df_stream[ann] = dict_ann[ann]

    len_ori = {}
    for x in bfs_edges:
        if(x in dict_branches.keys()):
            len_ori[x] = dict_branches[x]['len']
        else:
            len_ori[x] = dict_branches[(x[1],x[0])]['len']        

    dict_tree = {}
    bfs_prev = dict(nx.bfs_predecessors(flat_tree,root_node))
    bfs_next = dict(nx.bfs_successors(flat_tree,root_node))
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
    dfs_nodes = dfs_nodes_modified(flat_tree,root_node,preference=preference_nodes)
    leaves=[n for n,d in flat_tree.degree() if d==1]
    id_leaf = 0
    dfs_nodes_copy = deepcopy(dfs_nodes)
    num_nonroot_leaf = len(list(set(leaves) - set([root_node])))
    while len(dfs_nodes_copy)>1:
        node = dfs_nodes_copy.pop()
        prev_node = dict_tree[node]['prev']
        if(node in leaves):
            dict_shift_dist[(prev_node,node)] = -(np.float(1)/dist_scale)*(num_nonroot_leaf-1)/2.0 + id_leaf*(np.float(1)/dist_scale)
            id_leaf = id_leaf+1
        else:
            next_nodes = dict_tree[node]['next']
            dict_shift_dist[(prev_node,node)] = (sum([dict_shift_dist[(node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))
    if (flat_tree.degree(root_node))>1:
        next_nodes = dict_tree[root_node]['next']
        dict_shift_dist[(root_node,root_node)] = (sum([dict_shift_dist[(root_node,next_node)] for next_node in next_nodes]))/float(len(next_nodes))


    #dataframe of bins
    df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary','center','edge'])
    dict_ann_df = {ann: pd.DataFrame(index=list(df_stream['CELL_LABEL'].unique())) for ann in list_ann_numeric}
    dict_merge_num = {ann:[] for ann in list_ann_numeric} #number of merged sliding windows          
    list_paths = find_root_to_leaf_paths(flat_tree, root_node)
    max_path_len = find_longest_path(list_paths,len_ori)
    size_w = max_path_len/np.float(factor_num_win)
    if(size_w>min(len_ori.values())/np.float(factor_min_win)):
        size_w = min(len_ori.values())/np.float(factor_min_win)

    step_w = size_w/2 #step of sliding window (the divisor should be even)    
    if(len(dict_shift_dist)>1):
        max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
    else:
        max_width = max_path_len/np.float(factor_width)
    # max_width = (max_path_len/np.float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
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
                df_bins.loc['edge',"win"+str(total_bins+i_win)] = [(root_node,root_node)]
                dict_df_ann_common = dict()
                for ann in list_ann_numeric:
                    dict_df_ann_common[ann] = list()
                for j in range(degree_st):
                    df_edge_j = dict_edge_filter[(edge_i[0],nb_nodes[j])]
                    cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                    df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                    df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                    for ann in list_ann_numeric:
                        dict_df_ann_common[ann].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
    #                     ann_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
    #                                                             df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[ann].mean()
    #                     dict_ann_df[ann].ix[ann_values_common2.index,"win"+str(total_bins+i_win)] = \
    #                     dict_ann_df[ann].ix[ann_values_common2.index,"win"+str(total_bins+i_win)] + ann_values_common2
                    df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[0],nb_nodes[j]))
                for ann in list_ann_numeric:
                    ann_values_common = pd.concat(dict_df_ann_common[ann]).groupby(['CELL_LABEL'])[ann].mean()
                    dict_ann_df[ann].loc[ann_values_common.index,"win"+str(total_bins+i_win)] = ann_values_common
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
                for ann in list_ann_numeric:
                    dict_ann_df[ann]["win"+str(total_bins+i_win)] = 0
                    ann_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=mat_w[i_win,0],\
                                                    df_edge_i.lam_ordered<=mat_w[i_win,1])].groupby(['CELL_LABEL'])[ann].mean()
                    dict_ann_df[ann].loc[ann_values.index,"win"+str(total_bins+i_win)] = ann_values
                    dict_merge_num[ann].append(1)
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
                    for ann in list_ann_numeric:
                        dict_ann_df[ann]["win"+str(total_bins)] = 0
                        ann_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered>=bd_bins[0],\
                                                        df_edge_i.lam_ordered<=bd_bins[1])].groupby(['CELL_LABEL'])[ann].mean()
                        dict_ann_df[ann].loc[ann_values.index,"win"+str(total_bins)] = ann_values
                        dict_merge_num[ann].append(len(id_stack))
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
                    dict_df_ann_common = dict()
                    for ann in list_ann_numeric:
                        dict_ann_df[ann]["win"+str(total_bins+i_win)] = 0
                        dict_df_ann_common[ann] = list()
                        dict_df_ann_common[ann].append(df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
                                                                df_edge_i.lam_ordered<=len_ori[edge_i])])
    #                     ann_values_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered>mat_w_common[i_win,0],\
    #                                                             df_edge_i.lam_ordered<=len_ori[edge_i])].groupby(['CELL_LABEL'])[ann].mean()
    #                     dict_ann_df[ann].ix[ann_values_common1.index,"win"+str(total_bins+i_win)] = ann_values_common1
                        dict_merge_num[ann].append(1)
                    df_bins.loc['edge',"win"+str(total_bins+i_win)] = [edge_i]
                    for j in range(degree_end - 1):
                        df_edge_j = dict_edge_filter[(edge_i[1],nb_nodes[j])]
                        cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])]['CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] = \
                        df_bins.loc[cell_num_common2.index,"win"+str(total_bins+i_win)] + cell_num_common2
                        for ann in list_ann_numeric:
                            dict_df_ann_common[ann].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
                                                                    df_edge_j.lam_ordered<=mat_w_common[i_win,1])])
    #                         ann_values_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered>=0,\
    #                                                                 df_edge_j.lam_ordered<=mat_w_common[i_win,1])].groupby(['CELL_LABEL'])[ann].mean()
    #                         dict_ann_df[ann].ix[ann_values_common2.index,"win"+str(total_bins+i_win)] = \
    #                         dict_ann_df[ann].ix[ann_values_common2.index,"win"+str(total_bins+i_win)] + ann_values_common2
                        if abs(((sum(mat_w_common[i_win,:])+len_ori[edge_i])/2)-(len_ori[edge_i]+size_w/2.0))< step_w/100.0:
                            df_bins.loc['edge',"win"+str(total_bins+i_win)].append((edge_i[1],nb_nodes[j]))
                    for ann in list_ann_numeric:
                        ann_values_common = pd.concat(dict_df_ann_common[ann]).groupby(['CELL_LABEL'])[ann].mean()
                        dict_ann_df[ann].loc[ann_values_common.index,"win"+str(total_bins+i_win)] = ann_values_common
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

    for ann in list_ann_numeric:
        dict_ann_df[ann] = dict_ann_df[ann].reindex(cell_list_sorted)    

    #original count
    df_bins_ori = df_bins.reindex(cell_list_sorted+['boundary','center','edge'])
    if(log_scale):
        df_n_cells= df_bins_ori.iloc[:-3,:].sum()
        df_n_cells = df_n_cells/df_n_cells.max()*factor_zoomin
        df_bins_ori.iloc[:-3,:] = df_bins_ori.iloc[:-3,:]*np.log2(df_n_cells+1)/(df_n_cells+1) 

    df_bins_cumsum = df_bins_ori.copy()
    df_bins_cumsum.iloc[:-3,:] = df_bins_ori.iloc[:-3,:][::-1].cumsum()[::-1]

    #normalization  
    df_bins_cumsum_norm = df_bins_cumsum.copy()
    df_bins_cumsum_norm.iloc[:-3,:] = min_width + max_width*(df_bins_cumsum.iloc[:-3,:])/(df_bins_cumsum.iloc[:-3,:]).values.max()

    df_bins_top = df_bins_cumsum_norm.copy()
    df_bins_top.iloc[:-3,:] = df_bins_cumsum_norm.iloc[:-3,:].subtract(df_bins_cumsum_norm.iloc[0,:]/2.0)
    df_bins_base = df_bins_top.copy()
    df_bins_base.iloc[:-4,:] = df_bins_top.iloc[1:-3,:].values
    df_bins_base.iloc[-4,:] = 0-df_bins_cumsum_norm.iloc[0,:]/2.0

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
            if node_i == root_node:
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
            if(flat_tree.degree(root_node)>1):
                id_wins = id_wins[1:]
        else:
            id_wins = id_wins[1:] # remove the overlapped window
            x_st = dict_ep_center[(prev_node,edge_i[0])][0] - step_w
        y_st = dict_shift_dist[edge_i]
        for cellname in cell_list_sorted:
            ##top line
            px_top = df_bins_top.loc['center',list(map(lambda x: 'win' + str(x), id_wins))]
            py_top = df_bins_top.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))]
            px_top_prime = x_st  + px_top
            py_top_prime = y_st  + py_top
            dict_ep_top[cellname][edge_i] = [px_top_prime[-1],py_top_prime[-1]]
            df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = px_top_prime
            df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = py_top_prime
            ##base line
            px_base = df_bins_base.loc['center',list(map(lambda x: 'win' + str(x), id_wins))]
            py_base = df_bins_base.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))]
            px_base_prime = x_st + px_base
            py_base_prime = y_st + py_base
            dict_ep_base[cellname][edge_i] = [px_base_prime[-1],py_base_prime[-1]]
            df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = px_base_prime
            df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins))] = py_base_prime
        dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

    id_wins_start = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==(root_node,root_node)]
    if(len(id_wins_start)>0):
        mean_shift_dist = np.mean([dict_shift_dist[(root_node,x)] \
                                for x in dict_forest[cell_list_sorted[0]][root_node]['next']])
        for cellname in cell_list_sorted:
            ##top line
            px_top = df_bins_top.loc['center',list(map(lambda x: 'win' + str(x), id_wins_start))]
            py_top = df_bins_top.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))]
            px_top_prime = 0  + px_top
            py_top_prime = mean_shift_dist  + py_top
            df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = px_top_prime
            df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = py_top_prime
            ##base line
            px_base = df_bins_base.loc['center',list(map(lambda x: 'win' + str(x), id_wins_start))]
            py_base = df_bins_base.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))]
            px_base_prime = 0 + px_base
            py_base_prime = mean_shift_dist + py_base
            df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = px_base_prime
            df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_start))] = py_base_prime

    #determine joints points
    dict_joint_top = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
    dict_joint_base = {cellname:dict() for cellname in cell_list_sorted} #coordinates of joint points
    if(flat_tree.degree(root_node)==1):
        id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1]
    else:
        id_joints = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if len(x)>1 and x[0]!=(root_node,root_node)]
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
        if(flat_tree.degree(root_node)>1 and \
           edge_i_top==(root_node,dict_forest[cell_list_sorted[0]][root_node]['next'][0])):
            id_wins_top.insert(0,1)
            id_wins_top.insert(0,0)
        for cellname in cell_list_sorted:
            x_top = df_top_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
            y_top = df_top_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
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
        if(flat_tree.degree(root_node)>1 and \
           edge_i_base==(root_node,dict_forest[cell_list_sorted[0]][root_node]['next'][-1])):
            id_wins_base.insert(0,1)
            id_wins_base.insert(0,0)
        for cellname in cell_list_sorted:
            x_base = df_base_x.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
            y_base = df_base_y.loc[cellname,list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
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
    cur_node = root_node
    next_node = dict_tree_copy[cur_node]['next'][0]
    dict_tree_copy[cur_node]['next'].pop(0)
    poly_edges.append((cur_node,next_node))
    cur_node = next_node
    while(not(next_node==root_node and cur_node == dict_tree[root_node]['next'][-1])):
        while(len(dict_tree_copy[cur_node]['next'])!=0):
            next_node = dict_tree_copy[cur_node]['next'][0]
            dict_tree_copy[cur_node]['next'].pop(0)
            poly_edges.append((cur_node,next_node))
            if(cur_node == dict_tree[root_node]['next'][-1] and next_node==root_node):
                break
            cur_node = next_node
        while(len(dict_tree_copy[cur_node]['next'])==0):
            next_node = dict_tree_copy[cur_node]['prev']
            poly_edges.append((cur_node,next_node))
            if(cur_node == dict_tree[root_node]['next'][-1] and next_node==root_node):
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

    extent = {'xmin':"",'xmax':"",'ymin':"",'ymax':""}
    for cellname in cell_list_sorted:
        for edge_i in bfs_edges:
            xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
            xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
            ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
            ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
            if(extent['xmin']==""):
                extent['xmin'] = xmin
            else:
                if(xmin < extent['xmin']) :
                    extent['xmin'] = xmin

            if(extent['xmax']==""):
                extent['xmax'] = xmax
            else:
                if(xmax > extent['xmax']):
                    extent['xmax'] = xmax

            if(extent['ymin']==""):
                extent['ymin'] = ymin
            else:
                if(ymin < extent['ymin']):
                    extent['ymin'] = ymin

            if(extent['ymax']==""):
                extent['ymax'] = ymax
            else:
                if(ymax > extent['ymax']):
                    extent['ymax'] = ymax
    dict_im_array = dict()
    for ann in list_ann_numeric:                    
        im_nrow = factor_nrow
        im_ncol = factor_ncol
        xmin = extent['xmin']
        xmax = extent['xmax']
        ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
        ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1
        im_array = {cellname: np.zeros((im_nrow,im_ncol)) for cellname in cell_list_sorted}
        df_bins_ann = dict_ann_df[ann]
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                id_wins_all = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==edge_i]
                prev_edge = ''
                id_wins_prev = []
                if(flat_tree.degree(root_node)>1):
                    if(edge_i == bfs_edges[0]):
                        id_wins = [0,1]
                        im_array = fill_im_array(im_array,df_bins_ann,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)
                    id_wins = id_wins_all
                    if(edge_i[0] == root_node):
                        prev_edge = (root_node,root_node)
                        id_wins_prev = [0,1]
                    else:
                        prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                        id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                    im_array = fill_im_array(im_array,df_bins_ann,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)
                else:
                    id_wins = id_wins_all
                    if(edge_i[0]!=root_node):
                        prev_edge = (dict_tree[edge_i[0]]['prev'],edge_i[0])
                        id_wins_prev = [i for i,x in enumerate(df_bins_cumsum_norm.loc['edge',:]) if x[0]==prev_edge]
                    im_array = fill_im_array(im_array,df_bins_ann,flat_tree,df_base_x,df_base_y,df_top_x,df_top_y,xmin,xmax,ymin,ymax,im_nrow,im_ncol,step_w,dict_shift_dist,id_wins,edge_i,cellname,id_wins_prev,prev_edge)
        dict_im_array[ann] = im_array
    return verts,extent,cell_list_sorted,dict_ann_df,dict_im_array


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

def scale_marker_expr(params):
    df_marker_detection = params[0]
    marker = params[1]
    percentile_expr = params[2]
    marker_values = df_marker_detection[marker].copy()
    if(min(marker_values)<0):
        min_marker_values = np.percentile(marker_values[marker_values<0],100-percentile_expr)
        marker_values[marker_values<min_marker_values] = min_marker_values
        max_marker_values = np.percentile(marker_values[marker_values>0],percentile_expr)
        marker_values[marker_values>max_marker_values] = max_marker_values
        marker_values = marker_values - min(marker_values)
    else:
        max_marker_values = np.percentile(marker_values[marker_values>0],percentile_expr)
        marker_values[marker_values>max_marker_values] = max_marker_values
    marker_values = marker_values/max_marker_values
    return marker_values

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

def get_colors(adata,ann):
    df_cell_colors = pd.DataFrame(index=adata.obs.index)
    if(is_numeric_dtype(adata.obs[ann])):
        cm = mpl.cm.get_cmap()
        norm = mpl.colors.Normalize(vmin=0, vmax=max(adata.obs[ann]),clip=True)
        df_cell_colors[ann+'_color'] = [mpl.colors.to_hex(cm(norm(x))) for x in adata.obs[ann]]
    else:
        if(ann+'_color' not in adata.uns_keys()):  
            ### a hacky way to generate colors from seaborn
            tmp = pd.DataFrame(index=adata.obs_names,
                   data=np.random.rand(adata.shape[0], 2))
            tmp[ann] = adata.obs[ann]
            fig = plt.figure()
            ax_i = fig.add_subplot(1,1,1)
            sc_i=sns.scatterplot(ax=ax_i,x=0, y=1,hue=ann,data=tmp,linewidth=0)             
            colors_sns = sc_i.get_children()[0].get_facecolors()
            colors_sns_scaled = (255*colors_sns).astype(int)
            ax_i.remove()
            adata.uns[ann+'_color'] = {tmp[ann][i]:'#%02x%02x%02x' % (colors_sns_scaled[i][0], colors_sns_scaled[i][1], colors_sns_scaled[i][2])
                                       for i in np.unique(tmp[ann],return_index=True)[1]}            
        dict_color = adata.uns[ann+'_color']
        df_cell_colors[ann+'_color'] = ''
        for x in dict_color.keys():
            id_cells = np.where(adata.obs[ann]==x)[0]
            df_cell_colors.loc[df_cell_colors.index[id_cells],ann+'_color'] = dict_color[x]
    return(df_cell_colors[ann+'_color'].tolist())