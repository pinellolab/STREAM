import stream as st
import os 
from pathlib import Path
import tarfile
import tempfile
from shutil import rmtree
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

_root = os.path.abspath(os.path.dirname(__file__))

def stream_test_Nestorowa_2016():

	workdir = os.path.join(_root,'datasets/Nestorowa_2016/')

	temp_folder=tempfile.gettempdir()

	tar = tarfile.open(workdir+'output/stream_result.tar.gz')
	tar.extractall(path=temp_folder)
	tar.close()
	ref_temp_folder=os.path.join(temp_folder,'stream_result')

											 
	print(workdir+'data_Nestorowa.tsv.gz')
	input_file=os.path.join(workdir,'data_Nestorowa.tsv.gz')
	label_file=os.path.join(workdir,'cell_label.tsv.gz')
	label_color_file=os.path.join(workdir,'cell_label_color.tsv.gz')
	comp_temp_folder = os.path.join(temp_folder,'stream_result_comp')

	try:
		st.set_figure_params(dpi=80,style='white',figsize=[5.4,4.8],
	                     rc={'image.cmap': 'viridis'})
		adata=st.read(file_name=input_file,workdir=comp_temp_folder)
		adata.var_names_make_unique()
		adata.obs_names_make_unique()
		st.add_cell_labels(adata,file_name=label_file)
		st.add_cell_colors(adata,file_name=label_color_file)
		st.cal_qc(adata,assay='rna')
		st.filter_features(adata,min_n_cells = 5)
		st.select_variable_genes(adata,loess_frac=0.1,save_fig=True)
		st.dimension_reduction(adata,method='se',n_neighbors=50,n_components=4,n_jobs=2)
		st.plot_dimension_reduction(adata,color=['label','Gata1','n_genes'],n_components=3,show_graph=False,show_text=False,save_fig=True,fig_name='dimension_reduction.pdf')
		st.plot_visualization_2D(adata,method='umap',n_neighbors=50,color=['label','Gata1','n_genes'],use_precomputed=False,save_fig=True,fig_name='visualization_2D.pdf')
		st.seed_elastic_principal_graph(adata,n_clusters=10,use_vis=True)
		st.plot_dimension_reduction(adata,color=['label','Gata1','n_genes'],n_components=2,show_graph=True,show_text=False,save_fig=True,fig_name='dr_seed.pdf')
		st.plot_branches(adata,show_text=True,save_fig=True,fig_name='branches_seed.pdf')	
		st.elastic_principal_graph(adata,epg_alpha=0.01,epg_mu=0.05,epg_lambda=0.01)
		st.plot_dimension_reduction(adata,color=['label','Gata1','n_genes'],n_components=2,show_graph=True,show_text=False,save_fig=True,fig_name='dr_epg.pdf')
		st.plot_branches(adata,show_text=True,save_fig=True,fig_name='branches_epg.pdf')
		###Extend leaf branch to reach further cells 
		st.extend_elastic_principal_graph(adata, epg_ext_mode='WeigthedCentroid',epg_ext_par=0.8)
		st.plot_dimension_reduction(adata,color=['label'],n_components=2,show_graph=True,show_text=True,save_fig=True,fig_name='dr_extend.pdf')
		st.plot_branches(adata,show_text=True,save_fig=True,fig_name='branches_extend.pdf')
		st.plot_flat_tree(adata,color=['label','branch_id_alias','S4_pseudotime'],dist_scale=0.5,show_graph=True,show_text=True,save_fig=True)
		st.plot_stream_sc(adata,root='S4',color=['label','Gata1'],dist_scale=0.5,show_graph=True,show_text=False,save_fig=True)
		st.plot_stream(adata,root='S4',color=['label','Gata1'],save_fig=True)
		st.detect_leaf_markers(adata,marker_list=adata.uns['var_genes'],root='S4',n_jobs=4)
		st.detect_transition_markers(adata,root='S4',marker_list=adata.uns['var_genes'],n_jobs=4)
		st.detect_de_markers(adata,marker_list=adata.uns['var_genes'],root='S4',n_jobs=4)
	except:
		print("STREAM analysis failed!")
		raise
	else:
		print("STREAM analysis finished!")

	print(ref_temp_folder)
	print(comp_temp_folder)

	pathlist = Path(ref_temp_folder)
	for path in pathlist.glob('**/*'):
		if path.is_file() and (not path.name.startswith('.')):
			file = os.path.relpath(str(path),ref_temp_folder)
			print(file)
			if(file.endswith('pdf')):
				if(os.path.getsize(os.path.join(comp_temp_folder,file))>0):
					print('The file %s passed' %file)
				else:
					raise Exception('Error! The file %s is not matched' %file)
			else:
				checklist = list()
				df_ref = pd.read_csv(os.path.join(ref_temp_folder,file),sep='\t')
				df_comp = pd.read_csv(os.path.join(comp_temp_folder,file),sep='\t')
				for c in df_ref.columns:
					if(is_numeric_dtype(df_ref[c])):
						checklist.append(all(np.isclose(df_ref[c],df_comp[c])))
					else:
						print(c)
						checklist.append(all(df_ref[c]==df_comp[c]))
				if(all(checklist)):
					print('The file %s passed' %file)
				else:
					raise Exception('Error! The file %s is not matched' %file)

	print('Successful!')

	rmtree(comp_temp_folder,ignore_errors=True)
	rmtree(ref_temp_folder,ignore_errors=True)


def main():
	stream_test_Nestorowa_2016()
	

if __name__ == "__main__":
    main()