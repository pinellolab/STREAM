[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/stream/README.html)

[![Build Status](https://travis-ci.org/pinellolab/STREAM.svg)](https://travis-ci.org/pinellolab/STREAM)

# STREAM (Latest version v1.0)

Latest News
-----------
> Jun 1, 2020  

Version 1.0 is now available. The v1.0 has added a lot of new functionality:
1) added QC metrics and plots
2) added support of scATAC-seq analysis using peaks as features
3) added support of interactive plots with plotly
4) redesigned all plotting-related functions
5) redesigned *mapping* procedure
6) removed support of STREAM command line interface  

See [v1.0](https://github.com/pinellolab/STREAM/releases/tag/v1.0) for more details.


> Jan 14, 2020

Version 0.4.1 is now available. We added support of feature `top_pcs` for *Mapping*

> Nov 26, 2019

Version 0.4.0 is now available. Numerous changes have been introduced. Please check [v0.4.0](https://github.com/pinellolab/STREAM/releases/tag/v0.4.0) for details.

Introduction
------------

STREAM (**S**ingle-cell **T**rajectories **R**econstruction, **E**xploration **A**nd **M**apping) is an interactive pipeline capable of disentangling and visualizing complex branching trajectories from both single-cell transcriptomic and epigenomic data.

STREAM is now published in *Nature Communications*! Please cite our paper [Chen H, et al. Single-cell trajectories reconstruction, exploration and mapping of omics data with STREAM.](https://www.nature.com/articles/s41467-019-09670-4)  *Nature Communications*, volume 10, Article number: 1903 (2019). if you find STREAM helpful for your research.

<img src="https://github.com/pinellolab/STREAM/blob/stream_python2/STREAM/static/images/Figure1.png">

STREAM is written using the class `anndata` [Wolf et al. Genome Biology (2018)](http://anndata.rtfd.io) and available as user-friendly open source software and can be used interactively as a web-application at [stream.pinellolab.org](http://stream.pinellolab.org/), as a bioconda package [https://bioconda.github.io/recipes/stream/README.html](https://bioconda.github.io/recipes/stream/README.html) and as a standalone command-line tool with Docker [https://github.com/pinellolab/STREAM](https://github.com/pinellolab/STREAM)

Installation with Bioconda (Recommended)
----------------------------------------
```sh
$ conda install -c bioconda stream
```

If you are new to conda environment:

1)	If Anaconda (or miniconda) is already installed with **Python 3**, skip to 2) otherwise please download and install Python3 Anaconda from here: https://www.anaconda.com/download/

2)	Open a terminal and add the Bioconda channel with the following commands:

```sh
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

3)	Create an environment named `env_stream` , install **stream**, **jupyter**, and activate it with the following commands:

* *For single cell **RNA-seq** analysis*:
```sh
$ conda create -n env_stream python stream=1.0 jupyter
$ conda activate env_stream
```
* *For single cell **ATAC-seq** analysis*:
```sh
$ conda create -n env_stream python stream=1.0 stream_atac jupyter
$ conda activate env_stream
```

4)  To perform STREAM analyis in Jupyter Notebook as shown in **Tutorial**, type `jupyter notebook` within `env_stream`:

```sh
$ jupyter notebook
```

You should see the notebook open in your browser.

Tutorial
--------

* Example for scRNA-seq: [1.1-STREAM_scRNA-seq (Bifurcation).ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.1.STREAM_scRNA-seq%20%28Bifurcation%29.ipynb?flush_cache=true)

* Example for scRNA-seq: [1.2-STREAM_scRNA-seq (Multifurcation) on 2D visulization.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.2.use_vis.ipynb?flush_cache=true)

* Example for scRNA-seq: [1.3-STREAM_scRNA-seq (Multifurcation) on original embedding.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/1.2.STREAM_scRNA-seq%20%28Multifurcation%29.ipynb?flush_cache=true)

* Example for scATAC-seq(using peaks): [2.1-STREAM_scATAC-seq_peaks.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.1-STREAM_scATAC-seq_peaks.ipynb?flush_cache=true)

* Example for scATAC-seq(using k-mers): [2.2-STREAM_scATAC-seq_k-mers.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.2.STREAM_scATAC-seq_k-mers.ipynb?flush_cache=true)

* Example for scATAC-seq(using motifs): [2.3-STREAM_scATAC-seq_motifs.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/2.3.STREAM_scATAC-seq_motifs.ipynb?flush_cache=true)

* Example for *mapping* feature: [3-STREAM_mapping.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/3.STREAM_mapping.ipynb?flush_cache=true)

* Example for complex trajectories: [4-STREAM_complex_trajectories.ipynb](https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/4.STREAM_complex_trajectories.ipynb?flush_cache=true)

Tutorials for v0.4.1 and earlier versions can be found [here](https://github.com/pinellolab/STREAM/tree/master/tutorial/archives/v0.4.1_and_earlier_versions)

Installation with Docker
------------------------

With Docker no installation is required, the only dependence is Docker itself. Users will completely get rid of all the installation and configuration issues. Docker will do all the dirty work for you!

Docker can be downloaded freely from here: [https://store.docker.com/search?offering=community&type=edition](https://store.docker.com/search?offering=community&type=edition)

To get an image of STREAM, simply execute the following command:

```sh
$ docker pull pinellolab/stream
```

>Basic usage of *docker run* 
>```sh
>$ docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
> ```
>Options:  
>```
>--publish , -p	Publish a container’s port(s) to the host  
>--volume , -v	Bind mount a volume  
>--workdir , -w	Working directory inside the container  
>```

To use STREAM inside the docker container:
* Mount your data folder and enter STREAM docker container:

```bash
$ docker run --entrypoint /bin/bash -v /your/data/file/path/:/data -w /data -p 8888:8888 -it pinellolab/stream:1.0
```
* Inside the container, launch Jupyter notebook:
```
root@46e09702ce87:/data# jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
Access the notebook through your desktops browser on http://127.0.0.1:8888. The notebook will prompt you for a token which was generated when you create the notebook.


STREAM interactive website
--------------------------

In order to make STREAM user friendly and accessible to non-bioinformatician, we have created an interactive website: [http://stream.pinellolab.org](https://stream.pinellolab.partners.org/)

The website can also run on a local machine. More details can be found [https://github.com/pinellolab/STREAM_web](https://github.com/pinellolab/STREAM_web)



Credits: H Chen, L Albergante, JY Hsu, CA Lareau, GL Bosco, J Guan, S Zhou, AN Gorban, DE Bauer, MJ Aryee, DM Langenau, A Zinovyev, JD Buenrostro, GC Yuan, L Pinello
