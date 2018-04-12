############################################################
# Dockerfile to build STREAM & webapp
############################################################

# Set the base image to anaconda python 2.7
FROM continuumio/anaconda

# File Author / Maintainer
MAINTAINER Luca Pinello

ENV SHELL bash

RUN conda install r-base
RUN conda config --add channels defaults
RUN conda config --add channels conda-forge
RUN conda config --add channels bioconda

#Add build tools
RUN ln -s /bin/tar /bin/gtar
RUN apt-get update && apt-get install build-essential zlib1g-dev -y

#Add R dependencies
RUN git clone https://github.com/ropensci/git2r.git
RUN R CMD INSTALL --configure-args='--with-zlib-lib=/usr/lib/x86_64-linux-gnu' git2r
RUN Rscript -e 'install.packages("devtools",repos="https://cran.rstudio.com")'
RUN Rscript -e 'options(unzip="internal");devtools::install_github("Albluca/distutils")'
RUN Rscript -e 'options(unzip="internal");devtools::install_github("Albluca/ElPiGraph.R", ref = "Development")'
RUN Rscript -e 'install.packages("igraph",repos="https://cran.rstudio.com")'
RUN Rscript -e 'install.packages("KernSmooth",repos="https://cran.rstudio.com")'

#add Python dependencies
RUN apt-get install libreadline-dev -y
RUN pip install rpy2==2.8.5
RUN pip install networkx==1.10
RUN pip install shapely
RUN pip install python-igraph

#Copy the script and run test dataset
#RUN mkdir /OUTPUT
#COPY STREAM /STREAM

#website dependencies
RUN pip install dash==0.21.0  # The core dash backend
RUN pip install dash-renderer==0.11.3  # The dash front-end
RUN pip install dash-html-components==0.9.0  # HTML components
RUN pip install dash-core-components==0.21.1  # Supercharged components
RUN pip install plotly --upgrade  # Plotly graphing library used in examples
RUN pip install gunicorn


RUN apt-get install unzip libxml2 libxml2-dev -y

#ATAC-script dependencies
RUN mv /opt/conda/bin/xml2-config /opt/conda/bin/xml2-config_old
RUN Rscript -e 'source("https://bioconductor.org/biocLite.R");biocLite("BiocInstaller");biocLite("BSgenome.Hsapiens.UCSC.hg19");biocLite("TFBSTools");BiocInstaller::biocLite("GreenleafLab/chromVAR")'
RUN mv /opt/conda/bin/xml2-config_old /opt/conda/bin/xml2-config

# upload button
COPY upload-button.zip /
RUN unzip upload-button.zip && cd upload-button && python setup.py install
RUN rm upload-button.zip
RUN rm -Rf upload-button

# create environment
COPY STREAM /STREAM
COPY /STREAM/STREAM.css /opt/conda/lib/python2.7/site-packages/dash_core_components/
COPY /STREAM/Loading-State.css /opt/conda/lib/python2.7/site-packages/dash_core_components/
RUN mkdir /tmp/UPLOADS_FOLDER
RUN mkdir /tmp/RESULTS_FOLDER

WORKDIR /STREAM
#RUN  python STREAM.py -m exampleDataset/data_guoji.tsv -l exampleDataset/cell_label.tsv -c exampleDataset/cell_label_color.tsv -o /OUTPUT/test
#RUN python stream.py -m exampleData/data_guoji.tsv -l exampleData/cell_label.tsv -c exampleData/cell_label_color.tsv -o test


EXPOSE 10001
CMD ["bash", "start_server_docker.sh"]

# Reroute to enable the ariadne-cli and ariadne-webapp commands
#ENTRYPOINT ["/opt/conda/bin/python", "/Ariadne/ariadne_router.py"]
