docker run  -v $PWD/STREAM/exampleDataset:/data -w /data  -v $PWD:/output pinellolab/stream STREAM -m data_guoji.tsv -l cell_label.tsv -c cell_label_color.tsv -s all -o /output/STREAM_Results
