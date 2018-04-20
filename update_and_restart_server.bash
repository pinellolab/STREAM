git pull 
docker stop $(docker ps -q --filter ancestor=pinellolab/stream)
docker run -p 10001:10001 \
-v /Users/sailor/Projects/STREAM/STREAM:/STREAM \
-v /Volumes/pinello/PROJECTS/2016_12_STREAM/STREAM_DB_top_15/:/STREAM/precomputed \
-d -it pinellolab/stream STREAM_webapp


#-v /Volumes/Data/STREAM/UPLOADS_FOLDER:/tmp/UPLOADS_FOLDER \
#-v /Volumes/Data/STREAM/RESULTS_FOLDER:/tmp/RESULTS_FOLDER \

#-v /Volumes/Data/STREAM/tmp:/tmp \
#-v /Volumes/Data/STREAM_DB_top_15/:/STREAM/precomputed \

