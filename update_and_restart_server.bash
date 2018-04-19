git pull 
docker stop $(docker ps -q --filter ancestor=pinellolab/stream)
docker run -p 10001:10001 -v /Volumes/pinello/PROJECTS/2016_12_STREAM/STREAM_DB_top_15/:/STREAM/precomputed -v /Users/sailor/Projects/STREAM/STREAM:/STREAM -d -it pinellolab/stream STREAM_webapp
