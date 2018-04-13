import os
import subprocess as sb
import sys

if len(sys.argv)==1:

	print	'\n- STREAM Docker Container -\n\t- use `STREAM` to use the command line\n\t- use `STREAM_webapp` to start the web application\n'
	sys.exit(1)

if sys.argv[1]=='STREAM':
	sb.call(["/opt/conda/bin/python", "/STREAM/STREAM.py"]+ sys.argv[2:])
elif sys.argv[1]=='STREAM_webapp':
	sb.call(["/bin/bash", "-c", "/STREAM/start_server_docker.sh"])
else:
	print	'\n- STREAM Docker Container -\n\t- use `STREAM` to use the command line\n\t- use `STREAM_webapp` to start the web application\n'
