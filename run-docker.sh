docker build -t eloquence-ip:prod .
docker run -d --name eloquence-ip-prod --net="host" -v `pwd`/playground-data:/data eloquence-ip:prod
