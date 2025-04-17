docker build -t eloquence-interactive-playground .
docker run -d  --net="host" -v `pwd`/playground-data:/data eloquence-interactive-playground
