#!/usr/bin/env bash
set -euo pipefail

docker build -t eloquence-ip:prod .
docker rm -f eloquence-ip-prod 2>/dev/null || true
docker run -d --name eloquence-ip-prod --net="host" -v `pwd`/playground-data:/data eloquence-ip:prod
