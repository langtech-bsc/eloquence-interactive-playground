#!/bin/bash
set -euo pipefail

# Build using the same image name as the automated updater script
docker build -t eloquence-ip:dev .

# Run the container with the same name and options used by check_updates_on_main.sh
docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v `pwd`/playground-data:/data eloquence-ip:dev
