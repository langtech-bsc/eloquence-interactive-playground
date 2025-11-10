#!/bin/bash
set -euo pipefail

# If a container with the same name exists, remove it to avoid "name already in use" errors.
existing=$(docker ps -a --filter "name=eloquence-ip-dev" --format '{{.ID}}')
if [ -n "${existing}" ]; then
	echo "Found existing container(s) with name 'eloquence-ip-dev':"
	docker ps -a --filter "name=eloquence-ip-dev"
	echo "Stopping and removing existing container(s)..."
	docker rm -f eloquence-ip-dev
fi

# Build using the same image name as the automated updater script
docker build -t eloquence-ip:dev .

# Run the container with the same name and options used by check_updates_on_main.sh
docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v "$(pwd)"/playground-data:/data eloquence-ip:dev
