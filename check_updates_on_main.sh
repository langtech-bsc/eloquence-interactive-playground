#!/bin/bash
set -euo pipefail

# Script working directory
cd /home/mt/eloquence-interactive-playground

# Log file (kept in the repo so everyone can view it)
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/updates.log"
mkdir -p "$LOG_DIR"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Record current HEAD so we can detect what changed after pull
OLD_HEAD=$(git rev-parse --verify HEAD)

# Always pull from origin/main (ensure we fetch remote changes every run)
PULL_OUTPUT=$(git pull origin main 2>&1 || true)
NEW_HEAD=$(git rev-parse --verify HEAD)

remote_changed=0
local_ahead=0
CHANGED_FILES=""

if [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    # Pull brought new commits from remote
    remote_changed=1
    CHANGED_FILES=$(git diff --name-only "$OLD_HEAD" "$NEW_HEAD" || true)
else
    # No remote changes from pull. Check if this machine has local commits
    # that are ahead of origin/main — these should also trigger a rebuild.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        counts=$(git rev-list --left-right --count origin/main...HEAD 2>/dev/null || echo "0 0")
        behind=$(echo "$counts" | awk '{print $1}')
        ahead=$(echo "$counts" | awk '{print $2}')
        if [ "$ahead" -gt 0 ]; then
            local_ahead=1
            # Files changed between origin/main and local HEAD
            CHANGED_FILES=$(git diff --name-only origin/main HEAD || true)
        fi
    else
        # No origin/main ref (e.g. initial commit or missing remote); treat local HEAD as changed
        local_ahead=1
        CHANGED_FILES=$(git ls-files || true)
    fi
fi

# Decide whether to rebuild: only rebuild when there are changes outside the log dir
rebuild_needed=0
if [ -n "$CHANGED_FILES" ]; then
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        # If the change is inside the logs directory, skip it
        case "$f" in
            "$LOG_DIR"/*|logs/*) ;;
            *) rebuild_needed=1; break ;;
        esac
    done <<< "$CHANGED_FILES"
fi

if [ "$rebuild_needed" -eq 1 ]; then
    # Only write to the log and commit/push when there were upstream commits or local commits
    if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
        TS=$(timestamp)
        # Truncate the log so it only contains the latest execution results
        : > "$LOG_FILE"
        reason=""
        if [ "$remote_changed" -eq 1 ]; then
            reason="remote"
        else
            reason="local"
        fi
        echo "[$TS] Changes detected on main ($reason); preparing rebuild." >> "$LOG_FILE"
        echo "git pull output:" >> "$LOG_FILE"
        echo "$PULL_OUTPUT" >> "$LOG_FILE"
        echo "Changed files:" >> "$LOG_FILE"
        echo "$CHANGED_FILES" >> "$LOG_FILE"
    fi

    # Perform rebuild regardless, but only log when there were upstream commits
    docker stop eloquence-ip-dev || true
    docker rm eloquence-ip-dev || true
    docker rmi -f eloquence-ip:dev || true
    docker build -t eloquence-ip:dev .
    docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v `pwd`/playground-data:/data eloquence-ip:dev

    # If we logged earlier (i.e. a remote or local commit triggered the rebuild), commit & push the log
    if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
        git add "$LOG_FILE" || true
        if ! git diff --quiet --cached -- "$LOG_FILE"; then
            TS=$(timestamp)
            git commit -m "Automated update log: $TS" || true
            git push origin main || true
            echo "[$TS] Pushed log commit." >> "$LOG_FILE"
        fi
    fi
else
    # No rebuild needed; ensure container is running but do not write to log
    CONTAINER_ID=$(docker ps -a --filter "name=eloquence-ip-dev" -q || true)
    RUNNING_ID=$(docker ps --filter "name=eloquence-ip-dev" -q || true)
    if [ -n "$RUNNING_ID" ]; then
        : # running, nothing to do and do not log
    elif [ -n "$CONTAINER_ID" ]; then
        docker start eloquence-ip-dev
    else
        # If the image exists locally, create & start a container. If not, skip silently
        # (do not write to logs or attempt to pull from a registry).
        if docker image inspect eloquence-ip:dev >/dev/null 2>&1; then
            docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v `pwd`/playground-data:/data eloquence-ip:dev
        else
            echo "Image eloquence-ip:dev not found locally; skipping container creation." >&2
        fi
    fi
fi

