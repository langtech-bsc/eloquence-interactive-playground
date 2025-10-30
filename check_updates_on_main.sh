#!/bin/bash
set -euo pipefail

# Script working directory
cd /home/mt/eloquence-interactive-playground

# Prevent concurrent runs (use an exclusive flock on a temp lockfile)
LOCKFILE="/tmp/eloquence_update.lock"
exec 200>"$LOCKFILE"
# Try to acquire the lock; if another instance holds it, exit silently
if ! flock -n 200; then
    # optional: write a short note to stderr (not to repo log) so cron mail shows it
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Another instance is already running; exiting." >&2
    exit 0
fi


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

# If the last commit (local HEAD or the fetched origin/main) is an automated
# CI/CD commit (prefixed with [CI/CD]) then stop the script to avoid acting on
# machine-generated commits.
LAST_SUBJ_HEAD=$(git log -1 --pretty=%s HEAD 2>/dev/null || true)
LAST_SUBJ_ORIGIN=$(git log -1 origin/main --pretty=%s 2>/dev/null || true)
if (echo "$LAST_SUBJ_HEAD" | grep -qE '^\[CI/CD\]' || echo "$LAST_SUBJ_ORIGIN" | grep -qE '^\[CI/CD\]'); then
    echo "$(timestamp) Detected last commit prefixed with [CI/CD]; exiting to avoid automated loop." >&2
    exit 0
fi

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

        # Record an identifier for the human-originating change so we only
        # create a single automated commit per human change. For remote
        # pulls the new upstream HEAD is the identifier; for local-ahead
        # cases use the current HEAD.
        if [ "$remote_changed" -eq 1 ]; then
            HUMAN_REF="$NEW_HEAD"
        else
            HUMAN_REF=$(git rev-parse --verify HEAD 2>/dev/null || true)
        fi
        echo "[${TS}] Human change identifier: ${HUMAN_REF}" >> "$LOG_FILE"
    fi

    # Perform rebuild regardless. When logging is enabled (remote or local commits),
    # capture build & run output into the repo log so the last execution is recorded.
    docker stop eloquence-ip-dev || true
    docker rm eloquence-ip-dev || true
    docker rmi -f eloquence-ip:dev || true

    BUILD_SUCCESS=0
    if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
        TS=$(timestamp)
        echo "[$TS] Starting docker build for eloquence-ip:dev" >> "$LOG_FILE"
        build_start=$(date +%s)
        # Capture build output into the repo log
        if docker build -t eloquence-ip:dev . 2>&1 | tee -a "$LOG_FILE"; then
            BUILD_SUCCESS=1
        else
            BUILD_SUCCESS=0
            echo "[$(timestamp)] Docker build failed." >> "$LOG_FILE"
        fi
        build_end=$(date +%s)
        echo "Build duration: $((build_end - build_start)) seconds" >> "$LOG_FILE"

        if [ "$BUILD_SUCCESS" -eq 1 ]; then
            IMAGE_ID=$(docker images -q eloquence-ip:dev || true)
            echo "Built image id: ${IMAGE_ID}" >> "$LOG_FILE"
        fi
    else
        # No logging requested; run build normally (silent to repo log)
        if docker build -t eloquence-ip:dev .; then
            BUILD_SUCCESS=1
        else
            BUILD_SUCCESS=0
        fi
    fi

    # Start the container only if the build succeeded (or image exists)
    if [ "$BUILD_SUCCESS" -eq 1 ] || docker image inspect eloquence-ip:dev >/dev/null 2>&1; then
        if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
            TS=$(timestamp)
            echo "[$TS] Starting container eloquence-ip-dev from image eloquence-ip:dev" >> "$LOG_FILE"
            CONTAINER_ID=$(docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v `pwd`/playground-data:/data eloquence-ip:dev 2>&1)
            echo "Started container: ${CONTAINER_ID}" >> "$LOG_FILE" || true

            # Wait up to 60s for the app to start (give Gradio time to initialize)
            echo "Waiting up to 60s for application to initialize..." >> "$LOG_FILE"
            wait_time=0
            interval=5
            max_wait=60
            while [ $wait_time -lt $max_wait ]; do
                sleep $interval
                wait_time=$((wait_time + interval))
                # Optionally, check logs for a known ready marker (Application startup complete)
                if docker logs eloquence-ip-dev --timestamps --tail 50 2>/dev/null | grep -q "Application startup complete"; then
                    echo "Application reported startup complete after ${wait_time}s" >> "$LOG_FILE"
                    break
                fi
            done

            # Capture the recent container logs into the repo log (last 200 lines)
            echo "=== Container logs (last 200 lines) ===" >> "$LOG_FILE"
            docker logs eloquence-ip-dev --timestamps --tail 200 >> "$LOG_FILE" 2>&1 || true
        else
            # Non-logged path: start container without appending to repo log
            docker run --name eloquence-ip-dev -d --net="host" -e GRADIO_SERVER_PORT=8081 -v `pwd`/playground-data:/data eloquence-ip:dev
        fi
    else
        if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
            echo "[$(timestamp)] Build failed and image not available; container not started." >> "$LOG_FILE"
        fi
    fi

    # If we logged earlier (i.e. a remote or local commit triggered the rebuild), commit & push the log
    if [ "$remote_changed" -eq 1 ] || [ "$local_ahead" -eq 1 ]; then
        git add -- "$LOG_FILE" || true
        if ! git diff --quiet --cached -- "$LOG_FILE"; then
            TS=$(timestamp)
            # Prevent creating multiple automated commits for the same
            # human-originating change: include HUMAN_REF in the automated
            # commit message and skip committing if the latest commit
            # already references that HUMAN_REF.
            # Refresh remote ref so concurrent runners see latest remote commits
            git fetch origin --quiet || true
            LAST_MSG_LOCAL=$(git log -1 --pretty=%B 2>/dev/null || true)
            LAST_MSG_REMOTE=$(git log -1 origin/main --pretty=%B 2>/dev/null || true)
            if [ -n "${HUMAN_REF:-}" ] && (echo "$LAST_MSG_LOCAL" | grep -q "$HUMAN_REF" || echo "$LAST_MSG_REMOTE" | grep -q "$HUMAN_REF"); then
                echo "[$(timestamp)] Automated commit already exists for ${HUMAN_REF}; skipping commit." >> "$LOG_FILE"
            else
                # Commit only the log file to avoid including other staged files.
                git commit -m "[CI/CD] $TS - $HUMAN_REF" -- "$LOG_FILE" || true
                git push origin main || true
            fi
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

