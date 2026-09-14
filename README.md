# Pilot3 RAG on the BSC Interactive Playground

This fork runs the **Pilot_3 RAG pipeline** (LaBSE retriever + Salamandra-7B) behind the
**BSC Interactive Playground** UI. Two containers come up on your own machine; the only
remote piece is the LLM, which runs on a BSC GPU VM and is reached over an SSH tunnel.

```
Browser ──▶ localhost:8086  (IP UI container)
                 │  pilot3-http://pilot3-server:8000   (Docker bridge network)
                 ▼
            pilot3-server  (Pilot_3 FastAPI + LaBSE, CPU)
                 │  LLM_API_BASE = http://host.docker.internal:9001/v1
                 ▼
            SSH tunnel on your host  localhost:9001 ──▶ BSC VM vLLM
```

Everything except the LLM is CPU-only, so the whole stack runs locally. See
[INTEGRATION_LAUNCH.md](INTEGRATION_LAUNCH.md) for the architecture rationale and the
integration's code-level changes.

## Prerequisites

- **Docker.** Docker Desktop works as-is. With Colima, use the Apple Virtualization
  backend — the default QEMU backend needs `qemu-img`, whereas `vz` is built into macOS 13+:
  ```bash
  colima start --vm-type vz --cpu 4 --memory 6
  ```
  > A profile's VM type is fixed at creation. To convert an existing QEMU profile:
  > `colima delete default && colima start --vm-type vz --cpu 4 --memory 6`.
- **Your own SSH account on the BSC VM.** vLLM binds to `127.0.0.1` inside the VM, so the
  only way to reach it is a tunnel that terminates in your own account. Request one from
  BSC, put the private key at `~/.ssh/bsc_vllm`, and `chmod 600 ~/.ssh/bsc_vllm` — SSH
  refuses keys that are readable by others.
- **The current vLLM port.** BSC stops the model between sessions and the port changes on
  restart. Ask them for the live port rather than reusing one from these docs.
- ~5 GB free disk. The first run takes 10-20 minutes: two image builds, plus LaBSE and the
  Chroma index downloading from HuggingFace. Both are public — no HF token needed.

## 1. Clone and configure

```bash
git clone <repo-url> && cd eloquence-interactive-playground-BSC
cp pilot3/.env.example pilot3/.env
```

`pilot3/.env` works unmodified in the standard setup. Leave
`LLM_API_BASE=http://host.docker.internal:9001/v1` as-is: the server runs in a
bridge-networked container, so `host.docker.internal` is how it reaches the tunnel on your
host. `localhost` there would resolve to the container itself.

`pilot3/.env` is gitignored. Never commit it.

## 2. Open the tunnel to the BSC LLM

Keep this terminal open for as long as you use the app — generation needs it:

```bash
ssh -i ~/.ssh/bsc_vllm -o IdentitiesOnly=yes -N \
  -L 9001:127.0.0.1:<VLLM_PORT> <your-bsc-user>@212.128.227.234
```

`9001` is the local port (it must match `LLM_API_BASE`); `<VLLM_PORT>` is where vLLM listens
inside the VM.

`-o IdentitiesOnly=yes` is not optional. Without it SSH offers every key in your agent first
and BSC cuts the connection with `Too many authentication failures` before reaching
`bsc_vllm`.

Confirm the endpoint before touching the UI — it must return JSON listing
`salamandra-7b-instruct`:

```bash
curl -s localhost:9001/v1/models
```

Two failures to tell apart, neither of which is a problem on your machine:

- **`channel N: open failed: connect failed: Connection refused`** — you authenticated fine,
  but nothing is listening on `<VLLM_PORT>` inside the VM. The model is stopped (BSC's
  normal state between test windows) or the port moved.
- **`curl: (52) Empty reply from server`** — something holds that port but it is not vLLM.
  Wrong port.

Either way, ask BSC to confirm the model is running and on which port. Authentication
problems look different: they fail at login and you never get a tunnel at all.

Pre-flight check that skips the tunnel entirely:

```bash
ssh -i ~/.ssh/bsc_vllm -o IdentitiesOnly=yes -o BatchMode=yes \
  <your-bsc-user>@212.128.227.234 \
  'curl -s -m 5 http://127.0.0.1:<VLLM_PORT>/v1/models || echo "vLLM down"'
```

## 3. Build and start the stack

```bash
docker compose up -d --build     # or: make deploy
docker compose logs -f           # tail both services
```

> The two are not quite identical. `make deploy` passes `--env-file ./pilot3/.env`, so a
> `HF_CACHE_DIR` set there also drives the cache volume path; plain `docker compose up` only
> reads `pilot3/.env` inside the container and falls back to `~/.cache/pilot3-hf` for the
> mount. If you set `HF_CACHE_DIR`, use an absolute path that your Docker VM actually mounts.

Wait for:

- `pilot3-server` → `Uvicorn running on http://0.0.0.0:8000`
- `eloquence-ip-bsc` → `Running on local URL: http://0.0.0.0:8086`

The `0.0.0.0` in those banners is the bind address, not a URL to open.

Health checks:

```bash
curl -s localhost:8000/health                                            # {"status":"ready","current_llm":"salamandra"}
curl -s -o /dev/null -w "8086 -> %{http_code}\n" http://localhost:8086/  # expect 200
```

`/health` is ready immediately — no weights load locally. If the tunnel is down, only
generation fails; retrieval still works.

> Editing anything under `playground-data/configurations/` needs no rebuild — it is a
> mounted volume. Just `docker compose restart ui`.

## 4. Use it

1. Open `http://localhost:8086`.
2. Log in with an account from `prep_scripts/prepare_users.py`. The UI seeds its user
   database on first container start, so a fresh clone works without extra steps.
3. Playground → **Task configuration: "Pilot3 RAG (Call Center)"**.
4. Type a question and submit. The answer appears in the chat, retrieved documents in the
   right-hand panel.

No LLM selection is required — the Pilot3 path defaults to Salamandra-7B. Only the two
**Pilot3** entries in the dropdown map to anything real:

- `Salamandra-7B (Pilot3)` → pipeline `llm_type=salamandra` (default).
- `Llama-Krikri-8B (Pilot3, WIP)` → pipeline `llm_type=krikri`, still work in progress.

Any other entry falls back to Salamandra on this task.

## Logs

`[RETRIEVER:baseline]` and `[GENERATOR]` lines go to the server container's stdout and flush
live:

```bash
docker compose logs -f server     # or: docker logs -f pilot3-server
docker compose logs -f ui         # Gradio + handler side
```

## Shutting down

```bash
docker compose down               # both containers
```

Then Ctrl+C the SSH tunnel terminal.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Tunnel: `channel N: open failed: connect failed: Connection refused` | Auth succeeded; nothing is listening on the remote port. Model stopped or port changed — ask BSC. |
| Tunnel: `Too many authentication failures` | SSH offered too many agent keys. Add `-o IdentitiesOnly=yes`. |
| Tunnel: `Address already in use` on 9001 | Stale tunnel. `lsof -ti tcp:9001 \| xargs kill`, or pick another local port and update `LLM_API_BASE` to match. |
| `localhost:8086` (or `:8000`) returns `000` / refused, but `docker compose ps` shows `Up` | An IDE auto port-forward (Cursor / VS Code) bound the port on loopback. Stop forwarding in the **Ports** panel and set `"remote.autoForwardPorts": false`, or use the LAN IP: `http://$(ipconfig getifaddr en0):8086`. |
| Chat: `could not reach the Pilot_3 pipeline at http://pilot3-server:8000 ... Connection refused` | The `server` container is not up yet. `docker compose logs -f server`, wait for the Uvicorn banner, submit again. |
| Generation: `APIConnectionError` / `Connection refused` (Errno 111) | No tunnel. Confirm with `lsof -nP -iTCP:9001 -sTCP:LISTEN`, then redo step 2. |
| Generation: `RemoteProtocolError: Server disconnected` | Tunnel is up but the BSC side is not serving. See step 2. |
| Generation: `404` / model not found | `PILOT3_SALAMANDRA_MODEL` does not match the id from `curl localhost:9001/v1/models`. Fix `pilot3/.env`, then `docker compose restart server`. |
| `error while attempting to bind ... 8086/8000: address already in use` | `docker compose down`, or free it: `lsof -ti tcp:8086 \| xargs kill`. |
| Several GB re-download on every cold start | `HF_CACHE_DIR` points somewhere the VM does not mount. Under Colima keep it under `$HOME` (default `~/.cache/pilot3-hf`); `/tmp` is not mounted. |
| Base image will not pull on Apple Silicon | Add `platform: linux/amd64` to the affected service in `docker-compose.yml`. |
| `host.docker.internal` does not resolve | Old Docker Desktop; the compose file already sets `extra_hosts: host.docker.internal:host-gateway`. Update Docker Desktop. |
| `colima start` fails: `qemu-img not found` | The profile uses the QEMU backend. Recreate it with `vz` — see Prerequisites. |
| "Pilot3 RAG (Call Center)" missing from the task list | `playground-data` not mounted, or a stale image. Check the `ui` volume mount and rebuild with `docker compose up --build`. |

## Running without Docker

See [RUN_WITHOUT_DOCKER.md](RUN_WITHOUT_DOCKER.md).

---

## Upstream project README

The original BSC Interactive Playground README, kept for reference. Its deployment
instructions predate the Pilot3 integration — `make deploy` in this fork reads
`./pilot3/.env`, and `OPENAI_API_ENDPOINT_URL` is not used on the Pilot3 path.

> Based on https://huggingface.co/spaces/akazakov/rag-gradio-sample-project/tree/main/gradio_app
>
> ### Environment Variables
>
> To run this project, you will need to add the following environment variables to your
> `.env` file: `OPENAI_API_ENDPOINT_URL`
>
> Example `.env` file:
>
> ```bash
> OPENAI_API_ENDPOINT_URL=http://172.17.0.1:8080/v1/completions
> ```
>
> ### Deployment (docker compose)
>
> ```bash
> make deploy      # deploy
> make undeploy    # delete deployment
> make stop        # stop deployment
> ```
