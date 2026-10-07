# Pilot3 RAG on the BSC Interactive Playground

This fork runs the **Pilot_3 RAG pipeline** (LaBSE retriever + Salamandra-7B) behind the
**BSC Interactive Playground** UI. Two containers come up on your own machine; the only
remote piece is the LLM, which runs on a BSC GPU VM and is reached over an SSH tunnel.

```
Browser ──▶ localhost:8086  (IP UI container)
                 │  pilot3-http://pilot3-server:8000   (Docker bridge network)
                 ▼
            pilot3-server  (Pilot_3 FastAPI + LaBSE, CPU)
                 │  PILOT3_SALAMANDRA_API_BASE = http://host.docker.internal:9001/v1
                 │  PILOT3_KRIKRI_API_BASE     = http://host.docker.internal:9002/v1
                 ▼
            SSH tunnel on your host  localhost:9001 ──▶ BSC VM vLLM (Salamandra)
                                     localhost:9002 ──▶ BSC VM vLLM (Krikri)
```

Everything except the LLM is CPU-only, so the whole stack runs locally.


The integration itself is: `pilot3/` (the vendored pipeline — `server.py`, `pipeline.py`,
`retriever.py`, `generator.py`, its own Dockerfile), a `Pilot3TaskHandler` in
`gradio_app/backend/task_handlers.py`, a relaxed LLM-selection gate in
`gradio_app/app_handlers.py`, and the task config at
`playground-data/configurations/task_configs/pilot3_rag.json`.

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
- **The current vLLM ports.** Each model is served by its own vLLM instance on its own port
  (at the time of writing Salamandra on `58093`, Krikri on `58105`). BSC stops the models
  between sessions and the ports can change on restart, so ask them for the live ones.
- **~11 GB free disk and roughly 35 minutes for the first run.** Measured end-to-end on a
  Mac with an empty cache: 27 minutes to build both images (about 4 GB of wheels, including
  two separate torch installs), then 9 minutes on first startup while the pipeline downloads
  LaBSE, LaBSE-TID, MiniLM-L6-TID and the two Chroma indexes (about 3.7 GB) from HuggingFace. All are public — no
  HF token needed. Disk goes to the UI image (4.3 GB), the pipeline image (2.4 GB), base
  images (0.4 GB) and the model cache (about 3.7 GB).


## 1. Clone and configure

```bash
git clone <repo-url> && cd eloquence-interactive-playground-BSC
cp pilot3/.env.example pilot3/.env
```

## 2. Open the tunnel to the BSC LLM

Keep this terminal open for as long as you use the app — generation needs it:

```bash
ssh -i ~/.ssh/bsc_vllm -o IdentitiesOnly=yes -N \
  -L 9001:127.0.0.1:<SALAMANDRA_PORT> \
  -L 9002:127.0.0.1:<KRIKRI_PORT> \
  <your-bsc-user>@212.128.227.234
```

`9001` and `9002` are the local ports; they must match `PILOT3_SALAMANDRA_API_BASE` and
`PILOT3_KRIKRI_API_BASE` in `pilot3/.env`. `<SALAMANDRA_PORT>` / `<KRIKRI_PORT>` are where
each vLLM listens inside the VM. If you only need one model, forward only its port. Either
variable can be left unset, and that model then uses `LLM_API_BASE`.

`-o IdentitiesOnly=yes` is not optional. Without it SSH offers every key in your agent first
and BSC cuts the connection with `Too many authentication failures` before reaching
`bsc_vllm`.

Confirm the endpoints before touching the UI. Each must return JSON listing the model id
set in `pilot3/.env` (`PILOT3_SALAMANDRA_MODEL` / `PILOT3_KRIKRI_MODEL`):

```bash
curl -s localhost:9001/v1/models   # salamandra-7b-instruct
curl -s localhost:9002/v1/models   # Llama-Krikri-8B-Instruct
```

An empty reply means the tunnel is up but that model isn't running on BSC.

## 3. Build and start the stack

```bash
docker compose up -d --build     # or: make deploy
docker compose logs -f           # tail both services
```

Wait for:

- `pilot3-server` → `Uvicorn running on http://0.0.0.0:8000`
- `eloquence-ip-bsc` → `Running on local URL: http://0.0.0.0:8086`


## 4. Use it

1. Open `http://localhost:8086`.
2. Log in with an account from `prep_scripts/prepare_users.py`. The UI seeds its user
   database on first container start, so a fresh clone works without extra steps.
3. Playground → **Task configuration: "Pilot3 RAG (Call Center)"**.
4. Type a question and submit. The answer appears in the chat, retrieved documents in the
   right-hand panel.

### Retriever and LLM options

Under **Task & Model Selection** the Pilot3 task exposes two independent choices. Both
combinations are live and can be switched between turns — nothing is reloaded, so the change
takes effect on the next message.

**Retriever** (the `Retriever` radio):

- `Baseline LaBSE` → `retriever_type=baseline`, stock `sentence-transformers/LaBSE`. Default.
- `Fine-tuned LaBSE (TID)` → `retriever_type=finetuned`, the `Cutting3dg3/LaBSE-TID` model
  fine-tuned on the pilot's data.
- `Fine-tuned MiniLM-L6, English (TID)` → `retriever_type=minilm_l6_tid`, the
  `Cutting3dg3/MiniLM_L6_TID_d` model (English-only `all-MiniLM-L6-v2` fine-tuned with TID on
  doc2dial; training code in `Pilot_3/mMiniLM`). Not meant for Greek input. As in its
  training, it embeds only the last 5 turns, joined by `[SEP]`, up to 512 tokens.

The two LaBSE retrievers query the same Chroma collection. MiniLM-L6 queries its own index,
`Cutting3dg3/minilm-tid-chromadb` (the same propositions, encoded by the base MiniLM-L6).
All are instantiated at server start, so switching costs nothing at query time. The server
log line tells you which one ran: `[RETRIEVER:baseline]`, `[RETRIEVER:finetuned]` or
`[RETRIEVER:minilm_l6_tid]`.
Similarity scores are not comparable across embedding models.

**Response generator** (the `Available LLMs` radio). Only the two **Pilot3** entries are
served by the pipeline; the list is filtered to them on this task:

- `Salamandra-7B (Pilot3)` → `llm_type=salamandra`. Default.
- `Llama-Krikri-8B (Pilot3)` → `llm_type=krikri`.

Switching changes both the model name and the endpoint the pipeline calls: Salamandra goes
to `PILOT3_SALAMANDRA_API_BASE`, Krikri to `PILOT3_KRIKRI_API_BASE`. The server log shows
which endpoint was used, e.g. `[GENERATOR] Sending 1 turns to 'Llama-Krikri-8B-Instruct' at
http://host.docker.internal:9002/v1`. If Krikri fails with `Connection error`, its tunnel
(`9002`) is missing or Krikri isn't running on BSC. A `404` means `PILOT3_KRIKRI_MODEL`
doesn't match the id from `curl localhost:9002/v1/models`.

After editing `pilot3/.env`, run `docker compose up -d server`. A plain `restart` does not
reload `env_file`.

The generation parameters under **LLM Parameters** (temperature, top-p, max tokens) are
forwarded to the pipeline on both paths.
