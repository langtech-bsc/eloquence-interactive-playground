Based on https://huggingface.co/spaces/akazakov/rag-gradio-sample-project/tree/main/gradio_app

## NurseLLM: English and Serbian dialogue managers

Select **DM**, choose **NurseLLM**, and use **Language** below the conversation
and above **Summary**. Serbian is the default. Changing language starts a fresh
conversation and clears the summary. **Clear** resets the selected language's
session; browser sessions stay separate.

Configure both URLs in the single `NurseLLM` entry in
`playground-data/configurations/models.json`:

```json
"default_language": "sr",
"language_endpoints": {
    "sr": "http://127.0.0.1:8004",
    "en": "http://127.0.0.1:8005"
}
```

Both use `DialogueManagerInteractor`: Serbian calls `/v1/chat/uns_dm`, English
calls `/v1/chat/uns_dm_eng`. Start the DM services and their NER/LLM backends
separately. Summaries use the existing Nemo service.

Tests:

```bash
python -m unittest discover -s tests -p test_dialogue_manager_languages.py -v
python -m unittest discover -s tests -p test_attention_summarization.py -v
```

## Summarization with Nemo attention

The **Summarization** and **DM** tasks both generate summaries with the separate
Nemo attention service when clicking **Summarize conversation**. Conversation
continues through the selected LLM or dialogue manager. Select text in **Summary**
to highlight the corresponding tokens inside the conversation bubbles. Press ESC or click the conversation to clear the
highlight. Attention is an interpretation, not a causal explanation.

Start the attention service separately; IP does not start or download a model.
The service is registered in `playground-data/configurations/models.json` with
`model_name: "Mistral-Nemo-Instruct-2407"`, `display_name: "Nemo"`,
`interactor: "attention_summarization"`,
`api_endpoint: "http://127.0.0.1:5010/summarize_with_attention"` and
`interface: "service"`. This keeps it out of the conversation model selector.
The display name is the internal registry key; `interface: "service"` keeps
this entry hidden from the conversation model selector. If a display name is
omitted, the registry falls back to `model_name`.
Change that entry to configure the endpoint. `ATTENTION_REQUEST_TIMEOUT`
(default 300 seconds) sets the request timeout in the IP environment. The
current Compose deployment uses host networking; with bridge networking, use an
address reachable from the IP container. The interactor maps
`Mistral-Nemo-Instruct-2407` (also accepting the full Hub ID) to the service
alias `model=nemo`, reusing its bundled weights and
preloaded model instead of loading another instance. Summarization uses
`top_k=16`, retaining the strongest 16 entries per attention row, matching the
standalone `/demo` default. This filters the visualization response without
changing summary generation. In DM, summarizing does not end or reset the
dialogue-manager session; **Clear conversation** still resets it.

The button posts `operation: "summarize"` to the IP's existing form-encoded
`/query`, which resolves the interactor from `models.json` and calls Nemo.
Both `task_config: "Summarization"` and `task_config: "DM"` support this operation.
The local IP port comes from `GRADIO_SERVER_PORT` (default 8080).
API clients can use the same operation:

```bash
curl -s http://127.0.0.1:8080/dev/query \
  -F 'body={"operation":"summarize","task_config":"Summarization","llm_name":"Nemo","history":[["Tengo fiebre desde ayer.","¿Tienes otros síntomas?"]],"input_text":"","docs_k":0,"temp":0,"top_p":1,"max_tokens":150,"index_name":null}'
```

Adjust the IP host, port and path prefix to your deployment. `llm_name` is retained
for API compatibility and does not select the summarizer. The response contains
the summary in `text` and the sidecar payload in `summary_attention`, including
token texts, character offsets, lengths, sparse COO attention and `message_spans`
mapping prompt characters to conversation messages. Requests without `operation`
continue to use the ordinary chat path.

The browser computes highlighting locally with the demo's normalization and
gamma (0.35). It never sends selection events to the model. Original conversation
history is stored separately from rendered HTML. Summaries use the visible text of
Markdown/HTML messages, while attention is inserted into their formatted text nodes
so emphasis, code and safe links remain usable. Wait for the assistant to finish
streaming before summarizing; results for a changed history are discarded.
Attention is invalidated when the conversation, task, model or system prompt changes. Token metadata is
validated before rendering: incompatible or misaligned sidecar responses produce
an error instead of highlighting unrelated text.

The attention service is used as-is. Its legacy token-by-token offsets can be
invalid for accents or emojis. The IP recomputes character spans using the local
Nemo tokenizer snapshot configured by `tokenizer_path` in the service entry.
Only the tokenizer is included; no model weights are loaded by the IP. Every
token piece and token count must match the service response before replacing
offsets. Incompatible metadata is rejected rather than truncated or clamped.
Rebuild/restart only the IP after its source changes; the sidecar needs no edits.
  
## About
...

## Deploy with docker compose

### Prerequisites

Make

[Docker](https://docs.docker.com/engine/install/ubuntu/)

[Docker compose](https://docs.docker.com/compose/install/)

### Environment Variables
To run this project, you will need to add the following environment variables to your .env file.


`OPENAI_API_ENDPOINT_URL`

Example .env file

```bash
OPENAI_API_ENDPOINT_URL=http://172.17.0.1:8080/v1/completions
```

## Deployment (docker compose)

To deploy run

```bash
make deploy
```

To delete deployment run
```bash
make undeploy
```

To stop deployment run
```bash
make stop
```
