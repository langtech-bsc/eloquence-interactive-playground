import os

from openai import OpenAI

# llm_type -> the exact model string the (vLLM) server expects. Overridable via env so the
# repo ships no hard-coded endpoint-specific names; BSC's value goes in the .env.
AVAILABLE_MODELS = {
    "salamandra": os.environ.get("PILOT3_SALAMANDRA_MODEL", "salamandra-7b-instruct"),
    "krikri": os.environ.get("PILOT3_KRIKRI_MODEL", "Llama-Krikri-8B-Instruct"),
}

DEFAULT_MODEL = AVAILABLE_MODELS["salamandra"]

# OpenAI-compatible endpoint (e.g. the near end of an SSH tunnel to the LLM server).
LLM_API_BASE = os.environ.get("LLM_API_BASE", "http://localhost:9001/v1")
# vLLM on a private VM is usually keyless; the OpenAI client still needs a non-empty value.
LLM_API_KEY = os.environ.get("LLM_API_KEY") or "EMPTY"

# The served model's context window. salamandra-7b-instruct exposes only 2048 tokens, so a
# multi-turn chat with 20 retrieved documents + a 512-token answer request overflows it and
# the server returns HTTP 400. Configurable so a larger model lifts the cap without a code
# change (e.g. PILOT3_MODEL_MAX_CONTEXT=8192).
MODEL_MAX_CONTEXT = int(os.environ.get("PILOT3_MODEL_MAX_CONTEXT", "2048"))
_CONTEXT_SAFETY_MARGIN = 64   # headroom for chat-template tokens our estimate can't see
_MIN_OUTPUT_TOKENS = 64       # never clamp the answer below this


def _estimate_tokens(text: str) -> int:
    # Conservative upper-bound estimate (no remote tokenizer available): multilingual text
    # averages ~3 chars/token; biasing high means we trim slightly early rather than 400.
    return len(text) // 3 + 1

_SYSTEM_PROMPT = (
    "You are a customer service assistant for a call center. You will receive background "
    "information retrieved from the company knowledge base, followed by the conversation "
    "with the user.\n\n"
    "CRITICAL: The background information is your ONLY source of facts. You have no "
    "general knowledge. Even if you know an answer from training, you MUST NOT use it. "
    "A passing mention of a term in the background does not license you to elaborate on "
    "it from outside knowledge — the answer must be explicitly stated.\n\n"
    "Rules:\n"
    "1. If the user's question is unrelated to customer service (e.g. small talk, jokes, "
    "math, code, general trivia), politely tell them you can only help with customer "
    "service inquiries.\n"
    "2. If the background information explicitly answers the question, respond directly "
    "and naturally. Do NOT reference the source — never say things like \"according to "
    "the document\", \"based on the context\", or \"the information provided says\". "
    "Answer as if you already knew it.\n"
    "3. If the background information does not explicitly answer the question — even if "
    "you know the answer from general knowledge — you MUST say you do not have that "
    "information. Never guess, infer beyond what is stated, or fill gaps with outside "
    "knowledge.\n"
    "4. Reply in the same language the user uses.\n"
    "5. Be concise and direct."
)


class Generator:
    """Thin client over an OpenAI-compatible chat endpoint (the LLM runs remotely).

    No weights are loaded locally — `generate()` POSTs to LLM_API_BASE/chat/completions and
    the server applies the model's chat template to the `messages` we send.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 512, **_ignored):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        print(f"[GENERATOR] Using remote model '{self.model_name}' at {LLM_API_BASE}")

    def unload(self):
        # Nothing is loaded locally anymore; kept for API compatibility.
        pass

    def _build_messages(self, dialog_history: list[str], retrieved_docs: dict,
                        max_docs: int | None = None) -> list[dict]:
        docs = retrieved_docs.get("documents", [[]])[0]
        if max_docs is not None:
            docs = docs[:max_docs]
        context = "\n\n".join(f"[Document {i + 1}]: {doc}" for i, doc in enumerate(docs))

        # inject context into system prompt so it's available across all turns
        system = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}"
        messages = [{"role": "system", "content": system}]

        # rebuild conversation turns (dialog_history alternates user/assistant)
        roles = ["user", "assistant"]
        for i, turn in enumerate(dialog_history):
            messages.append({"role": roles[i % 2], "content": turn})

        return messages

    def _fit_to_context(self, dialog_history: list[str], retrieved_docs: dict,
                        requested_output: int):
        """Shrink the prompt so input + output fits MODEL_MAX_CONTEXT.

        Drops the least-relevant retrieved documents first (they are ordered most- to
        least-relevant), then the oldest conversation turns in user/assistant pairs so the
        role alternation and the latest user message are preserved. Returns
        (messages, max_tokens, n_docs_used).
        """
        total_docs = len(retrieved_docs.get("documents", [[]])[0])
        n_docs = total_docs
        history = list(dialog_history)
        while True:
            messages = self._build_messages(history, retrieved_docs, max_docs=n_docs)
            input_est = sum(_estimate_tokens(m["content"]) for m in messages) + 4 * len(messages)
            budget = MODEL_MAX_CONTEXT - input_est - _CONTEXT_SAFETY_MARGIN
            if budget >= _MIN_OUTPUT_TOKENS:
                break
            if n_docs > 0:
                n_docs -= 1                       # drop least-relevant document
            elif len(history) > 2:
                history = history[2:]             # drop oldest user/assistant pair
            else:
                budget = _MIN_OUTPUT_TOKENS       # can't shrink further; let the server arbitrate
                break
        max_tokens = max(_MIN_OUTPUT_TOKENS, min(requested_output, budget))
        return messages, max_tokens, n_docs

    def generate(
        self,
        dialog_history: list[str],
        retrieved_docs: dict,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
    ) -> str:
        requested = max_new_tokens or self.max_new_tokens
        messages, max_tokens, n_docs = self._fit_to_context(dialog_history, retrieved_docs, requested)

        total_docs = len(retrieved_docs.get("documents", [[]])[0])
        if n_docs < total_docs:
            print(f"[GENERATOR] Trimmed injected context to {n_docs}/{total_docs} docs to fit "
                  f"the {MODEL_MAX_CONTEXT}-token window")
        print(f"\n[GENERATOR] Sending {len(messages) - 1} turns to '{self.model_name}' "
              f"(temp={temperature}, top_p={top_p}, max_tokens={max_tokens})")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else 0,
            top_p=top_p if top_p is not None else 1.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
