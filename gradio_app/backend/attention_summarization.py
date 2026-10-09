"""Shared summarization dispatch and rendering, independent of Gradio."""

from __future__ import annotations

import gzip
import html
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import requests
from markdown_it import MarkdownIt

from gradio_app.messages import AttentionMessageSpan, RequestQueryLLM, ResponseQueryLLM, SummaryAttention


@lru_cache(maxsize=4)
def _load_attention_tokenizer(tokenizer_path):
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return Tokenizer.from_str(source.read())
    return Tokenizer.from_file(str(path))


def align_attention_offsets(payload, tokenizer_path):
    """Repair legacy offsets in the IP only after verifying the service's tokens."""
    if not isinstance(payload, dict):
        return payload  # Response schema validation reports malformed payloads.
    tokenizer = None
    aligned = dict(payload)
    for prefix in ("prompt", "summary"):
        text = payload.get(prefix)
        pieces = payload.get(prefix + "_token_text")
        if not isinstance(text, str) or not isinstance(pieces, list):
            continue
        if tokenizer is None:
            tokenizer = _load_attention_tokenizer(tokenizer_path)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = [tokenizer.decode([index], skip_special_tokens=False) for index in encoding.ids]
        if decoded != pieces or len(encoding.ids) != payload.get(prefix + "_length"):
            raise ValueError(
                f"Configured attention tokenizer does not match the service's {prefix} tokens."
            )
        aligned[prefix + "_offsets"] = [list(span) for span in encoding.offsets]
    return aligned


@dataclass
class ConversationSession:
    """Gradio creates a separate copy for each browser session."""

    history: list = field(default_factory=list)
    revision: int = 0
    summary_request: int = 0
    chat_running: bool = False
    attention: dict | None = None

    def invalidate(self):
        self.revision += 1
        self.attention = None
        self.chat_running = False

    def replace_history(self, history):
        self.invalidate()
        self.history = deepcopy(history)


class _MessageMarkup(HTMLParser):
    """Keep safe markup separate from the text whose characters receive attention."""

    _tags = set("a b strong em i u s del code pre p br hr blockquote ul ol li "
                "h1 h2 h3 h4 h5 h6 table thead tbody tfoot tr td th span div "
                "details summary sup sub img".split())
    _attributes = {
        "a": {"href", "title"}, "img": {"src", "alt", "title"},
        "code": {"class"}, "ol": {"start"}, "li": {"value"},
        "td": {"align", "colspan", "rowspan"},
        "th": {"align", "colspan", "rowspan"},
    }

    def __init__(self, message):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.text = ""
        self._after_break = False
        self._open_tags = []
        markup = MarkdownIt("commonmark", {"html": True, "breaks": True, "linkify": True}).enable(
            ["table", "strikethrough", "linkify"]
        ).render(message).strip()
        self.feed(markup)
        self.close()
        while self._open_tags:
            self.parts.append((f'</{self._open_tags.pop()}>', None))

    def handle_starttag(self, tag, attrs):
        if tag not in self._tags:
            self.handle_data(self.get_starttag_text())
            return
        safe = []
        for key, value in attrs:
            if value is None or key not in self._attributes.get(tag, set()):
                continue
            if key in {"href", "src"}:
                # Strip URL control characters before checking the scheme.
                value = "".join(char for char in value.strip() if ord(char) > 32)
                try:
                    scheme = urlsplit(value).scheme.lower()
                except ValueError:
                    continue
                if scheme not in {"", "http", "https", "mailto"}:
                    continue
            if key == "class" and not value.startswith("language-"):
                continue
            safe.append(f' {key}="{html.escape(value, quote=True)}"')
        self.parts.append((f'<{tag}{"".join(safe)}>', None))
        if tag not in {"br", "hr", "img"}:
            self._open_tags.append(tag)
        if tag == "br":
            self.handle_data("\n")
            self._after_break = True

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in {"br", "hr", "img"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self._open_tags:
            # Balance raw HTML so a stray closing tag cannot escape the bubble.
            while self._open_tags:
                opened = self._open_tags.pop()
                self.parts.append((f'</{opened}>', None))
                if opened == tag:
                    break
        elif tag not in {"br", "hr", "img"}:
            self.handle_data(f'</{tag}>')

    def handle_data(self, data):
        if self._after_break:
            data = data.removeprefix("\n")
            self._after_break = False
        start = len(self.text)
        self.text += data
        # A span between table rows/cells is invalid HTML and gets moved by the
        # browser. Keep this invisible formatting whitespace as a plain text node.
        table_spacing = (self._open_tags and self._open_tags[-1] in
                         {"table", "thead", "tbody", "tfoot", "tr"} and not data.strip())
        self.parts.append((data, None if table_spacing else (start, len(self.text))))

    def with_attention(self, prompt, offsets, start):
        return "".join(
            part if span is None else token_html(
                prompt, offsets, start + span[0], start + span[1]
            )
            for part, span in self.parts
        )


def build_summary_prompt(history, system_prompt=None):
    if any(len(turn) != 2 or any(message is not None and not isinstance(message, str) for message in turn)
           for turn in history):
        raise ValueError("Summarization requires text conversation turns.")
    if not history or not any((message or "").strip() for turn in history for message in turn):
        raise ValueError("No conversation to summarize.")
    system_prompt = (system_prompt or "").strip()
    prompt = system_prompt + "\n\n" if system_prompt else ""
    prompt += "Summarize the following conversation in a concise paragraph:\n\n"
    spans = []
    for turn_index, turn in enumerate(history):
        if turn_index:
            prompt += "\n\n"
        for side, label in enumerate(("User", "Assistant")):
            if side:
                prompt += "\n"
            prompt += label + ": "
            start = len(prompt)
            # Summarize the visible text; retain the original Markdown in history.
            prompt += _MessageMarkup(turn[side] or "").text
            spans.append(AttentionMessageSpan(
                turn_index=turn_index, side=side, start=start, end=len(prompt),
            ))
    # The sidecar strips the raw prompt; keep trailing message whitespace internal.
    prompt += "\n\nSummary:"
    return prompt, spans


def summarize_with_attention(llm_handler, history, system_prompt=None,
                             temperature=None, top_p=None, max_tokens=None):
    prompt, spans = build_summary_prompt(history, system_prompt)
    payload = llm_handler.summarize_with_attention(
        prompt,
        temperature=0.0 if temperature is None else temperature,
        top_p=1.0 if top_p is None else top_p,
        max_tokens=300 if max_tokens is None else max_tokens,
    )
    result = SummaryAttention.model_validate(payload)
    if result.prompt != prompt:
        raise ValueError("Attention service returned a different prompt; token mapping is invalid.")
    result.message_spans = spans
    return result


def query_summary(history, llm_name, system_prompt=None, temperature=None,
                  top_p=None, max_tokens=None, *, port=8080, timeout=315):
    """The browser callback submits the same /query request as API clients."""
    body = RequestQueryLLM(
        operation="summarize", task_config="Summarization",
        history=history, llm_name=llm_name or "", input_text="",
        system_prompt=system_prompt, docs_k=0, temp=temperature, top_p=top_p,
        max_tokens=max_tokens, index_name=None,
    )
    response = requests.post(
        f"http://127.0.0.1:{port}/query",
        data={"body": body.model_dump_json()}, timeout=(10, timeout),
    )
    response.raise_for_status()
    result = ResponseQueryLLM.model_validate(response.json())
    if result.error:
        raise RuntimeError(result.error)
    attention = result.summary_attention
    prompt, spans = build_summary_prompt(history, system_prompt)
    if (attention is None or result.text != attention.summary or attention.prompt != prompt
            or attention.message_spans != spans):
        raise ValueError("The query returned attention for a different conversation.")
    return attention


def token_html(text, offsets, start=0, end=None):
    """Render original text, grouping tokens that share a Unicode character."""
    end = len(text) if end is None else end
    events = {start: [], end: []}
    for index, (left, right) in enumerate(offsets):
        left, right = max(start, left), min(end, right)
        if left < right:
            events.setdefault(left, []).append((index, True))
            events.setdefault(right, []).append((index, False))
    active = set()
    positions = sorted(events)
    fragments = []
    for left, right in zip(positions, positions[1:]):
        for index, entering in events[left]:
            if entering:
                active.add(index)
            else:
                active.discard(index)
        piece = html.escape(text[left:right])
        if active:
            indices = ",".join(str(index) for index in sorted(active))
            fragments.append(f'<span class="attention-token" data-attn-indices="{indices}">{piece}</span>')
        else:
            fragments.append(piece)
    return "".join(fragments)


def render_attention(history, result):
    version = uuid.uuid4().hex
    rendered = deepcopy(history)
    for span in result.message_spans:
        if span.start == span.end:
            continue
        markup = _MessageMarkup(history[span.turn_index][span.side] or "")
        if markup.text != result.prompt[span.start:span.end]:
            raise ValueError("Conversation text no longer matches the attention prompt.")
        content = markup.with_attention(result.prompt, result.prompt_offsets, span.start)
        rendered[span.turn_index][span.side] = (
            f'<div class="attention-message" data-attn-id="{version}">{content}</div>'
        )
    # Character offsets are consumed in Python, avoiding JavaScript's UTF-16 indices.
    data = html.escape(json.dumps({
        "version": version,
        "prompt_length": result.prompt_length,
        "summary_length": result.summary_length,
        "attention": result.attention.model_dump(),
    }, separators=(",", ":")), quote=True)
    content = token_html(result.summary, result.summary_offsets)
    summary_html = (
        f'<div id="attention-summary" data-attention="{data}">'
        '<div class="attention-summary-label">See below a summary of the conversation.</div>'
        '<div class="attention-summary-hint">For explainability, select summary text to highlight '
        'conversation tokens according to the attention they receive. '
        'Darker yellow indicates higher attention. ESC clears the highlight. '
        'Attention is an interpretation, not a causal explanation.</div>'
        f'<div class="attention-summary-text">{content}</div>'
        '<div class="attention-summary-footer"><strong>'
        'Summary using Mistral-Nemo-Instruct-2407.</strong></div>'
        '</div>'
    )
    return summary_html, rendered
