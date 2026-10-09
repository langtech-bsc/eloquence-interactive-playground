import ast
import datetime
import json
import logging
from html.parser import HTMLParser
import os
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import gradio as gr
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from typing import List, Optional

from gradio_app.backend.attention_summarization import (
    ConversationSession, build_summary_prompt, render_attention,
    summarize_with_attention, query_summary, token_html,
    align_attention_offsets, _load_attention_tokenizer,
)
from gradio_app.messages import RequestQueryLLM, ResponseQueryLLM, SummaryAttention

ROOT = Path(__file__).resolve().parents[1]
NEMO_TOKENIZER = "assets/tokenizers/nemo/tokenizer.json.gz"


def legacy_nemo_payload(prompt, summary=" The user has a fever 😊."):
    """Reproduce the original service's per-token search, including Unicode drift."""
    tokenizer = _load_attention_tokenizer(NEMO_TOKENIZER)
    payload = {"model": "/models/nemo", "prompt": prompt, "summary": summary}
    for prefix in ("prompt", "summary"):
        text = payload[prefix]
        encoding = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([index], skip_special_tokens=False) for index in encoding.ids]
        pos, offsets = 0, []
        for piece in pieces:
            found = text.find(piece, pos)
            start = pos if found == -1 else found
            pos = start + len(piece)
            offsets.append([start, pos])
        payload[prefix + "_token_text"] = pieces
        payload[prefix + "_offsets"] = offsets
        payload[prefix + "_length"] = len(pieces)
    P, S = payload["prompt_length"], payload["summary_length"]
    payload["attention"] = {"indices": [[P + i, 0] for i in range(S)], "values": [1.0] * S}
    return payload


def load_definitions(path, names, namespace):
    """Execute actual callbacks without importing model/UI startup dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def sidecar_payload(prompt, summary="Tiene fiebre 😊."):
    # A controlled sidecar fixture: each character is a token, including Unicode.
    P = len(prompt)
    fever = prompt.index("fiebre")
    indices, values = [], []
    for j in range(len(summary)):
        indices.extend([[P + j, 0], [P + j, fever]])
        values.extend([0.75, 0.25])
    return {
        "model": "/models/nemo", "prompt": prompt, "summary": summary,
        "prompt_token_text": list(prompt),
        "prompt_offsets": [[i, i + 1] for i in range(P)], "prompt_length": P,
        "summary_token_text": list(summary),
        "summary_offsets": [[i, i + 1] for i in range(len(summary))],
        "summary_length": len(summary),
        "attention": {"indices": indices, "values": values},
        "aggregation": "fixture", "prompt_length_full": P,
    }


def ui_namespace(handler=None):
    handler = handler or Mock()

    def submit_query(*args, **kwargs):
        client, _ = query_api(handler)
        def post(url, data, **options):
            assert url.endswith("/query")
            return client.post("/query", data=data)
        with patch("gradio_app.backend.attention_summarization.requests.post", side_effect=post):
            return query_summary(*args, **kwargs)

    namespace = {
        "gr": gr, "json": json, "deepcopy": deepcopy, "os": os,
        "ConversationSession": ConversationSession,
        "render_attention": render_attention,
        "query_summary": Mock(side_effect=submit_query),
        "llm_handler": handler,
        "settings": SimpleNamespace(BASIC_CONFIG={}, ATTENTION_REQUEST_TIMEOUT=300),
        "List": List,
        "_dialogue_manager_session_id": lambda request: "session-test",
    }
    return load_definitions("gradio_app/app_handlers.py", {
        "summarize_conversation", "reset_attention_view", "interact_with_attention",
        "clear_conversation_with_attention", "load_history_with_attention",
        "store_history", "save_feedback",
    }, namespace)


def query_api(handler):
    app = FastAPI()
    namespace = {
        "app": app, "ResponseQueryLLM": ResponseQueryLLM,
        "RequestQueryLLM": RequestQueryLLM, "TypeAdapter": TypeAdapter,
        "ValidationError": ValidationError, "HTTPException": HTTPException, "json": json,
        "Form": Form, "File": File, "Optional": Optional, "UploadFile": UploadFile,
        "logger": logging.getLogger("attention-test"),
        "run_in_threadpool": run_in_threadpool,
        "summarize_with_attention": summarize_with_attention,
        "llm_handler": handler,
        "query_llm_general": AsyncMock(return_value=("Selected model answer", [], [], None, [])),
    }
    load_definitions("gradio_app/app.py", {"query_llm_endpoint"}, namespace)
    return TestClient(app), namespace


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.history = [["Tengo fiebre 😊 y <script>alert(1)</script>.  ", "¿Desde cuándo?"]]
        self.handler = Mock()
        self.handler.summarize_with_attention.side_effect = lambda prompt, **kwargs: sidecar_payload(prompt)

    def test_prompt_maps_visible_text_without_mutating_history(self):
        prompt, spans = build_summary_prompt(self.history, "Instrucción del sistema")
        self.assertTrue(prompt.startswith("Instrucción del sistema\n\n"))
        self.assertEqual(prompt, prompt.strip())
        for span in spans:
            self.assertEqual(prompt[span.start:span.end], self.history[span.turn_index][span.side].rstrip())

    def test_rendering_escapes_text_without_mutating_history(self):
        original = deepcopy(self.history)
        result = summarize_with_attention(self.handler, self.history)
        html, bubbles = render_attention(self.history, result)
        self.assertEqual(self.history, original)
        self.assertNotIn("<script>", bubbles[0][0])
        self.assertIn("&lt;", bubbles[0][0])
        self.assertIn("😊", html)
        self.assertIn("data-attn-indices", bubbles[0][1])
        self.assertIn("prompt_length_full", result.model_dump())
        self.assertIn("Summary using Mistral-Nemo-Instruct-2407", html)
        self.assertIn("ESC clears the highlight.", html)

    def test_rich_messages_keep_format_links_and_exact_character_indices(self):
        history = [["Tengo **fiebre** 😊 y [dolor](https://example.com?q=1&x=2).",
                    '<a href="#document-0" onclick="alert(1)">[0]</a> '
                    '`x < y` &amp; texto<br/>otra línea']]
        original = deepcopy(history)
        result = summarize_with_attention(self.handler, history)
        _, bubbles = render_attention(history, result)
        self.assertEqual(history, original)
        self.assertIn("<strong>", bubbles[0][0])
        self.assertIn('href="https://example.com?q=1&amp;x=2"', bubbles[0][0])
        self.assertIn('href="#document-0"', bubbles[0][1])
        self.assertIn("<code>", bubbles[0][1])
        self.assertNotIn("onclick", bubbles[0][1])
        self.assertIn("x < y & texto\notra línea", result.prompt)
        self.assertNotIn("**", result.prompt)
        self.assertNotIn("https://", result.prompt)

        class MappingParser(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.stack, self.characters = [], []

            def handle_starttag(self, tag, attrs):
                if tag in {"br", "hr", "img"}:
                    return
                attrs = dict(attrs)
                indices = attrs.get("data-attn-indices", "")
                self.stack.append(indices or (self.stack[-1] if self.stack else ""))

            def handle_endtag(self, tag):
                if self.stack:
                    self.stack.pop()

            def handle_data(self, data):
                self.characters.extend((char, self.stack[-1] if self.stack else "") for char in data)

        for span in result.message_spans:
            parser = MappingParser()
            parser.feed(bubbles[span.turn_index][span.side])
            self.assertEqual("".join(char for char, _ in parser.characters), result.prompt[span.start:span.end])
            for index, (_, tokens) in enumerate(parser.characters, span.start):
                self.assertEqual(tokens, str(index))

    def test_code_tables_and_unsafe_links(self):
        history = [["Tengo fiebre", '```python\nx < y && y > 0\n```\n\n'
                    '| A | B |\n|---|---|\n| uno | dos |\n\n'
                    '<a href="javascript:alert(1)" style="color:red">enlace</a>']]
        result = summarize_with_attention(self.handler, history)
        _, bubbles = render_attention(history, result)
        self.assertIn("x < y && y > 0", result.prompt)
        self.assertIn('<pre><code class="language-python">', bubbles[0][1])
        self.assertIn("<table>", bubbles[0][1])
        self.assertNotIn("javascript:", bubbles[0][1])
        self.assertNotIn("style=", bubbles[0][1])

    def test_markdown_dependencies_are_explicit_for_no_deps_docker_install(self):
        from importlib.metadata import requires
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        configured = {
            canonicalize_name(requirement.name): requirement
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            for requirement in [Requirement(line)]
        }
        for package in ("markdown-it-py", "linkify-it-py"):
            for item in requires(package):
                dependency = Requirement(item)
                if dependency.marker and not dependency.marker.evaluate():
                    continue
                name = canonicalize_name(dependency.name)
                self.assertIn(name, configured, f"Docker --no-deps requires an explicit {name} entry")
                pinned = next(iter(configured[name].specifier))
                self.assertEqual(pinned.operator, "==")
                self.assertIn(pinned.version, dependency.specifier)

    def test_soft_line_breaks_and_bare_urls_keep_visible_text(self):
        history = [["Tengo fiebre\nDesde ayer", "Consulta https://example.org"]]
        result = summarize_with_attention(self.handler, history)
        _, bubbles = render_attention(history, result)
        self.assertIn("Tengo fiebre\nDesde ayer", result.prompt)
        self.assertEqual(bubbles[0][0].count("<br>"), 1)
        self.assertIn('href="https://example.org"', bubbles[0][1])
        self.assertIn("Consulta https://example.org", result.prompt)

    def test_raw_html_is_balanced_inside_the_attention_message(self):
        history = [["Tengo fiebre", '</div><strong>Texto</strong><div>Final']]
        result = summarize_with_attention(self.handler, history)
        _, bubbles = render_attention(history, result)
        self.assertIn("&lt;", bubbles[0][1])
        self.assertEqual(bubbles[0][1].count("<div"), 2)
        self.assertEqual(bubbles[0][1].count("</div>"), 2)

    def test_render_rejects_mismatched_visible_text(self):
        result = summarize_with_attention(self.handler, self.history)
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            render_attention([["Changed message", "Reply"]], result)

    def test_shared_unicode_offsets_do_not_duplicate_characters(self):
        rendered = token_html("😊!", [(0, 1), (0, 1), (1, 2)])
        self.assertEqual(rendered.count("😊"), 1)
        self.assertIn('data-attn-indices="0,1"', rendered)

    def test_invalid_metadata_and_changed_prompt_are_rejected(self):
        prompt, _ = build_summary_prompt(self.history)
        payload = sidecar_payload(prompt)
        payload["summary_length"] += 1
        with self.assertRaises(ValidationError):
            SummaryAttention.model_validate(payload)
        payload = sidecar_payload(prompt)
        payload["attention"]["values"][0] = float("nan")
        with self.assertRaises(ValidationError):
            SummaryAttention.model_validate(payload)
        self.handler.summarize_with_attention.side_effect = lambda prompt, **kwargs: sidecar_payload(prompt + "changed")
        with self.assertRaisesRegex(ValueError, "different prompt"):
            summarize_with_attention(self.handler, self.history)

    def test_empty_conversations_are_rejected(self):
        for history in ([], [["", ""]]):
            with self.assertRaisesRegex(ValueError, "No conversation"):
                build_summary_prompt(history)

    def test_interactor_calls_custom_endpoint_and_maps_parameters(self):
        response = Mock()
        response.json.return_value = {"summary": "fixture"}
        http = SimpleNamespace(post=Mock(return_value=response), RequestException=requests.RequestException)
        ns = load_definitions("gradio_app/backend/BSCInteract.py", {"AttentionSummarizationInteractor"}, {"requests": http})
        interactor = ns["AttentionSummarizationInteractor"]("http://localhost:5010/")
        result = interactor.summarize("prompt", max_tokens=150, temperature=0.2, top_p=0.9)
        self.assertEqual(result, {"summary": "fixture"})
        arguments = http.post.call_args
        self.assertEqual(arguments.args[0], "http://localhost:5010/summarize_with_attention")
        self.assertEqual(arguments.kwargs["params"], {"model": "nemo"})
        self.assertEqual(arguments.kwargs["json"], {
            "prompt": "prompt", "max_new_tokens": 150,
            "temperature": 0.2, "top_p": 0.9, "top_k": 16,
        })
        http.post.side_effect = requests.ConnectionError("service unavailable")
        with self.assertRaisesRegex(RuntimeError, "Nemo attention summarization failed"):
            interactor.summarize("prompt")

    def test_model_configuration_selects_service_endpoint_and_model(self):
        http = SimpleNamespace(post=Mock())
        http.RequestException = requests.RequestException
        http.post.return_value.json.return_value = {"summary": "Configured service"}
        interactor = load_definitions("gradio_app/backend/BSCInteract.py",
                                     {"AttentionSummarizationInteractor"},
                                     {"requests": http, "align_attention_offsets": align_attention_offsets})
        tree = ast.parse((ROOT / "gradio_app/backend/query_llm.py").read_text())
        namespace = {
            alias.name: Mock for node in tree.body if isinstance(node, ast.ImportFrom)
            and node.module == "gradio_app.backend.BSCInteract" for alias in node.names
        }
        namespace.update(interactor)
        namespace.update(settings=SimpleNamespace(ATTENTION_REQUEST_TIMEOUT=75), logger=logging.getLogger(__name__))
        handler_class = load_definitions("gradio_app/backend/query_llm.py", {"LLMHandler"}, namespace)["LLMHandler"]
        models = [{k: v for k, v in entry.items()} for entry in
                  json.loads((ROOT / "playground-data/configurations/models.json").read_text())
                  if entry.get("interactor") == "attention_summarization"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["interface"], "service")
        entry = dict(models[0], api_endpoint="http://localhost:5432/summarize_with_attention", model_name="mounted-nemo")
        handler = handler_class({"Configured attention": entry})
        self.assertEqual(handler.summarize_with_attention("prompt"), {"summary": "Configured service"})
        kwargs = http.post.call_args.kwargs
        self.assertEqual(http.post.call_args.args[0], entry["api_endpoint"])
        self.assertEqual(kwargs["params"], {"model": "mounted-nemo"})
        self.assertEqual(kwargs["timeout"], (10, 75))
        with self.assertRaisesRegex(ValueError, "models.json"):
            handler_class({}).summarize_with_attention("prompt")

        # Load the actual registry expression without importing UI startup.
        configured_models = json.loads(
            (ROOT / "playground-data/configurations/models.json").read_text())
        app_tree = ast.parse((ROOT / "gradio_app/app_handlers.py").read_text())
        registry = [node for node in ast.walk(app_tree)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.DictComp)
                    and any(isinstance(target, ast.Name) and target.id == "available_llms"
                            for target in node.targets)]
        self.assertEqual(len(registry), 1)
        loaded = {"available_llms": configured_models}
        exec(compile(ast.Module(body=registry, type_ignores=[]), "app_handlers.py", "exec"), loaded)
        model_name = "Mistral-Nemo-Instruct-2407"
        service = loaded["available_llms"]["Nemo"]
        self.assertEqual(service["model_name"], model_name)
        self.assertEqual(service["display_name"], "Nemo")
        for model in configured_models:
            if model.get("display_name"):
                self.assertEqual(loaded["available_llms"][model["display_name"]], model)
        # Verify the registry fallback with a service without display_name.
        unnamed_service = {key: value for key, value in service.items() if key != "display_name"}
        fallback = {"available_llms": [unnamed_service]}
        exec(compile(ast.Module(body=registry, type_ignores=[]), "app_handlers.py", "exec"), fallback)
        self.assertEqual(fallback["available_llms"][model_name], unnamed_service)
        # Both Nemo spellings must reuse the preloaded service alias.
        for name in (model_name, "mistralai/" + model_name):
            configured_service = dict(service, model_name=name, tokenizer_path=None)
            configured_handler = handler_class({"Nemo": configured_service})
            configured_handler.summarize_with_attention("prompt")
            self.assertEqual(http.post.call_args.kwargs["params"], {"model": "nemo"})

    def test_legacy_nemo_offsets_are_repaired_without_altering_tokens_or_attention(self):
        prompt, _ = build_summary_prompt(self.history)
        payload = legacy_nemo_payload(prompt)
        original = deepcopy(payload)
        with self.assertRaisesRegex(ValidationError, "invalid prompt offsets"):
            SummaryAttention.model_validate(payload)
        aligned = align_attention_offsets(payload, NEMO_TOKENIZER)
        result = SummaryAttention.model_validate(aligned)
        self.assertEqual(payload, original)
        self.assertEqual(result.attention.model_dump(mode="json"), payload["attention"])
        for prefix in ("prompt", "summary"):
            self.assertEqual(aligned[prefix + "_token_text"], original[prefix + "_token_text"])
            self.assertEqual(aligned[prefix + "_length"], original[prefix + "_length"])
        self.handler.summarize_with_attention.return_value = aligned
        self.handler.summarize_with_attention.side_effect = None
        summary, rendered = render_attention(self.history, summarize_with_attention(self.handler, self.history))
        self.assertEqual(rendered[0][0].count("😊"), 1)
        self.assertEqual(summary.count("😊"), 1)

    def test_tokenizer_mismatch_is_rejected_instead_of_clamping_offsets(self):
        prompt, _ = build_summary_prompt(self.history)
        payload = legacy_nemo_payload(prompt)
        payload["prompt_token_text"][0] = "wrong token"
        with self.assertRaisesRegex(ValueError, "does not match.*prompt tokens"):
            align_attention_offsets(payload, NEMO_TOKENIZER)
        payload = legacy_nemo_payload(prompt)
        payload["summary_length"] += 1
        with self.assertRaisesRegex(ValueError, "does not match.*summary tokens"):
            align_attention_offsets(payload, NEMO_TOKENIZER)

    def test_interactor_repairs_original_service_response_inside_ip(self):
        prompt, _ = build_summary_prompt(self.history)
        http = SimpleNamespace(post=Mock(), RequestException=requests.RequestException)
        http.post.return_value.json.return_value = legacy_nemo_payload(prompt)
        cls = load_definitions("gradio_app/backend/BSCInteract.py", {"AttentionSummarizationInteractor"},
                               {"requests": http, "align_attention_offsets": align_attention_offsets})[
                                   "AttentionSummarizationInteractor"]
        interactor = cls("http://localhost:5010/summarize_with_attention", tokenizer_path=NEMO_TOKENIZER)
        self.handler.summarize_with_attention.side_effect = interactor.summarize
        client, _ = self.api_client()
        response = client.post("/query", data={"body": json.dumps(dict(self.api_body(), operation="summarize"))})
        self.assertEqual(response.status_code, 200)
        result = ResponseQueryLLM.model_validate(response.json())
        self.assertIsNone(result.error)
        self.assertEqual(len(result.summary_attention.message_spans), 2)

    def test_query_client_sends_form_operation_and_preserves_metadata(self):
        result = summarize_with_attention(self.handler, self.history)
        response = Mock()
        response.json.return_value = ResponseQueryLLM(
            text=result.summary, documents=[], summary_attention=result,
        ).model_dump()
        with patch("gradio_app.backend.attention_summarization.requests.post", return_value=response) as post:
            self.assertEqual(query_summary(self.history, "Selected conversation model", port=8082), result)
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8082/query")
        body = json.loads(post.call_args.kwargs["data"]["body"])
        self.assertEqual(body["operation"], "summarize")
        self.assertEqual(body["task_config"], "Summarization")
        self.assertEqual(body["history"], self.history)
        self.assertEqual(body["llm_name"], "Selected conversation model")
        response.json.return_value["error"] = "Attention service unavailable"
        with patch("gradio_app.backend.attention_summarization.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                query_summary(self.history, "Selected conversation model")

    def api_client(self):
        return query_api(self.handler)

    def api_body(self):
        return {
            "history": self.history, "input_text": "", "llm_name": "Selected conversation model",
            "task_config": "Summarization", "docs_k": 0,
            "temp": 0, "top_p": 1, "max_tokens": 150, "index_name": None,
        }

    def test_query_summary_returns_full_metadata_and_uses_nemo(self):
        client, namespace = self.api_client()
        body = dict(self.api_body(), operation="summarize")
        response = client.post("/query", data={"body": json.dumps(body)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "Tiene fiebre 😊.")
        self.assertEqual(payload["summary_attention"]["model"], "/models/nemo")
        self.assertEqual(len(payload["summary_attention"]["message_spans"]), 2)
        namespace["query_llm_general"].assert_not_awaited()

    def test_query_default_keeps_chat_path_and_does_not_call_nemo(self):
        client, namespace = self.api_client()
        response = client.post("/query", data={"body": json.dumps(self.api_body())}).json()
        self.assertEqual(response["text"], "Selected model answer")
        self.assertIsNone(response["summary_attention"])
        self.handler.summarize_with_attention.assert_not_called()
        namespace["query_llm_general"].assert_awaited_once()

    def test_query_rejects_wrong_task_and_reports_service_errors(self):
        client, _ = self.api_client()
        body = dict(self.api_body(), operation="summarize", task_config="Basic LLM")
        result = client.post("/query", data={"body": json.dumps(body)}).json()
        self.assertIn("requires the Summarization task", result["error"])
        self.handler.summarize_with_attention.side_effect = RuntimeError("Nemo unavailable")
        body["task_config"] = "Summarization"
        result = client.post("/query", data={"body": json.dumps(body)}).json()
        self.assertEqual(result["error"], "Nemo unavailable")
        self.assertEqual(result["text"], "")

    def test_query_invalid_operation_returns_validation_error(self):
        client, _ = self.api_client()
        body = dict(self.api_body(), operation="unsupported")
        response = client.post("/query", data={"body": json.dumps(body)})
        self.assertEqual(response.status_code, 422)
        self.handler.summarize_with_attention.assert_not_called()

    def test_absent_assistant_message_stays_absent_when_rendered(self):
        history = [["Tengo fiebre", None]]
        result = summarize_with_attention(self.handler, history)
        _, bubbles = render_attention(history, result)
        self.assertIsNone(bubbles[0][1])

    def summary_generator(self, namespace, session, task="Summarization"):
        return namespace["summarize_conversation"](
            session, "Selected conversation model", json.dumps({"name": task, "interface": "text"}),
            "", 0, 1, 150, None,
        )

    def test_ui_keeps_canonical_history_and_session_isolation(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        other = deepcopy(session)
        result = list(self.summary_generator(namespace, session))[-1]
        self.assertIn("attention-summary", result[1]["value"])
        self.assertIn("attention-message", result[2][0][0])
        self.assertEqual(session.history, self.history)
        self.assertIsNotNone(session.attention)
        self.assertIsNone(other.attention)
        reset = namespace["reset_attention_view"](session, '{"name":"Summarization"}')
        self.assertEqual(reset[0], self.history)
        self.assertIsNone(session.attention)

    def test_stale_summary_is_not_applied(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        generator = self.summary_generator(namespace, session)
        next(generator)
        session.replace_history([["New conversation", "Reply"]])
        self.assertEqual(list(generator), [])
        self.assertIsNone(session.attention)

    def test_summary_is_blocked_during_streaming_and_uses_completed_history(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        partial = self.history + [["Nueva pregunta", "Respuesta parcial"]]
        complete = self.history + [["Nueva pregunta", "Respuesta completa"]]
        namespace["interact"] = Mock(return_value=iter([
            (partial, "", gr.update(), gr.update()),
            (complete, "", gr.update(), gr.update()),
        ]))
        stream = namespace["interact_with_attention"](
            session, "Nueva pregunta", "Selected", 0, 0, 1, 150, "", "", '{}',
        )
        next(stream)
        next(stream)
        self.assertTrue(session.chat_running)
        with self.assertRaisesRegex(gr.Error, "finish responding"):
            list(self.summary_generator(namespace, session))
        namespace["query_summary"].assert_not_called()
        list(stream)
        self.assertFalse(session.chat_running)
        result = list(self.summary_generator(namespace, session))[-1]
        self.assertIn("Respuesta completa", session.attention["prompt"])
        self.assertEqual(session.history, complete)
        self.assertIn("attention-message", result[2][-1][1])

    def test_chat_lock_is_released_on_error_and_not_by_obsolete_stream(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        namespace["interact"] = Mock(side_effect=RuntimeError("chat failed"))
        args = (session, "Question", "Selected", 0, 0, 1, 150, "", "", '{}')
        failing = namespace["interact_with_attention"](*args)
        next(failing)
        with self.assertRaisesRegex(RuntimeError, "chat failed"):
            next(failing)
        self.assertFalse(session.chat_running)
        first = namespace["interact_with_attention"](*args)
        second = namespace["interact_with_attention"](*args)
        next(first)
        next(second)
        first.close()
        self.assertTrue(session.chat_running)
        second.close()
        self.assertFalse(session.chat_running)

    def test_history_snapshot_change_without_revision_discards_summary(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        generator = self.summary_generator(namespace, session)
        next(generator)
        session.history[0][1] = "Completed reply"
        self.assertEqual(list(generator), [])
        self.assertIsNone(session.attention)

    def test_newer_summary_wins(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        first = self.summary_generator(namespace, session)
        second = self.summary_generator(namespace, session)
        next(first)
        next(second)
        self.assertEqual(list(first), [])
        self.assertEqual(len(list(second)), 1)

    def test_dm_keeps_dialogue_manager_summary(self):
        self.handler.get_llm_generator.return_value.end.return_value = {"summary": "DM summary"}
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        result = list(self.summary_generator(namespace, session, task="DM"))[-1]
        self.assertEqual(result[0]["value"], "DM summary")
        self.assertFalse(result[1]["visible"])
        self.handler.summarize_with_attention.assert_not_called()

    def test_normal_conversation_uses_original_history_and_selected_model(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history), attention={"old": True})
        new_history = self.history + [["Nueva pregunta", "Respuesta"]]
        namespace["interact"] = Mock(return_value=iter([(new_history, "", gr.update(), gr.update())]))
        list(namespace["interact_with_attention"](
            session, "Nueva pregunta", "Selected conversation model", 0, 0, 1, 150,
            "", "", '{"name":"Summarization"}',
        ))
        self.assertEqual(namespace["interact"].call_args.args[0], self.history)
        self.assertEqual(namespace["interact"].call_args.args[2], "Selected conversation model")
        self.assertEqual(session.history, new_history)
        self.assertIsNone(session.attention)

    def test_save_uses_canonical_history_after_highlighting(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history))
        list(self.summary_generator(namespace, session))
        saved = Mock()
        namespace.update({
            "datetime": datetime, "USER_HISTORY_FILE": "history.json",
            "_get_user_filepath": Mock(return_value="unused-test-path"),
            "_load_json": Mock(return_value=[]), "_save_json": saved,
            "_build_history_choices": lambda logs: [],
        })
        namespace["store_history"](SimpleNamespace(username="test"), session, "System prompt")
        self.assertEqual(saved.call_args.args[1][0]["history"], self.history)

    def test_clear_and_load_replace_canonical_history_and_remove_attention(self):
        namespace = ui_namespace(self.handler)
        session = ConversationSession(history=deepcopy(self.history), attention={"old": True})
        namespace["clear_conversation"] = Mock(return_value=([], "", gr.update(), "", gr.update()))
        namespace["clear_conversation_with_attention"]('{"name":"Summarization"}', "Selected", session, None)
        self.assertEqual(session.history, [])
        self.assertIsNone(session.attention)
        loaded = [["Another saved conversation", "Reply"]]
        namespace["load_history_confirm"] = Mock(return_value=(loaded, "", gr.update()))
        namespace["load_history_with_attention"](None, "0", session)
        self.assertEqual(session.history, loaded)
        self.assertIsNot(session.history, loaded)


if __name__ == "__main__":
    unittest.main()
