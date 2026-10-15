"""Language routing and browser-session isolation without loading model clients."""

import ast
import json
import logging
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def model_config():
    return {
        "model_name": "NurseLLM", "interactor": "dialogue_manager",
        "api_endpoint": "http://127.0.0.1:8004", "interface": "text",
        "default_language": "sr",
        "language_endpoints": {
            "sr": "http://127.0.0.1:8004", "en": "http://127.0.0.1:8005",
        },
    }


def handler_fixture():
    requests = Mock()
    requests.post.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"response": "answer"}, raise_for_status=lambda: None,
    )
    namespace = {"requests": requests, "quote": quote}
    definitions("gradio_app/backend/BSCInteract.py", {"DialogueManagerInteractor"}, namespace)
    source = ast.parse((ROOT / "gradio_app/backend/query_llm.py").read_text())
    clients = {
        alias.name: Mock()
        for node in source.body if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    clients.update(namespace)
    clients.update({"deepcopy": deepcopy, "logger": logging.getLogger(__name__)})
    definitions("gradio_app/backend/query_llm.py", {"LLMHandler"}, clients)
    return clients["LLMHandler"]({"NurseLLM": model_config()}), requests


def ui_fixture(handler):
    gr = SimpleNamespace(
        Request=SimpleNamespace, Error=RuntimeError,
        update=lambda **kwargs: kwargs, Radio=lambda **kwargs: kwargs,
    )
    namespace = {
        "__name__": __name__, "gr": gr, "json": json, "deepcopy": deepcopy,
        "dataclass": dataclass, "field": field, "llm_handler": handler,
        "os": os, "ThreadPoolExecutor": ThreadPoolExecutor,
        "dynamic_data": {}, "logger": logging.getLogger(__name__),
        "LANGUAGE_CHOICES": [("English", "en"), ("Spanish", "es")],
    }
    definitions("gradio_app/backend/attention_summarization.py", {"ConversationSession"}, namespace)
    definitions("gradio_app/app_handlers.py", {
        "_dialogue_manager_session_id", "clear_conversation", "change_dm_language",
        "clear_conversation_with_attention", "load_task",
        "interact_with_attention", "_get_online_models",
    }, namespace)
    return namespace


class DialogueManagerLanguageTests(unittest.TestCase):
    def setUp(self):
        self.handler, self.requests = handler_fixture()

    def chat(self, language=None, session="browser-a"):
        return self.handler(
            "NurseLLM", "", [], [], task_name="DM", language=language,
            session_id=f"{session}-{language or 'sr'}", user_input="Hello",
        )

    def test_default_serbian_and_english_use_their_distinct_chat_routes(self):
        original = deepcopy(self.handler.available_llms)
        self.chat()
        self.chat("en")
        self.chat("sr")
        self.assertEqual([call.args[0] for call in self.requests.post.call_args_list], [
            "http://127.0.0.1:8004/v1/chat/uns_dm",
            "http://127.0.0.1:8005/v1/chat/uns_dm_eng",
            "http://127.0.0.1:8004/v1/chat/uns_dm",
        ])
        self.assertEqual(len(self.handler._cache), 2)
        self.assertEqual(self.handler.available_llms, original)

    def test_simultaneous_users_do_not_change_each_others_endpoint(self):
        barrier = Barrier(2)

        def post(url, **kwargs):
            barrier.wait(timeout=5)
            return SimpleNamespace(json=lambda: {"response": url}, raise_for_status=lambda: None)

        self.requests.post.side_effect = post
        with ThreadPoolExecutor(max_workers=2) as pool:
            english = pool.submit(self.chat, "en", "english-user")
            serbian = pool.submit(self.chat, "sr", "serbian-user")
            self.assertEqual(english.result(), "http://127.0.0.1:8005/v1/chat/uns_dm_eng")
            self.assertEqual(serbian.result(), "http://127.0.0.1:8004/v1/chat/uns_dm")

    def test_unsupported_language_is_rejected_before_sending_a_request(self):
        with self.assertRaisesRegex(ValueError, "No dialogue-manager endpoint"):
            self.chat("es")
        self.requests.post.assert_not_called()

    def test_clear_resets_only_the_selected_language_and_browser(self):
        ui = ui_fixture(self.handler)
        session = ui["ConversationSession"](history=[["Hello", "Reply"]], attention={"old": True})
        request = SimpleNamespace(session_hash="browser-a")
        ui["clear_conversation_with_attention"](
            '{"name":"DM"}', "NurseLLM", session, request, dm_language="en",
        )
        self.requests.post.assert_called_once_with(
            "http://127.0.0.1:8005/reset/gradio-browser-a-en", timeout=30,
        )
        self.assertEqual(session.history, [])
        self.assertIsNone(session.attention)
        self.assertNotEqual(
            ui["_dialogue_manager_session_id"](request, "en"),
            ui["_dialogue_manager_session_id"](request, "sr"),
        )
        self.assertNotEqual(
            ui["_dialogue_manager_session_id"](request, "en"),
            ui["_dialogue_manager_session_id"](SimpleNamespace(session_hash="browser-b"), "en"),
        )

    def test_language_change_clears_history_and_invalidates_in_flight_results(self):
        ui = ui_fixture(self.handler)
        session = ui["ConversationSession"](
            history=[["Zdravo", "Odgovor"]], revision=3,
            attention={"old": True}, chat_running=True,
        )
        outputs = ui["change_dm_language"](
            '{"name":"DM"}', "NurseLLM", session, "en",
            SimpleNamespace(session_hash="browser-a"),
        )
        self.assertEqual(outputs[0], [])
        self.assertEqual(session.history, [])
        self.assertGreater(session.revision, 3)
        self.assertFalse(session.chat_running)
        self.assertIsNone(session.attention)
        self.assertEqual(outputs[2]["value"], "")
        self.assertEqual(outputs[3]["value"], "")
        self.assertEqual(self.requests.post.call_args.args[0],
                         "http://127.0.0.1:8005/reset/gradio-browser-a-en")

    def test_dm_selector_overrides_audio_language_without_affecting_transcription(self):
        ui = ui_fixture(self.handler)
        ui["interact"] = Mock(return_value=iter([]))
        for task, expected in (("DM", "en"), ("Transcription", "es")):
            session = ui["ConversationSession"]()
            list(ui["interact_with_attention"](
                session, "Hello", "NurseLLM", 0, 0, 1, 150, None, "",
                json.dumps({"name": task}), language="es", dm_language="en",
            ))
            self.assertEqual(ui["interact"].call_args.args[10], expected)
        self.assertTrue(ui["load_task"]('{"name":"DM"}')[-1]["visible"])
        self.assertFalse(ui["load_task"]('{"name":"Transcription","interface":"audio"}')[-1]["visible"])

    def test_whisper_keeps_per_request_languages_and_a_single_cached_client(self):
        class AudioClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def set_params(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return kwargs["language"]

        self.handler.available_llms["Whisper"] = {
            "model_name": "whisper-large-v3", "interactor": "whisper",
            "api_endpoint": "http://localhost:9080/v1",
        }
        self.handler.INTERACTOR_CLASSES["whisper"] = AudioClient
        for language in ("en", "es", "sr"):
            self.assertEqual(self.handler("Whisper", "", [], [], language=language), language)
        self.assertEqual(len(self.handler._cache), 1)
        self.assertNotIn("language", self.handler._cache["Whisper"].kwargs)

    def test_nurse_is_available_when_only_the_english_service_is_online(self):
        ui = ui_fixture(self.handler)
        requests = SimpleNamespace(
            RequestException=Exception,
            get=Mock(side_effect=lambda url, **kwargs: SimpleNamespace(ok=":8005/" in url)),
        )
        with patch.dict("sys.modules", {"requests": requests}), patch.dict(os.environ, {"ELOQ_PROBE_MODELS_ON_LOAD": "true"}):
            _, online = ui["_get_online_models"](self.handler.available_llms)
        self.assertEqual(online, ["NurseLLM"])

    def test_model_registry_has_one_nurse_with_both_languages(self):
        entries = json.loads((ROOT / "playground-data/configurations/models.json").read_text())
        nurses = [entry for entry in entries if entry.get("interactor") == "dialogue_manager"]
        self.assertEqual(len(nurses), 1)
        self.assertEqual(nurses[0]["display_name"], "NurseLLM")
        self.assertEqual(nurses[0]["language_endpoints"], model_config()["language_endpoints"])


if __name__ == "__main__":
    unittest.main()
