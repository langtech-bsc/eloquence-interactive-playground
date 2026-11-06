import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import httpx

from gradio_app.backend import rag_with_wiki as rag

ROOT = Path(__file__).resolve().parents[1]


class WikiRoutingTests(unittest.TestCase):
    def setUp(self):
        # Load the real handler without starting Gradio or loading UI dependencies.
        helpers = ModuleType("gradio_app.helpers")
        helpers.check_llm_interface = lambda name, interface, available_llms: (
            available_llms.get(name, {}).get("interface") == interface
        )
        spec = importlib.util.spec_from_file_location(
            "wiki_task_handler", ROOT / "gradio_app/backend/task_handlers.py"
        )
        self.handler = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"gradio_app.helpers": helpers}):
            spec.loader.exec_module(self.handler)
        self.llm = Mock(return_value="Agent final answer")
        self.llm.available_llms = {
            "Gemma + Tools": {"interface": "text", "api_endpoint": "http://127.0.0.1:9084/v1"},
            "Android model": {"interface": "text", "api_endpoint": "http://127.0.0.1:10002/v1"},
        }
        self.retriever = Mock()

    def run_task(self, config_name, index):
        config = json.loads((ROOT / "playground-data/configurations/task_configs" / config_name).read_text())
        with patch.object(self.handler, "answer_with_wikipedia", return_value="Answer") as answer:
            result = list(self.handler.LocalTaskHandler(self.llm, self.retriever, config)(
                "Android model", None, [["question", ""]], "question", 5, index,
            ))
        return result, answer

    def test_wiki_calls_agent_directly_without_local_tool_loop(self):
        result, local_loop = self.run_task("wiki.json", None)
        self.assertEqual(result, [("Agent final answer", [], [])])
        self.llm.assert_called_once_with(
            "Gemma + Tools", None, [["question", ""]], [],
            task_name="Wiki", user_input="question",
        )
        self.assertEqual(self.retriever.mock_calls, [])
        local_loop.assert_not_called()

    def test_rag_wiki_preserves_selected_model_and_chunks(self):
        self.retriever.search_with_scores.return_value = [
            {"text": "Local fact", "metadata": {"start_time": 12}, "distance": 1.2},
        ]
        result, local_loop = self.run_task("rag_with_wiki.json", "example")
        local_loop.assert_called_once()
        arguments = local_loop.call_args.kwargs
        self.llm.assert_not_called()
        self.assertEqual(result, [("Answer", ["Local fact"], [{"start_time": 12}])])
        self.assertEqual(arguments["model_config"]["api_endpoint"], "http://127.0.0.1:10002/v1")
        self.retriever.search_with_scores.assert_called_once()


class ModelRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_model_receives_tools_with_auto_choice(self):
        payloads = []

        def respond(request):
            self.assertEqual(str(request.url), "http://127.0.0.1:10002/v1/chat/completions")
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": "Hello!",
            }}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(rag.httpx, "AsyncClient", return_value=client):
            result = await rag.call_selected_model(
                model_config={"api_endpoint": "http://127.0.0.1:10002/v1", "model_name": "test-model"},
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{"type": "function", "function": {"name": "search_wikipedia"}}],
            )
        self.assertEqual(result["content"], "Hello!")
        self.assertEqual(payloads[0]["tool_choice"], "auto")
        self.assertEqual(payloads[0]["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
