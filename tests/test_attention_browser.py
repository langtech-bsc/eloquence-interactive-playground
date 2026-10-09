"""Optional browser test: install Playwright and Chromium to run it."""

import ast
import json
import re
import socket
import threading
from copy import deepcopy
from unittest.mock import Mock

import gradio as gr
import pytest

playwright = pytest.importorskip("playwright.sync_api")

from test_attention_summarization import ROOT, sidecar_payload, ui_namespace
from gradio_app.backend.attention_summarization import (
    ConversationSession, build_summary_prompt, render_attention, _load_attention_tokenizer,
)
from gradio_app.messages import SummaryAttention


def production_js():
    tree = ast.parse((ROOT / "settings.py").read_text())
    general = next(node.value.value for node in ast.walk(tree)
                   if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == "JS_CODE")
    attention = (ROOT / "assets/attention.js").read_text()
    return f"async () => {{ await ({general})(); ({attention})(); }}"


def test_selection_in_real_gradio_bubbles_and_stale_results():
    history = [["Tengo fiebre 😊 y <script>alert(1)</script>.",
                '**Recomendación:** consulta [la guía](https://example.com). '
                '<a href="#document-0">[0]</a> y `x < y`.\n'
                'Más información: https://example.org']]
    handler = Mock()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    slow = False

    def summarize(prompt, **kwargs):
        if slow:
            entered.set()
            assert release.wait(15)
            finished.set()
        return sidecar_payload(prompt)

    handler.summarize_with_attention.side_effect = summarize
    namespace = ui_namespace(handler)
    namespace["interact"] = Mock(side_effect=lambda raw, question, *args: iter([
        (raw + [[question, "Respuesta del modelo seleccionado"]], "", gr.update(), gr.update()),
    ]))
    with gr.Blocks(js=production_js(), css=(ROOT / "assets/attention.css").read_text()) as demo:
        session = gr.State(ConversationSession(history=deepcopy(history)))
        task = gr.State('{"name":"Summarization","interface":"text"}')
        system = gr.State("")
        temperature, top_p, maximum = gr.State(0), gr.State(1), gr.State(150)
        model = gr.Radio(["Selected conversation model"], value="Selected conversation model", elem_id="llm_name")
        chat = gr.Chatbot(value=history, sanitize_html=False, elem_id="chatbot")
        summary_box = gr.Textbox(visible=False)
        summary_html = gr.HTML()
        summarize_button = gr.Button("Summarize conversation", elem_id="summarize_btn")
        text = gr.Textbox(elem_id="input_textbox")
        submit = gr.Button("Submit", elem_id="submit_btn")
        context, rag = gr.HTML(visible=False), gr.Column(visible=False)
        docs, index, language, mode, text_model = gr.State(0), gr.State(""), gr.State(None), gr.State(None), gr.State(None)
        summarize_button.click(
            namespace["summarize_conversation"],
            [session, model, task, system, temperature, top_p, maximum],
            [summary_box, summary_html, chat, session],
        )
        submit.click(
            namespace["interact_with_attention"],
            [session, text, model, docs, temperature, top_p, maximum, index, system, task, language, mode, text_model],
            [chat, context, rag, text, session, summary_box, summary_html],
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
        with playwright.sync_playwright() as runner:
            try:
                browser = runner.chromium.launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Playwright Chromium is not installed")
                raise
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_selector("#chatbot strong")
            assert page.locator("#chatbot a").count() == 3
            # Keep an input-focus retry pending, then select before it can fire.
            page.locator('#input_textbox textarea').evaluate("""node => {
                node.disabled = true;
                node.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
            }""")
            # Queue the production reply-focus debounce before attention mounts.
            page.locator('#chatbot [aria-label="chatbot conversation"]').evaluate(
                "node => node.appendChild(document.createElement('span'))"
            )
            page.wait_for_timeout(20)
            page.locator("#summarize_btn").click()
            page.wait_for_selector("#attention-summary .attention-token")
            assert page.locator("#chatbot .attention-message").first.inner_text() == history[0][0]
            assert page.locator("#chatbot script").count() == 0
            assert page.locator("#chatbot strong").count() == 1
            assert page.locator("#chatbot br").count() == 1
            assert page.locator('#chatbot a[href="https://example.org"]').inner_text() == "https://example.org"
            assert page.locator("#chatbot a").count() == 3
            assert page.locator("#chatbot code").inner_text() == "x < y"
            assert page.locator('#chatbot a[href="https://example.com"]').inner_text() == "la guía"
            page.locator('#chatbot a[href="https://example.com"]').evaluate(
                "node => node.addEventListener('click', event => { event.preventDefault(); window.linkClicked = true; })"
            )
            page.locator('#chatbot a[href="https://example.com"]').click()
            assert page.evaluate("window.linkClicked") is True
            assert page.locator('#input_textbox textarea').is_disabled()
            page.locator(".attention-summary-text").evaluate("""node => {
                document.querySelector('#input_textbox textarea').disabled = false;
                const tokens = node.querySelectorAll('.attention-token');
                const range = document.createRange();
                range.setStart(tokens[6].firstChild, 0);
                range.setEnd(tokens[11].firstChild, 1);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                document.dispatchEvent(new Event('selectionchange'));
            }""")
            page.wait_for_function("""() => [...document.querySelectorAll('#chatbot .attention-token')]
                .some(span => span.style.backgroundColor)""")
            highlighted = page.locator("#chatbot .attention-token").evaluate_all(
                "nodes => nodes.filter(n => n.style.backgroundColor).map(n => [n.textContent, n.style.backgroundColor])"
            )
            # Pending input timers must not erase an immediate selection.
            page.wait_for_timeout(600)
            assert page.evaluate("window.getSelection().toString()") == "fiebre"
            assert page.locator('#chatbot .attention-token[style*="background"]').count() == 1
            assert len(highlighted) == 1
            assert highlighted[0][0] == "f"
            assert highlighted[0][1] == "rgba(255, 190, 40, 0.682)"
            page.keyboard.press("Escape")
            page.wait_for_function("""() => [...document.querySelectorAll('#chatbot .attention-token')]
                .every(span => !span.style.backgroundColor)""")
            other = browser.new_page()
            other.goto(f"http://127.0.0.1:{port}")
            other.wait_for_selector("#summarize_btn")
            assert other.locator("#attention-summary").count() == 0

            # A new chat message completes while an older summary is still running.
            slow = True
            page.locator("#summarize_btn").click()
            page.wait_for_function("() => document.body.innerText.includes('Generating summary with Mistral-Nemo-Instruct-2407')")
            assert entered.wait(5)
            page.locator("#input_textbox textarea").fill("Nueva pregunta")
            page.locator("#submit_btn").click()
            page.wait_for_function("() => document.querySelector('#chatbot').innerText.includes('Nueva pregunta')")
            release.set()
            assert finished.wait(5)
            page.wait_for_timeout(500)
            assert page.locator("#attention-summary").count() == 0
            assert page.locator("#chatbot .attention-message").count() == 0
            assert namespace["interact"].call_args.args[0] == history
            assert namespace["interact"].call_args.args[2] == "Selected conversation model"
            assert "Nueva pregunta" not in other.locator("#chatbot").inner_text()
            # The reverse race: summary clicked while the chat is streaming.
            chat_entered, chat_release = threading.Event(), threading.Event()
            def stream(raw, question, *args):
                yield raw + [[question, "Respuesta parcial"]], "", gr.update(), gr.update()
                chat_entered.set()
                assert chat_release.wait(15)
                yield raw + [[question, "Respuesta completa y definitiva"]], "", gr.update(), gr.update()

            namespace["interact"].side_effect = stream
            slow = False
            calls_before = handler.summarize_with_attention.call_count
            page.locator("#input_textbox textarea").fill("Pregunta con streaming")
            page.locator("#submit_btn").click()
            page.wait_for_function("() => document.querySelector('#chatbot').innerText.includes('Respuesta parcial')")
            assert chat_entered.wait(5)
            page.locator("#summarize_btn").click()
            page.wait_for_function("() => document.body.innerText.includes('finish responding')")
            assert handler.summarize_with_attention.call_count == calls_before
            chat_release.set()
            page.wait_for_function("() => document.querySelector('#chatbot').innerText.includes('Respuesta completa y definitiva')")
            page.wait_for_timeout(300)
            page.locator("#summarize_btn").click()
            page.wait_for_selector("#attention-summary .attention-token")
            assert "Respuesta completa y definitiva" in page.locator("#chatbot").inner_text()
            assert "Respuesta parcial" not in handler.summarize_with_attention.call_args.args[0]
            assert "Respuesta completa y definitiva" in handler.summarize_with_attention.call_args.args[0]
            assert errors == []
            browser.close()
    finally:
        release.set()
        demo.close()


def test_nemo_offsets_and_colors_match_formatted_message_characters():
    """Compare DOM characters with an independent projection of sparse model rows."""
    history = [["María tiene **fiebre** 😊. Otra vez: fiebre y tos.",
                "Consulta [la guía](https://example.com) y `x < y`.\n\n"
                "| Síntoma | Estado |\n|---|---|\n| fiebre | sí |"]]
    prompt, spans = build_summary_prompt(history)
    summary = "María repite fiebre 😊, fiebre y tos."
    tokenizer = _load_attention_tokenizer("assets/tokenizers/nemo/tokenizer.json.gz")
    payload = {"model": "/models/nemo", "prompt": prompt, "summary": summary,
               "message_spans": [span.model_dump() for span in spans]}
    for prefix in ("prompt", "summary"):
        encoding = tokenizer.encode(payload[prefix], add_special_tokens=False)
        payload[prefix + "_offsets"] = list(encoding.offsets)
        payload[prefix + "_length"] = len(encoding.ids)
        payload[prefix + "_token_text"] = [
            tokenizer.decode([index], skip_special_tokens=False) for index in encoding.ids
        ]
    P, S = payload["prompt_length"], payload["summary_length"]
    eligible = [i for i, (a, b) in enumerate(payload["prompt_offsets"])
                if a < b and any(a < span.end and b > span.start for span in spans)]
    indices, values = [], []
    for j in range(S):
        indices.extend([[P + j, eligible[j % len(eligible)]],
                        [P + j, eligible[(j + 9) % len(eligible)]], [P + j, 1]])
        values.extend([0.45, 0.25, 0.10])
        if j:
            indices.append([P + j, P + j - 1])
            values.append(0.20)  # Summary columns must not color prompt tokens.
    payload["attention"] = {"indices": indices, "values": values}
    summary_html, rendered = render_attention(history, SummaryAttention.model_validate(payload))
    with gr.Blocks(js=production_js(), css=(ROOT / "assets/attention.css").read_text()) as demo:
        gr.Chatbot(value=rendered, sanitize_html=False, elem_id="chatbot")
        gr.HTML(summary_html)
        gr.Textbox(elem_id="input_textbox")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
        with playwright.sync_playwright() as runner:
            try:
                browser = runner.chromium.launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Playwright Chromium is not installed")
                raise
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_selector("#attention-summary .attention-token")
            assert page.locator("#chatbot table").count() == 1
            assert page.locator("#chatbot strong").inner_text() == "fiebre"
            assert page.locator("#chatbot a").inner_text() == "la guía"
            ranges = {(a, b) for a, b in payload["summary_offsets"] if a < b}
            ranges |= {(a + 1, b - 1) for a, b in payload["summary_offsets"] if b - a > 2}
            ranges |= {(m.start(), m.end()) for m in re.finditer(r"\w+", summary)}
            ranges |= {(a, payload["summary_offsets"][j + 1][1])
                       for j, (a, b) in enumerate(payload["summary_offsets"][:-1])}
            ranges.add((0, len(summary)))
            serialized = {}
            for start, end in sorted(ranges):
                # Exercise both DOM representations of a token boundary.
                for boundary in ("left", "right"):
                    page.evaluate("""async ({start, end, boundary}) => {
                        const root = document.querySelector('.attention-summary-text');
                        const utf16 = i => Array.from(root.textContent).slice(0, i).join('').length;
                        const locate = offset => {
                            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
                            let node, last;
                            while ((node = walker.nextNode())) {
                                last = node;
                                if (offset < node.length || (boundary === 'left' && offset === node.length))
                                    return [node, offset];
                                offset -= node.length;
                            }
                            return [last, last.length];
                        };
                        const range = document.createRange();
                        range.setStart(...locate(utf16(start)));
                        range.setEnd(...locate(utf16(end)));
                        window.getSelection().removeAllRanges();
                        window.getSelection().addRange(range);
                        document.dispatchEvent(new Event('selectionchange'));
                        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                    }""", {"start": start, "end": end, "boundary": boundary})
                    selected = {j for j, (a, b) in enumerate(payload["summary_offsets"])
                                if a < end and b > start and a < b}
                    weights = [0.0] * P
                    for (row, col), value in zip(indices, values):
                        if row - P in selected and col < P:
                            weights[col] += value
                    maximum = max(weights, default=0)
                    messages = page.locator("#chatbot .attention-message").evaluate_all("""nodes => nodes.map(root => {
                        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
                        const parts = [];
                        let node;
                        while ((node = walker.nextNode())) {
                            const span = node.parentElement.closest('.attention-token');
                            parts.push({text: node.textContent,
                                indices: span ? span.dataset.attnIndices.split(',').map(Number) : [],
                                color: span?.style.backgroundColor || ''});
                        }
                        return parts;
                    })""")
                    assert len(messages) == len(spans)
                    for message, span in zip(messages, spans):
                        assert "".join(part["text"] for part in message) == prompt[span.start:span.end]
                        position = span.start
                        for part in message:
                            for char in part["text"]:
                                expected_ids = [i for i, (a, b) in enumerate(payload["prompt_offsets"])
                                                if a <= position < b]
                                # Table structure whitespace cannot contain spans.
                                if not part["indices"] and char.isspace():
                                    position += 1
                                    continue
                                assert part["indices"] == expected_ids
                                weight = sum(weights[i] for i in expected_ids)
                                alpha = min(1.0, (weight / maximum) ** 0.35) if maximum else 0
                                color = ""
                                if alpha > 0.02:
                                    key = f"{alpha:.3f}"
                                    if key not in serialized:
                                        serialized[key] = page.evaluate("""alpha => {
                                            const span = document.createElement('span');
                                            span.style.backgroundColor = `rgba(255,190,40,${alpha})`;
                                            return span.style.backgroundColor;
                                        }""", key)
                                    color = serialized[key]
                                assert part["color"] == color, (start, end, position, char)
                                position += 1
            assert errors == []
            browser.close()
    finally:
        demo.close()
