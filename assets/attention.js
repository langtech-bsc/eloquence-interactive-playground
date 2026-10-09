function installAttentionVisualization() {
    // Keep COO sparse. Prompt identity rows and summary columns are not needed.
    function buildRows(data) {
        const rows = new Map();
        const P = data.prompt_length;
        data.attention.indices.forEach(([row, col], i) => {
            if (row < P || col >= P) return;
            const j = row - P;
            if (!rows.has(j)) rows.set(j, []);
            rows.get(j).push([col, data.attention.values[i]]);
        });
        return rows;
    }

    function selectionWeights(rows, selected, promptLength) {
        const weights = new Float64Array(promptLength);
        for (const j of selected) {
            for (const [col, weight] of rows.get(j) || []) weights[col] += weight;
        }
        return weights;
    }

    function opacity(weight, maximum) {
        return maximum > 0 ? Math.min(1, Math.pow(weight / maximum, 0.35)) : 0;
    }

    let active = null;
    let frame = null;
    let epoch = 0;
    let summaryEpoch = null;
    const blocked = new Set();
    const indicesOf = (span) => span.dataset.attnIndices.split(",").map(Number);

    function clearHighlight() {
        document.querySelectorAll("#chatbot .attention-token").forEach((span) => {
            if (span.style.backgroundColor) span.style.backgroundColor = "";
        });
    }

    function invalidate() {
        epoch += 1;
        const root = document.getElementById("attention-summary");
        if (root) {
            const data = JSON.parse(root.dataset.attention);
            blocked.add(data.version);
            root.classList.add("attention-stale");
        }
        active = null;
        clearHighlight();
    }

    function currentData() {
        const root = document.getElementById("attention-summary");
        if (!root) return null;
        if (active && active.root === root) return active;
        const data = JSON.parse(root.dataset.attention);
        if (blocked.has(data.version) || (summaryEpoch !== null && summaryEpoch !== epoch)) {
            root.classList.add("attention-stale");
            return null;
        }
        active = { root, data, rows: buildRows(data) };
        return active;
    }

    function updateHighlight() {
        const state = currentData();
        const selection = window.getSelection();
        if (!state || !selection || selection.isCollapsed || selection.rangeCount === 0) {
            clearHighlight();
            return;
        }
        const selected = new Set();
        state.root.querySelectorAll(".attention-summary-text .attention-token").forEach((span) => {
            if (selection.containsNode(span, true)) {
                for (const j of indicesOf(span)) selected.add(j);
            }
        });
        const weights = selectionWeights(state.rows, selected, state.data.prompt_length);
        // Normalize over every prompt token, including instructions not shown in bubbles.
        let maximum = 0;
        for (const value of weights) maximum = Math.max(maximum, value);
        document.querySelectorAll("#chatbot .attention-message").forEach((message) => {
            if (message.dataset.attnId !== state.data.version) return;
            message.querySelectorAll(".attention-token").forEach((span) => {
                const weight = indicesOf(span).reduce((sum, c) => sum + weights[c], 0);
                const alpha = opacity(weight, maximum);
                const color = alpha > 0.02 ? `rgba(255,190,40,${alpha.toFixed(3)})` : "";
                if (span.style.backgroundColor !== color) span.style.backgroundColor = color;
            });
        });
    }

    function schedule() {
        if (frame !== null) return;
        frame = requestAnimationFrame(() => {
            frame = null;
            updateHighlight();
        });
    }

    document.addEventListener("selectionchange", schedule);
    document.addEventListener("mouseup", schedule);
    document.addEventListener("keyup", (event) => {
        if (event.key === "Escape") {
            window.getSelection()?.removeAllRanges();
            clearHighlight();
        }
    });
    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) return;
        if (event.target.closest("#summarize_btn")) {
            invalidate();
            summaryEpoch = epoch;
        } else if (event.target.closest("#submit_btn, #clear_btn, #trigger_audio_submit, #load_history_confirm_btn")) {
            invalidate();
        } else if (event.target.closest("#chatbot")) {
            window.getSelection()?.removeAllRanges();
            clearHighlight();
        }
    }, true);
    document.addEventListener("change", (event) => {
        if (event.target instanceof Element && event.target.closest("#task_config, #llm_name")) invalidate();
    }, true);
    document.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && event.target instanceof Element &&
            event.target.closest("#input_textbox")) invalidate();
    }, true);
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
}
