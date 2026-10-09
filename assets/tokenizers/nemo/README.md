Compressed tokenizer for `mistralai/Mistral-Nemo-Instruct-2407` (Apache-2.0).
Contains vocabulary and tokenization rules, no model weights.

The IP uses it to check that tokens match the attention service and map them
to the correct text positions for highlighting. It preserves the summary and
attention weights.

When changing the attention model, use its matching tokenizer and update
`tokenizer_path` in `models.json`.
