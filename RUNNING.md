Running Authorship 
In the terminal, from the repository root:

```bash
python -m pip install -r requirements.txt
python train.py --smoke-test
python train.py
```

Mac:
python3

virtual environment:
activate it first: `source .venv/bin/activate`.

The smoke test checks all six experiments on a tiny subset for one epoch. 

Its results are not scientific results. 

The full command runs CNN and LSTM using input only, output only, and input plus output. 

Allow the full run to finish; progress appears every epoch.

## Method

The target is the original platform family: Claude, Gemini, or Grok. Existing train/validation/test assignments are preserved. The vocabulary is fitted using training text only, separately for each input condition. Both architectures use the same preprocessing and seed. Case, punctuation, and newlines are retained as tokens; other whitespace is discarded. Prompt text is limited to the first 128 tokens and response text to the first 256 tokens. The combined condition retains both budgets, separated by a special token. These truncation limits must be documented in the report.

The CNN uses three convolution sizes (3, 4, 5) and masked max pooling. The LSTM uses packed sequences and its final hidden state. Both learn 64-dimensional embeddings from scratch. Validation macro-F1 selects the checkpoint; training stops after three epochs without improvement or 15 epochs. Test data is evaluated after checkpoint selection. This is a baseline configuration, not proof that either architecture is optimal.

## Outputs

results/summary.csv compares all six experiments. Each experiment folder contains accuracy and macro-F1 in metrics.json, per-family precision/recall/F1 in classification_report.json, confusion matrices, loss curves, epoch history, test predictions without chat text, the vocabulary, configuration and environment versions, and a model checkpoint.

Upload the source code, requirements, guide, and full-run results to GitHub. The .gitignore excludes local environments, smoke-test outputs, and binary model checkpoints. Keep the original preparation README/license/source documentation. If your assignment requires trained weights, distribute those separately as appropriate rather than assuming they belong in ordinary Git commits.

For RQ1, compare cnn_output and lstm_output. For RQ2, compare input, output, and both within each architecture. Balanced three-class random guessing has expected accuracy of about 33.3%; do not assume macro-F1 has the same random baseline. Input-only success may reflect topic or user-selection differences. Output signals may reflect platform formatting, dates, or versions. Do not interpret family prediction as a causal finding about the underlying model.

No cross-domain experiment is implemented here. Topic labels are noisy, so any optional domain analysis needs separate planning.

After the full run, share results/summary.csv to interpret the findings and begin the report. Do not tune settings based on test results.
