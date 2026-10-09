# ShareChat authorship project: Gemini, Claude, Grok

This package contains a reproducible first-exchange dataset and preparation code. No CNN/LSTM results have been generated yet.

## Source
User-provided ShareChat downloads from https://huggingface.co/datasets/tucnguyen/ShareChat
Authors: Yueru Yan, Tuc Nguyen, Bo Su, Melissa Lieffers, Thai Le. ShareChat: A Dataset of Chatbot Conversations in the Wild. https://arxiv.org/abs/2512.17843
Upstream data license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ . Retain attribution, indicate modifications, use noncommercially, and follow upstream platform-specific terms. Code is newly written for this project. Raw files are not included.

## Reproduce
Place the following original files in a folder called upload:
- claude_results_final_language_filtered.csv.zip
- grok_results_final_language_filtered.csv.zip
- gemini_results_final_language_filtered.csv
Run `python3 prepare_data.py --input-dir upload --output-dir prepared`.
Python standard library only. The ZIP readers stream the CSV without extracting it. Only first-exchange rows are retained in memory.

## Dataset construction
Group by source conversation URL and use message_index 1 (user) and 2 (llm). Require both messages to be English and nonempty. Exclude conflicting duplicate source rows, duplicate normalized prompt-response pairs within each family, and source-labelled image/video/media tasks. Normalization is NFKC, lowercase and whitespace collapse, used only for matching, not altering text. Malformed topic labels become unknown. Upstream topic labels are noisy and must be reviewed for domain experiments. Media filtering uses source topics and cannot catch every attachment-dependent prompt.

Sample up to 2000 pairs per family, limited by the smallest family. Seed 42. Hash normalized prompts into roughly 70/15/15 train/validation/test buckets so identical prompts stay together across families. Downsample within each bucket to obtain equal family counts. See data_audit.json for exact counts. Near duplicates and shared responses across otherwise different prompts have not been exhaustively checked.

LLM_name is the family/platform target, not an exact model-version label. Gemini and Grok version metadata are preserved where available. Claude's version field is retained as source_version_metadata but not assumed to be a model version. Generic metadata becomes unknown. Model versions may vary within each family.

## Features and experiments
Only LLM_Input and/or LLM_output are features. Never feed conversation IDs, prompt hashes, topic, model metadata, split or labels to the classifier.
Train CNN and LSTM separately for input only, output only, and prompt + response: six runs. Use the same split, a training-only vocabulary, and validation-based selection. For combined input reserve tokens for both parts. Report accuracy, macro F1, per-family scores and confusion matrices. Balanced chance baseline: 1/3.

RQ1: output-only CNN vs LSTM. RQ2: compare three input conditions for each architecture. Input-only performance may reflect topic, user behavior, time period or platform selection. It does not prove a causal effect of users choosing a model.
RQ3: create a separate held-out-domain experiment with all families represented. Do not treat the existing random test split as a domain-held-out split.
RQ4: compare response lengths and formatting, then retrain after masking explicit platform names or removing formatting. Keep raw and ablated results distinct.

## Review before publication
The source is redacted, but this preprocessing is not a comprehensive privacy or copyright audit. Review the selected text before public redistribution. Source-labelled asking_about_the_model tasks and direct model self-identification remain for a later identity-masking ablation. Do not present classification as purely a linguistic fingerprint: native platform wrappers, search text and version differences can be signals.

## Report outline
1. Problem and data curation, including the smaller Claude pool.
2. CNN/LSTM and training details.
3. Six-run results table and confusion matrices.
4. Error analysis, identity masking, domain experiments if completed.
5. Lessons, limitations and reproducibility.
