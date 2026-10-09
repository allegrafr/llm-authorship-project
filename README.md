# ShareChat authorship project: Gemini, Claude, Grok

This package contains a reproducible first-exchange dataset and preparation code. 

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


