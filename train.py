"""Run the six required authorship experiments on the prepared ShareChat CSV."""
import argparse
import copy
import hashlib
import json
import platform
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from models import CNN, LSTM


def tokenize(text):
    return re.findall(r'\n|\w+|[^\w\s]', text, flags=re.UNICODE)


def pieces(row, mode):
    prompt = tokenize(row.LLM_Input)[:128]
    response = tokenize(row.LLM_output)[:256]
    if mode == 'input':
        return prompt
    if mode == 'output':
        return response
    return prompt + ['<SEP>'] + response


def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    tokens = torch.zeros(len(batch), max(5, int(lengths.max())), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        tokens[i, :len(sequence)] = torch.tensor(sequence)
    return tokens, lengths, torch.tensor(labels)


def evaluate(model, loader, device):
    model.eval()
    truth, predictions = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            logits = model(tokens.to(device), lengths.to(device))
            loss_sum += torch.nn.functional.cross_entropy(logits, labels.to(device), reduction='sum').item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return truth, predictions, loss_sum / len(truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--results', default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--model', choices=['all', 'cnn', 'lstm'], default='all')
    parser.add_argument('--input-mode', choices=['all', 'input', 'output', 'both'], default='all')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    frame = pd.read_csv(args.data, keep_default_na=False)
    required = {'LLM_name', 'LLM_Input', 'LLM_output', 'conversation_id', 'prompt_group', 'split'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    labels = sorted(frame.LLM_name.unique())
    if labels != ['Claude', 'Gemini', 'Grok']:
        raise ValueError(f'Expected Claude, Gemini, Grok; found {labels}')
    if set(frame.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, and test splits')
    if frame.conversation_id.duplicated().any():
        raise ValueError('Duplicate conversation IDs')
    if (frame.groupby('prompt_group').split.nunique() > 1).any():
        raise ValueError('Prompt groups overlap across splits')
    if any(frame[c].str.strip().eq('').any() for c in ['LLM_Input', 'LLM_output']):
        raise ValueError('Empty text found')
    for split in ['train', 'validation', 'test']:
        if set(frame.loc[frame.split == split, 'LLM_name']) != set(labels):
            raise ValueError(f'Missing family in {split}')
    if args.smoke_test:
        frame = frame.groupby(['split', 'LLM_name'], group_keys=False).head(6).copy()
    root = Path(args.results or ('results_smoke' if args.smoke_test else 'results'))
    root.mkdir(parents=True, exist_ok=True)
    architectures = ['cnn', 'lstm'] if args.model == 'all' else [args.model]
    modes = ['input', 'output', 'both'] if args.input_mode == 'all' else [args.input_mode]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_ids = {name: i for i, name in enumerate(labels)}
    for architecture in architectures:
        for mode in modes:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(42)
            folder = root / f'{architecture}_{mode}'
            folder.mkdir(exist_ok=True)
            counts = Counter()
            for row in frame[frame.split == 'train'].itertuples():
                counts.update(pieces(row, mode))
            vocab = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2}
            for token, count in counts.most_common():
                if count >= 2 and token not in vocab and len(vocab) < 20000:
                    vocab[token] = len(vocab)
            datasets, loaders = {}, {}
            for split in ['train', 'validation', 'test']:
                datasets[split] = frame[frame.split == split].reset_index(drop=True)
                samples = [([vocab.get(t, 1) for t in pieces(row, mode)] or [1], label_ids[row.LLM_name])
                           for row in datasets[split].itertuples()]
                loaders[split] = DataLoader(samples, batch_size=32, shuffle=split == 'train', collate_fn=collate)
            model = (CNN if architecture == 'cnn' else LSTM)(len(vocab)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            history, best, stale, best_epoch, best_state = [], -1.0, 0, 0, None
            for epoch in range(1, (1 if args.smoke_test else args.epochs) + 1):
                model.train()
                total_loss = 0.0
                for tokens, lengths, targets in loaders['train']:
                    optimizer.zero_grad()
                    logits = model(tokens.to(device), lengths.to(device))
                    loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * len(targets)
                truth, pred, val_loss = evaluate(model, loaders['validation'], device)
                val_f1 = f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0)
                history.append({'epoch': epoch, 'train_loss': total_loss / len(datasets['train']),
                                'validation_loss': val_loss, 'validation_macro_f1': val_f1})
                print(f'{architecture}/{mode} epoch {epoch}: validation macro-F1={val_f1:.3f}', flush=True)
                if val_f1 > best:
                    best, stale, best_epoch = val_f1, 0, epoch
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= 3:
                    break
            model.load_state_dict(best_state)
            truth, pred, _ = evaluate(model, loaders['test'], device)
            metrics = {'architecture': architecture, 'input_mode': mode, 'smoke_test': args.smoke_test,
                       'test_accuracy': accuracy_score(truth, pred),
                       'test_macro_f1': f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0),
                       'best_epoch': best_epoch, 'test_examples': len(truth)}
            (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            report = classification_report(truth, pred, labels=[0, 1, 2], target_names=labels, output_dict=True, zero_division=0)
            (folder / 'classification_report.json').write_text(json.dumps(report, indent=2))
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            pd.DataFrame({'conversation_id': datasets['test'].conversation_id,
                          'true_label': [labels[i] for i in truth], 'predicted_label': [labels[i] for i in pred]})\
                .to_csv(folder / 'predictions.csv', index=False)
            matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
            pd.DataFrame(matrix, index=labels, columns=labels).to_csv(folder / 'confusion_matrix.csv')
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(matrix, cmap='Blues')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
                   xlabel='Predicted family', ylabel='Actual family', title=f'{architecture.upper()}: {mode}')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
            fig.tight_layout()
            fig.savefig(folder / 'confusion_matrix.png', dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Training')
            ax.plot([h['epoch'] for h in history], [h['validation_loss'] for h in history], label='Validation')
            ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=f'{architecture.upper()}: {mode}')
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / 'loss_curve.png', dpi=160)
            plt.close(fig)
            config = {'seed': 42, 'device': str(device), 'labels': labels, 'vocabulary_size': len(vocab),
                      'prompt_token_limit': 128, 'response_token_limit': 256, 'batch_size': 32,
                      'learning_rate': 0.001, 'patience': 3, 'requested_epochs': args.epochs,
                      'embedding_dim': 64, 'hidden_or_channels': 64, 'dropout': 0.35,
                      'split_counts': frame.split.value_counts().to_dict(),
                      'dataset_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                      'python': platform.python_version(), 'torch': torch.__version__,
                      'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, **metrics}
            (folder / 'config.json').write_text(json.dumps(config, indent=2))
            (folder / 'vocabulary.json').write_text(json.dumps(vocab, ensure_ascii=False))
            torch.save(best_state, folder / 'model.pt')
    summaries = [json.loads(p.read_text()) for p in sorted(root.glob('*/metrics.json'))]
    pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
    print(f'Finished. See {root / "summary.csv"}.', flush=True)
    if args.smoke_test:
        print('Smoke-test scores are only for checking execution. Do not report them as project results.')


if __name__ == '__main__':
    main()
