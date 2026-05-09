# Tokenization Research Project

This repository contains a research pipeline for comparing tokenization methods
under the same language-model training and evaluation setup.

The project focuses on:

- classical tokenizers such as BPE, byte-BPE, Unigram, WordPiece, character,
  word-level, and morpheme tokenization;
- neural tokenizers, including the final MIRON implementation and the neural
  segmenter line;
- comparable training, evaluation, and reporting artifacts for publication.

## Repository Layout

- `configs/` - Hydra experiment configurations
- `scripts/` - entrypoint scripts for training, preprocessing, evaluation, and report generation
- `src/` - library code used by the scripts
- `project_tokenizers/` - tokenizer implementations; generated trained assets stay local
- `tests/` - regression and contract tests
- `tex/` - thesis and report sources plus generated publication-facing report assets

Generated artifacts such as `outputs/`, tokenizer checkpoints, caches, and large
prepared datasets are intentionally kept out of source control.

## Main Workflow

1. Train a tokenizer.
2. Preprocess data into packed/shuffled training artifacts.
3. Train the language model.
4. Run evaluation.
5. Build report artifacts and LaTeX tables.

Typical commands:

```powershell
python scripts\01_train_tokenizer.py --config ru/bpe_32k
python scripts\02_preprocess_data.py --config ru/bpe_32k
python scripts\03_train_model.py --config ru/bpe_32k
python scripts\04_evaluate.py --experiment ru/bpe_32k --evaluators miron,lambada,canary
python scripts\05_build_report_artifacts.py --all --artifacts all
```

## Tokenizer Families

The repository compares the following families:

- `bpe_32k`
- `byte_bpe_32k`
- `byte_bpe_8k`
- `char`
- `morpheme`
- `unigram_32k`
- `wordpiece_32k`
- `miron`
- `neural_segmenter_32k`

The exact experiment list may differ by language, but the code paths are shared
and the evaluation layer is designed to keep comparisons consistent.

## MIRON

`src/miron_lib/` contains the final MIRON reference implementation used by the
experiments. The current architecture is flatten-only:

- character encoder
- flatten bridge
- core language model
- decoder projection
- character decoder

See `src/miron_lib/README.md` for the architecture details and shape contracts.
