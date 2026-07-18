# SRL-Gated-Fusion-SEPF

Training code for **SRL-aware BERT models** that detect four communication signals from a single sentence: **S**entiment, **E**motion, **P**oliteness, and **F**ormality (SEPF). The models inject Semantic Role Labeling (SRL) information into BERT through role-attention and gated-fusion mechanisms, and are compared against vanilla BERT baselines.

> **This repository contains training and evaluation code only.**
> The trained model checkpoints and ready-to-run single-sentence inference plugins are hosted on Hugging Face: **https://huggingface.co/yeomtong**

## Repository structure

```
srl-aware-emotion-detection/
    Emotion_Model_Training.ipynb        # train & test emotion models (28 GoEmotions labels)
    dataset/
srl-aware-sentiment-detection/
    Sentiment_Model_Training.ipynb      # train & test sentiment models
    dataset/
srl-aware-politeness-detection/
    Create_Politeness_Datasets.ipynb    # build SRL-aligned politeness datasets
    Politeness_Model_Training.ipynb     # train & test politeness models
    dataset/
srl-aware-formality-detection/
    Create_Formality_Datasets.ipynb     # build SRL-aligned formality datasets
    Formality_Model_Training.ipynb      # train & test formality models
```

Each training notebook follows the same layout: data loading → SRL dataset & collate functions → model definitions → training loops for every variant → test evaluation.

## Model variants

Every task is trained in six configurations to isolate the effect of SRL information and the fusion strategy:

| Variant | Description |
|---|---|
| Vanilla BERT (no MLP) | BERT baseline with a linear classification head |
| Vanilla BERT (MLP) | BERT baseline with an MLP classification head |
| SRL, no fusion (no MLP / MLP) | SRL role-attention features, concatenation only |
| SRL, gated fusion (no MLP / MLP) | SRL role-attention features combined via a learned gated fusion |

(For politeness and formality, the SRL models are *directional* — e.g., ARG0 → ARG1 — and the no-fusion counterpart is a last-layer concatenation variant.)

## Usage

1. Open the training notebook for the task you want.
2. Replace every path marked `/Enter-your-path/` with the location of your data and model-log directories.
3. For politeness and formality, run the corresponding `Create_*_Datasets.ipynb` first to produce the SRL-aligned dataset files.
4. Run the notebook top to bottom. Each variant has its own training section; the final Test Evaluation section reports per-label precision/recall/F1.

The underlying SRL predictor (predicate-aware SRL BERT) is downloaded automatically from the Hugging Face Hub inside the notebooks.

### Requirements

```
torch
transformers
huggingface_hub
scikit-learn
scipy
pandas
numpy
tqdm
spacy            # politeness / formality only
```

For politeness and formality, also install the spaCy model: `python -m spacy download en_core_web_md`

## Trained models & single-sentence inference

If you just want predictions (with optional LLM-generated explanations) rather than training from scratch, use the plugins on Hugging Face — each includes the trained checkpoint and a self-contained script/notebook that takes a raw sentence and returns the prediction:

**https://huggingface.co/yeomtong**
