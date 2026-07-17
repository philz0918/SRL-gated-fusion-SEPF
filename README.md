# SRL-Aware SEPF Analysis with Gated Fusion

This repository contains code for **SRL-aware Sentiment, Emotion, Politeness, and Formality (SEPF) analysis**, where **Semantic Role Labeling (SRL) features are integrated into the model via a gated fusion mechanism**.

## Overview

The goal of this project is to improve language understanding by incorporating predicate–argument structure and semantic roles into downstream classification tasks. Rather than simply concatenating SRL features with text representations, this project uses **gated fusion**: a learned gate dynamically controls how much SRL-derived semantic information is blended with the contextual text encoding for each input, allowing the model to rely on semantic role structure when it is informative and fall back on the raw text representation when it is not.

This repository covers four tasks:

- **Sentiment detection**
- **Emotion detection**
- **Politeness detection**
- **Formality detection**

## Method: Gated Fusion of SRL Features

1. **Text encoding** — input text is encoded with a pretrained language model to obtain contextual representations.
2. **SRL feature extraction** — predicate–argument structures and semantic roles are extracted from the same input via Semantic Role Labeling.
3. **Gated fusion** — a gating layer computes element-wise weights that adaptively combine the text representation and the SRL representation:

   `fused = g ⊙ h_text + (1 − g) ⊙ h_srl`,  where `g = σ(W[h_text ; h_srl])`

4. **Task-specific classification** — the fused representation feeds into a classifier head for each of the SEPF tasks.

## Repository Contents

- `srl_aware_sentiment/` — SRL-aware sentiment detection with gated fusion
- `srl_aware_emotion/` — SRL-aware emotion detection with gated fusion
- `srl_aware_politeness/` — SRL-aware politeness detection with gated fusion
- `srl_aware_formality/` — SRL-aware formality detection with gated fusion
- `README.md` — project description

## Features

- Text preprocessing pipeline
- SRL-based feature extraction
- **Gated fusion module** for adaptively combining SRL and contextual text features
- Task-specific analysis for sentiment, emotion, politeness, and formality
- Example scripts for running the models
