---
title: Song Lyrics
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
models:
  - juliensimon/autonlp-song-lyrics-18753417
datasets:
  - juliensimon/autonlp-data-song-lyrics
---
# Song Lyrics Genre Classifier

Paste song lyrics and predict their musical genre using a fine-tuned transformer model. Returns the top 3 most likely genres with confidence scores.

<p align="center">
  <img src="https://img.shields.io/badge/Task-Text%20Classification-blue" alt="Text Classification">
  <img src="https://img.shields.io/badge/Training-AutoNLP-green" alt="AutoNLP">
  <img src="https://img.shields.io/badge/SDK-Gradio-orange" alt="Gradio">
</p>

## How It Works

1. Paste song lyrics into the text box
2. The model analyzes vocabulary, themes, and writing style
3. Get the top 3 predicted genres with confidence percentages

**Example output**: *"These lyrics are 72.34% Rock, 15.21% Pop and 8.45% Country."*

## Model

| Detail | Value |
|--------|-------|
| **Model** | [juliensimon/autonlp-song-lyrics-18753417](https://huggingface.co/juliensimon/autonlp-song-lyrics-18753417) |
| **Architecture** | AutoModelForSequenceClassification |
| **Training** | Hugging Face AutoNLP |
| **Dataset** | [juliensimon/autonlp-data-song-lyrics](https://huggingface.co/datasets/juliensimon/autonlp-data-song-lyrics) |
| **Task** | Multi-class text classification |

## Tech Stack

- **Transformers**: Sequence classification with softmax scoring
- **PyTorch**: Inference backend
- **Gradio**: Web interface

## Run Locally

```bash
git clone https://huggingface.co/spaces/juliensimon/song-lyrics
cd song-lyrics
pip install -r requirements.txt
python app.py
```
