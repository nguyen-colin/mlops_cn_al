---
title: Song Lyrics
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes: 
  - inference-api
---
# Song Lyrics Genre Classifier

Paste song lyrics to request the top three musical genres, estimated confidence
percentages, and explanations. These percentages are generated estimates, not
calibrated classification probabilities.

The app uses `openai/gpt-oss-20b` through Hugging Face remote inference and
`google/gemma-4-E2B-it` locally through a Transformers text-generation pipeline.
The interface is built with Gradio. Local inference requires a compatible CUDA
GPU and access to the model weights; remote inference requires Hugging Face login
and available inference access/credits.

## Automatic failover

By default, each request tries the remote model first. Selecting **Prefer local
model (automatic fallback enabled)** reverses the order. If the preferred model
raises an exception or returns an empty response, the alternative is attempted
automatically. Each model is attempted at most once per request.

Remote requests use a 30 second client timeout. Timeout, HTTP 429 rate-limit,
authentication/access, and server failures receive readable status messages.
Other inference errors also trigger fallback. Without login, remote inference is
skipped and local inference is attempted. If neither model succeeds, the app
returns an error explaining each failure without displaying raw exception details.


## Run

From the repository directory:

```bash
python -m pip install -r requirements.txt pytest
python app.py
```
