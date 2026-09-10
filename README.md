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

Remote requests use a 30-second client timeout. Timeout, HTTP 429 rate-limit,
authentication/access, and server failures receive readable status messages.
Other inference errors also trigger fallback. Without login, remote inference is
skipped and local inference is attempted. If neither model succeeds, the app
returns an error explaining each failure without displaying raw exception details.

Every successful response names the model that handled it and explains any
fallback. There is no persistent outage flag: the next request tries the preferred
model again, automatically returning to remote inference when it recovers in the
default mode. No background health checks are used.

Timeout configuration follows the [Hugging Face inference documentation](https://huggingface.co/docs/huggingface_hub/en/guides/inference#timeout).

## Run and test

From the repository directory:

```bash
python -m pip install -r requirements.txt pytest
python app.py
```

Run the unit tests separately:

```bash
python -m pytest tests/ -v
```

The GitHub Actions workflow in `.github/workflows/tests.yml` installs dependencies
and runs the tests on pushes and pull requests.

## Demonstrating failover and recovery

Run these deterministic demonstrations:

```bash
python -m pytest tests/test_app.py -v -k "remote_failure or remote_http_failure or remote_recovers or local_failure or both_models"
```

The tests simulate a remote timeout/rate limit and check that local inference
handles the request with an explicit fallback message. The recovery test makes
two requests: the first remote call times out and falls back locally; the second
remote call succeeds, and the app returns the remote result without another local
call. Reverse fallback and total failure are also tested. These tests mock model
inference; they demonstrate routing behavior, not real model quality or live GPU
availability. A successful Actions run provides evidence of this automated demo.

For a live UI demonstration, leave the local preference unchecked and submit lyrics
without logging in on a GPU-enabled Space with model access. The result should name
the local model and say remote inference was unavailable because login was required.
Then log in and submit again; if remote inference is available, the result will name
the remote model. Save screenshots of both outputs for the assignment. This live
demonstration must be performed on the deployed Space; it is not covered by mocks.

## Advantages and disadvantages

Automatic fallback improves availability during provider outages, rate limits,
and local GPU failures. It also lets users continue locally without remote login
when the local model is available.

Fallback can increase latency: a remote timeout may be followed by a local model
cold start. The models can produce different predictions, and local execution
requires GPU resources while remote execution may consume inference credits.
Retrying the preferred model on every request adds delay during prolonged outages.
Fallback is not guaranteed when both backends are unavailable. Unexpected inference
errors also trigger fallback, which can mask a backend defect until investigated.
