import time
import spaces
import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
from functools import lru_cache

LOCAL_MODEL = "google/gemma-4-E2B-it"
REMOTE_MODEL = "openai/gpt-oss-20b"
REMOTE_TIMEOUT_SECONDS = 30

# Local model
@lru_cache(maxsize=1)
def get_pipe():
    return pipeline(
        "text-generation",
        model=LOCAL_MODEL,
        dtype="auto",
        device="cuda",
    )

@spaces.GPU
def local_generate(prompt, temperature):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    outputs = get_pipe()(
        messages,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
    )

    return outputs[0]["generated_text"][-1]["content"]

def remote_generate(prompt, temperature, hf_token: gr.OAuthToken | None):
    
    client = InferenceClient(
        token=hf_token.token,
        model=REMOTE_MODEL,
        timeout=REMOTE_TIMEOUT_SECONDS,
    )
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat_completion(
        messages,
        max_tokens=200,
        temperature=temperature,
        top_p=0.95,
    )
    
    return response.choices[0].message.content

def failure_reason(error):
    """Describe failures without exposing API responses or credentials."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "request timed out"
    if status == 429:
        return "rate limit reached"
    if status in (401, 403):
        return "authentication or access denied"
    if status is not None and status >= 500:
        return "service unavailable"
    return "inference failed"


def predict(lyrics, temperature, use_local, hf_token: gr.OAuthToken | None):
    if not lyrics or not lyrics.strip():
        return "Please enter song lyrics."

    prompt = f"""
Analyze the following song lyrics.

Predict the top 3 most likely musical genres.

For each genre:
- Give the genre
- Give an estimated confidence percentage
- Give a short explanation

Lyrics:
{lyrics}
"""

    # Each request tries the preferred model first, allowing automatic recovery.
    order = ("local", "remote") if use_local else ("remote", "local")
    failures = []
    for backend in order:
        if backend == "remote" and not getattr(hf_token, "token", None):
            failures.append("Remote model unavailable: Hugging Face login required")
            continue

        try:
            start = time.perf_counter()
            
            if backend == "local":
                result = local_generate(prompt, temperature)
                model = LOCAL_MODEL
            else:
                result = remote_generate(prompt, temperature, hf_token)
                model = REMOTE_MODEL

            elapsed = time.perf_counter() - start
            
            if not isinstance(result, str) or not result.strip():
                raise ValueError("Empty model response")
        except Exception as error:
            # catch only around inference, and attempt each backend at most once.
            failures.append(f"{backend.capitalize()} model: {failure_reason(error)}")
            continue

        status = (
            f"Handled by {backend} model: {model}\n"
            f"Response time: {elapsed:.2f} seconds"
        )
        if failures:
            status += "\nAutomatic fallback: " + "; ".join(failures) + "."
        return f"{status}\n\n{result}"

    return "Unable to generate a prediction. " + "; ".join(failures) + ". Please try again."


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(lines=10, label="Song lyrics", placeholder="Paste song lyrics here..."),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
            gr.Checkbox(label="Prefer local model (automatic fallback enabled)", value=False),
        ],
        outputs=gr.Text(label="Genre prediction"),
        title="Song Lyrics Genre Classifier",
        examples=[
            ["I walk this empty street on the boulevard of broken dreams\nWhere the city sleeps and I'm the only one and I walk alone\nMy shadow's the only one that walks beside me\nMy shallow heart's the only thing that's beating", 0.7, False],
            ["You are my fire, the one desire\nBelieve when I say, I want it that way\nBut we are two worlds apart\nCan't reach to your heart when you say\nThat I want it that way", 1.0, False],
            ["I got my mind on my money and my money on my mind\nRolling down the street smoking indo, sipping on gin and juice\nLaid back with my mind on my money and my money on my mind", 0.7, True],
            ["Achy breaky heart, don't tell my heart\nMy achy breaky heart, I just don't think he'd understand\nAnd if you tell my heart, my achy breaky heart\nHe might blow up and kill this man", 1.0, True],
        ],
        cache_examples=False,
        flagging_mode="never",
    )

    with gr.Blocks() as demo:
        gr.LoginButton()
        iface.render()

    demo.launch()
