import spaces
import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient


LOCAL_MODEL = "google/gemma-4-E2B-it"
REMOTE_MODEL = "openai/gpt-oss-20b"

# Local model
pipe = pipeline(
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
    outputs = pipe(
        messages,
        max_new_tokens=200,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
    )
    return outputs[0]["generated_text"][-1]["content"]

def remote_generate(prompt, temperature, hf_token):
    client = InferenceClient(
        token=hf_token,
        model=REMOTE_MODEL,
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

def predict(lyrics, temperature, use_local, hf_token):
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

    if use_local:
        return local_generate(prompt, temperature)

    # User must log in for remote inference
    if hf_token is None or not getattr(hf_token, "token", None):
        return "Please log in with Hugging Face to use the remote model."

    return remote_generate(prompt, temperature, hf_token)


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=10, label="Song lyrics", placeholder="Paste song lyrics here..."),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
    outputs=gr.Text(label="Genre prediction"),
    title="Song Lyrics Genre Classifier",
    examples=[
        ["I walk this empty street on the boulevard of broken dreams\nWhere the city sleeps and I'm the only one and I walk alone\nMy shadow's the only one that walks beside me\nMy shallow heart's the only thing that's beating", 0.7, False],
        ["You are my fire, the one desire\nBelieve when I say, I want it that way\nBut we are two worlds apart\nCan't reach to your heart when you say\nThat I want it that way", 1.0, False],
        ["I got my mind on my money and my money on my mind\nRolling down the street smoking indo, sipping on gin and juice\nLaid back with my mind on my money and my money on my mind", 0.7, True],
        ["Achy breaky heart, don't tell my heart\nMy achy breaky heart, I just don't think he'd understand\nAnd if you tell my heart, my achy breaky heart\nHe might blow up and kill this man", 1.0, True],
    ],
    flagging_mode="never",
)

with gr.Blocks() as demo:
    gr.LoginButton()
    iface.render()

demo.launch()
