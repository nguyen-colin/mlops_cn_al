import torch
import spaces
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_name = "juliensimon/autonlp-song-lyrics-18753417"
tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForSequenceClassification.from_pretrained(repo_name)
labels = model.config.id2label
print(labels)

gemma_repo = "google/gemma-4-E2B-it"
gemma_processor = AutoProcessor.from_pretrained(gemma_repo)
gemma_model = AutoModelForMultimodalLM.from_pretrained(
    gemma_repo,
    dtype="auto",
    device_map="auto"
)

@spaces.GPU
def predict(lyrics, temperature, use_gemma):
    if use_gemma:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a music genre classifier. "
                    "Analyze song lyrics and determine the three most likely "
                    "musical genres."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"""Analyze the following song lyrics:
        {lyrics}
        Return the top 3 most likely musical genres.
        Give a short explanation for your prediction."""
                        ),
                    },
                ]
        inputs = gemma_processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(gemma_model.device)
        # Keep track of where the generated response begins
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = gemma_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
            )
        # Only decode newly generated tokens
        response = gemma_processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return response
    else:
        inputs = tokenizer(lyrics, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        scaled_logits = outputs.logits / temperature
        predictions = torch.nn.functional.softmax(scaled_logits, dim=-1)
        predictions = predictions.detach().numpy()[0]
        predictions = predictions * 100
        print(predictions)
        sorted_indexes = np.argsort(predictions)
        return "These lyrics are {:.2f}% {}, {:.2f}% {} and {:.2f}% {}.".format(
            predictions[sorted_indexes[-1]],
            labels[sorted_indexes[-1]],
            predictions[sorted_indexes[-2]],
            labels[sorted_indexes[-2]],
            predictions[sorted_indexes[-3]],
            labels[sorted_indexes[-3]],
        )


description = (
    "Paste song lyrics and predict their musical genre. "
    "The model returns the top 3 most likely genres with confidence scores. "
    "Fine-tuned with [Hugging Face AutoNLP](https://huggingface.co/autotrain)."
)

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=10, label="Song lyrics", placeholder="Paste song lyrics here..."),
        gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Temperature",
                   info="Lower = more confident/peaked, higher = more uniform/softer predictions"),
        gr.Checkbox(label="Use Gemma", value=False),
    ],
    outputs=gr.Text(label="Genre prediction"),
    title="Song Lyrics Genre Classifier",
    description=description,
    examples=[
        ["I walk this empty street on the boulevard of broken dreams\nWhere the city sleeps and I'm the only one and I walk alone\nMy shadow's the only one that walks beside me\nMy shallow heart's the only thing that's beating", 1.0, False],
        ["You are my fire, the one desire\nBelieve when I say, I want it that way\nBut we are two worlds apart\nCan't reach to your heart when you say\nThat I want it that way", 1.0, False],
        ["Swing low, sweet chariot, coming for to carry me home\nSwing low, sweet chariot, coming for to carry me home\nI looked over Jordan, and what did I see\nComing for to carry me home", 1.0, False],
        ["I got my mind on my money and my money on my mind\nRolling down the street smoking indo, sipping on gin and juice\nLaid back with my mind on my money and my money on my mind", 1.0, False],
        ["Achy breaky heart, don't tell my heart\nMy achy breaky heart, I just don't think he'd understand\nAnd if you tell my heart, my achy breaky heart\nHe might blow up and kill this man", 1.0, False],
    ],
    flagging_mode="never",
)
iface.launch()
