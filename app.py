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

@spaces.GPU
def predict(lyrics, temperature, use_gemma):
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
