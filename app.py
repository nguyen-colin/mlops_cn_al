Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Buckets
new
Docs
Pricing


Hugging Face is way more fun with friends and colleagues! 🤗 Join an organization
Spaces:
juliensimon
/
song-lyrics


like
12
App
Files
Community
song-lyrics
/
app.py

juliensimon's picture
juliensimon
Add description, examples, and Gradio 5 compat
f59ff56
verified
5 months ago
Raw

Download with hf CLI

Copy download link
History
Blame
Contribute
Delete
2.56 kB
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_name = "juliensimon/autonlp-song-lyrics-18753417"

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForSequenceClassification.from_pretrained(repo_name)
labels = model.config.id2label
print(labels)


def predict(lyrics):
    inputs = tokenizer(lyrics, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
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
    inputs=gr.Textbox(lines=10, label="Song lyrics", placeholder="Paste song lyrics here..."),
    outputs=gr.Text(label="Genre prediction"),
    title="Song Lyrics Genre Classifier",
    description=description,
    examples=[
        ["I walk this empty street on the boulevard of broken dreams\nWhere the city sleeps and I'm the only one and I walk alone\nMy shadow's the only one that walks beside me\nMy shallow heart's the only thing that's beating"],
        ["You are my fire, the one desire\nBelieve when I say, I want it that way\nBut we are two worlds apart\nCan't reach to your heart when you say\nThat I want it that way"],
        ["Swing low, sweet chariot, coming for to carry me home\nSwing low, sweet chariot, coming for to carry me home\nI looked over Jordan, and what did I see\nComing for to carry me home"],
        ["I got my mind on my money and my money on my mind\nRolling down the street smoking indo, sipping on gin and juice\nLaid back with my mind on my money and my money on my mind"],
        ["Achy breaky heart, don't tell my heart\nMy achy breaky heart, I just don't think he'd understand\nAnd if you tell my heart, my achy breaky heart\nHe might blow up and kill this man"],
    ],
    flagging_mode="never",
)
iface.launch()

