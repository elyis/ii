import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os


# map_location = torch.device('cpu')
#
#
# class Model(nn.Module):
#     def __init__(self, ninput_features):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(ninput_features, 1)
#
#     def forward(self, X):
#         ypred = torch.sigmoid(self.linear(X))
#         return ypred
#
#
# File = "pytorch_model.bin"
#
# loadmodel = Model(ninput_features=8)
# loadmodel.load_state_dict(torch.load(File, map_location=map_location), strict=False)
# loadmodel.eval()
#
# print(loadmodel.state_dict())


def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)


def generate(
        model, tok, text,
        do_sample=True, max_length=30, repetition_penalty=5.0,
        top_k=4, top_p=0.75, temperature=0.8,
        num_beams=None,
        no_repeat_ngram_size=3
):
    input_ids = tok.encode(text, return_tensors="pt")
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=1,
    )
    return list(map(tok.decode, out))

tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3medium_based_on_gpt2")

text = input()
generated = generate(model, tok, text, num_beams=1)
print(generated[0])
print()
print()
print()
text = generated[0].replace(text, "")
print(text)
print()
print()
print()

print("----------------------")
print(text)

text = text.split("")

for i in range(len(text)):
    if text[-1] not in ['!', '.', '?', ')', '(']:
        text.pop()
        break

index = text.find("\n")

if index != -1:
    text = text[index:-1]

print("###########")
print(text)
