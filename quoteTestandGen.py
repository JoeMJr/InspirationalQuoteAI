import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import os
import json
from quoteGPT import QuoteGPT
import quoteGPT as GPTFunc

# Will turn this into functions for later
print("Quote Generation:")

model_path = "inspoquotes_model.pth"

if os.path.exists(model_path):
    #
    vocab_size = 82 #CHANGE THIS TO THE ACTUAL VOCAB SIZE
    block_size = 128
    batch_size = 8
    #
    model = QuoteGPT(vocab_size, block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    #
    vocab = json.load(open("inspoVocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    #print("itos:", itos, ", itos type:", type(itos), "\n49th?", itos["49"]) # fixed the keyerror finally

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[str(i)] for i in l])
    #
    print("Loading le model.")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print("Quotes (x10):")
    
    for i in range(10):
        print(GPTFunc.generate(model, stoi, decode, start="Quote: ", length=300, temperature=0.8, top_k=50))

    print("End of x10 quotes")

else:
    print("No model found, ending prompting.")
