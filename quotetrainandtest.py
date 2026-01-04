import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import os
import json
import quoteGPT
import countQuotes

# Turn everything here into functions for easy use later

now = datetime.now()
print("Start Time =", now)

file_name = 'quotescsvClean.txt'
string_to_find = '<|endoftext|>'

model_path = "inspoquotes_model.pth"

lossVar = 0
vocabVar = 0

if os.path.exists(model_path):
    #
    print("Model Was already trained, Look in the files")
else:
    #
    print("Opening and reading the VC Quotes")
    myfile = open(file_name, "r", encoding="utf-8")
    text = myfile.read()
    myfile.close()
    print("Closing the VC Quotes")
    #
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print("Dataset has %d characters, vocabulary size %d" % (len(text), vocab_size))
    vocabVar = vocab_size

    print("stoi and itos calculation...")
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    #print("itos:", itos, ",\nitos type:", type(itos))
    print("Encoding and Decoding...")
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print("Setting up the sizes...")
    data = torch.tensor(encode(text), dtype=torch.long)
    block_size = 128
    batch_size = 8
    print("Making and saving the model")
    model = quoteGPT(vocab_size, block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    #
    with open("inspoVocab.json", "w") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)
    #
    # FINALLY FOUND A NICE WAY TO GET A GOOD AMOUNT OF TRAINING STEPS BASED ON THE DATA
    qoute_num = countQuotes.count_string_in_large_file(file_name, string_to_find)
    planned_epochs = 50
    the_num = (qoute_num / batch_size) * planned_epochs
    step_range = int(round(the_num, -3))
    #
    print("Training model...")
    # use this link to keep computer on https://www.youtube.com/watch?v=jfKfPfyJRdk
    for step in range(step_range):
        xb, yb = quoteGPT.get_batch(data, block_size, batch_size)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, loss {loss.item():.4f}")
    print(f"Step {step}, loss {loss.item():.4f}")
    lossVar = loss.item()
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

