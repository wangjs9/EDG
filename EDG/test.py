import torch
import torch.nn as nn
import numpy as np

pos_embedding = np.load('empathetic-dialogue/embedding_pos.txt', allow_pickle=True)
pos_embedding = torch.FloatTensor(pos_embedding)
embedding = nn.Embedding.from_pretrained(pos_embedding)
input = torch.LongTensor([-2])
input = embedding(input)
print(input)
