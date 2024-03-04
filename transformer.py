from training_parameters import *

import torch
import torch.nn as nn
from torch.nn import functional as F
#whole transformer architecture

class Head(nn.Module):
    """ single head of self attention """
    def __init__(self):
        super().__init__()
        self.key_layer   = nn.Linear(in_features=EMBEDDING_DIM, out_features=HEAD_SIZE, bias=False)
        self.query_layer = nn.Linear(in_features=EMBEDDING_DIM, out_features=HEAD_SIZE, bias=False)
        self.value_layer = nn.Linear(in_features=EMBEDDING_DIM, out_features=HEAD_SIZE, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones((CONTEXT_LEN, CONTEXT_LEN))))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B,T,C = x.shape
        q = self.query_layer(x) #(B,T,C)
        k = self.key_layer(x) #(B,T,C)
        v = self.value_layer(x) #(B,T,C)

        #compute scores based on affinities
        weights = (q @ k.transpose(-2,-1)) * HEAD_SIZE**-0.5
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout(weights)


        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEADS)])
        self.projections = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.projections(out)

class Block(nn.Module):
    """Transformer Block: Communication folled by computation."""
    def __init__(self):
        super().__init__()

        self.multi_self_attention_heads_layer = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork(EMBEDDING_DIM)
        self.layer_norm1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIM)
    def forward(self, x):
        x = x + self.multi_self_attention_heads_layer(self.layer_norm1(x))
        x = x + self.feed_forward_network(self.layer_norm2(x))
        return x

class FeedForwardNetwork(nn.Module):
    """A simple linear network followed by a non-linearity"""
    def __init__(self, EMBEDDING_DIM):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=EMBEDDING_DIM, out_features=EMBEDDING_DIM*4),
            nn.ReLU(),
            nn.Linear(4*EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT)
        )
    def forward(self, x):
        return self.ffn(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(CONTEXT_LEN, EMBEDDING_DIM)

        self.blocks = nn.Sequential(*(Block() for _ in range(N_LAYER)))
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.language_model_head_linear_layer = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = token_embeddings + pos_embedding #the x does not contain only information about the value of the token, but also about its position which is in line with the assumptions of the transformer architecture
        x = self.blocks(x)
        logits = self.language_model_head_linear_layer(x)

        if targets is not None:
            B,T,C = logits.shape
            logits_reshaped = logits.view(B*T,C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss=None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:,-CONTEXT_LEN:]
            logits, loss = self(idx_crop)
            logits_last_timestep = logits[:,-1,:]
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(split, train_data, valid_data, batch_size, context_length):
    data = train_data if split == 'train' else valid_data

    idxs = torch.randint(low=0, high=len(data)-CONTEXT_LEN, size=(BATCH_SIZE,))
    x = torch.stack([data[idx:idx+CONTEXT_LEN] for idx in idxs])
    y = torch.stack([data[idx+1:idx+CONTEXT_LEN+1] for idx in idxs])
    x,y = x.to(DEVICE), y.to(DEVICE)
    return x,y

@torch.no_grad()
def estimate_loss(model, train_data, valid_data):
    out = {}
    #turn on evaluation mode
    model.eval()
    for split in ['train','valid']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x_batch, y_batch = get_batch(split, train_data, valid_data, BATCH_SIZE, CONTEXT_LEN)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #turn back to training mode
    model.train()
    return out