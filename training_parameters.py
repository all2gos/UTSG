#hyperparameter board

import torch 

BATCH_SIZE = 2 #number of token chunks per batch
CONTEXT_LEN = 16 #length of the token chunks
LEARNING_RATE = 3e-4
MAX_ITERS = 200 #number of training iterations or steps
EVAL_INTERVAL = 300 #number of steps between evaluating the validation set to see how our validation loss is doing.
EVAL_ITERS = 200 #number of steps to do on the validation set per each interval. We do more than 1 to get a more accurate overall valid loss
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' #instead of using the cpu, we'll use the GPU if it's availble.
EMBEDDING_DIM = 384 #The vector size of the token embeddings
VOCAB_SIZE = 566
TORCH_SEED = 42
BLOCK_SIZE = 4
NUM_HEADS = 6
HEAD_SIZE = int(EMBEDDING_DIM/NUM_HEADS)

DROPOUT = 0.2
N_LAYER = 6