import torch
from titans_pytorch import MemoryAsContextTransformer

transformer = MemoryAsContextTransformer(
  num_tokens = 256,
  dim = 256,
  depth = 2,
  segment_len = 128,
  num_persist_mem_tokens = 4,
  num_longterm_mem_tokens = 16
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True)
loss.backward()

sampled = transformer.sample(token_ids[:, :4], 512)
print(sampled[0])