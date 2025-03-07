import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from titans_pytorch import MemoryAsContextTransformer
from tqdm import tqdm

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"using device : {device}")

SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
  return tokenizer(example["text"])

def group_texts(examples):
  # input_ids 그룹화
  concatenated_ids = sum(examples["input_ids"], [])
  total_length = (len(concatenated_ids) // SEQ_LENGTH) * SEQ_LENGTH
  result_ids = [concatenated_ids[i : i + SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)]
  
  # attention_mask 그룹화 (존재하는 경우)
  if "attention_mask" in examples:
    concatenated_mask = sum(examples["attention_mask"], [])
    result_mask = [concatenated_mask[i : i + SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)]
    return {"input_ids": result_ids, "attention_mask": result_mask}
  else:
    return {"input_ids": result_ids}


def collate_fn(batch):
  input_ids = [torch.tensor(item["input_ids"]) for item in batch]
  return torch.stack(input_ids)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns = ["text"])

lm_dataset = tokenized_dataset.map(group_texts, batched = True)
lm_dataset = lm_dataset.shuffle(seed=42).select(range(1000))

dataloader = DataLoader(lm_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)

model = MemoryAsContextTransformer(
  num_tokens = tokenizer.vocab_size,
  dim = 256,
  depth = 2,
  segment_len = SEQ_LENGTH,
  num_persist_mem_tokens=4,
  num_longterm_mem_tokens=16,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

model.train()
for epoch in tqdm(range(NUM_EPOCHS)):
  total_loss = 0.0
  for batch in dataloader:
    batch = batch.to(device)
    optimizer.zero_grad()

    loss = model(batch, return_loss = True)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  avg_loss = total_loss / len(dataloader)
  print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss ; {avg_loss:.4f}")

model.eval()
with torch.no_grad():
  sample_prompt = batch[:,:8]
  sampled_ids = model.sample(sample_prompt, sample_length = 50, use_cache=True, temperature=0.7)
  generate_text = tokenizer.decode(sampled_ids.squeeze().tolist(), skip_special_tokens = True)
  print(f"generated text : {generate_text}")