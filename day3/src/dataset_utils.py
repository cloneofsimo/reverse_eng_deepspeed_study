
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

def get_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_dataset():
    from datasets import load_dataset
    small_train_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train").shuffle(seed=42).select(range(10000))
    return small_train_dataset

# def get_train_loader(local_rank, train_dataset, train_micro_batch_size_per_gpu, tokenizer):
#     return DataLoader(
#         train_dataset, 
#         batch_size=train_micro_batch_size_per_gpu, 
#         sampler=RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset), 
#         drop_last=True,
#         collate_fn=Collator(tokenizer)
#     )

def get_collate_fn(tokenizer, max_length=1024):
    return Collator(tokenizer, max_length)

class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, data):
        data = [item['text'] for item in data]
        return self.tokenizer(
            data,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )