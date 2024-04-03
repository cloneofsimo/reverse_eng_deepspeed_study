
import math

import torch
import torch.nn as nn

def get_hf_model(model_name_or_path, dtype, tokenizer, ds_config=None):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.deepspeed import HfDeepSpeedConfig
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(model_config, '_attn_implementation') and ('gpt2' not in model_name_or_path.lower()):
        model_config._attn_implementation = 'sdpa'
        print(f'built-in flash attention will be used (config._attn_implementation: {model_config._attn_implementation})')
    else:
        print(f'there is no built-in flash attention for this model class ({model_name_or_path})')
        print(f'use bettertransformer or monkey patch :D')

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # code from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py
    dschf = HfDeepSpeedConfig(ds_config) if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=model_config,
        torch_dtype=dtype,
    )
    model_dtype = next(iter(model.parameters())).dtype
    assert dtype == model_dtype, print(f'model dtype is {model_dtype} but expected dtype is {dtype}')

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    resized_vocab_size = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(resized_vocab_size)  # make the vocab size multiple of 8

    return (model, resized_vocab_size)

def get_dummy_mlp_model(input_dim, hidden_dim, dtype):
    class GoodNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(GoodNet, self).__init__()
            self.emb = nn.Embedding(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, input_dim) # logit
        def forward(self, x):
            x = torch.relu(self.emb(x))
            x = torch.relu(self.fc2(x))
            return self.out(x)
    return GoodNet(input_dim, hidden_dim).to(dtype)