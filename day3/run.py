import click
import time
import itertools
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import deepspeed
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from ds_config import get_train_ds_config
from src.models import get_dummy_mlp_model, get_hf_model
from src.dataset_utils import get_tokenizer, get_dataset, get_collate_fn
from src.utils import ContextManagers, get_torch_profiler

from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# HF_MODEL_PATH = 'gpt2-large'
HF_MODEL_PATH = 'mistralai/Mistral-7B-v0.1'


def _reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--stage", default=2, help="stage 1/2/3/")
@click.option("--precision", default="fp16", help="lower precision training")
@click.option("--offload", default=None, help="cpu or nvme offload")
@click.option("--zeropp", is_flag=True, help="zero++")
@click.option("--zeropp_hpz", default=8, help="zero++ hpz")
@click.option("--use_hf_model", is_flag=True, help="use mistral 7B for profiling")
@click.option("--use_torch_profiler", is_flag=True, help="use torch profiler")
def main(local_rank, stage, precision, offload, zeropp, zeropp_hpz, use_hf_model, use_torch_profiler):
    assert (stage in [1,2,3]) and (precision in ['fp16', 'bf16']) and (offload in [None, 'cpu', 'nvme'])

    ## deepspeed init for multi gpu setting
    if local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(local_rank)
        device = torch.device(get_accelerator().device_name(), local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    global_rank = torch.distributed.get_rank()

    ## get dataloader
    _reset_seeds()
    train_micro_batch_size_per_gpu, gradient_accumulation_steps, learning_rate, epochs = 4, 2, 0.001, 2
    tokenizer = get_tokenizer(HF_MODEL_PATH)
    train_dataset = get_dataset()
    collate_fn = get_collate_fn(tokenizer)

    ## get DS config
    ds_config = get_train_ds_config(
        precision=precision, 
        offload=offload, 
        stage=stage,

        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,

        zero_quantized_weights=True if zeropp else False,
        zero_hpz_partition_size=zeropp_hpz if zeropp else 1,
        zero_quantized_gradients=True if zeropp else False,   
    )

    ## get model
    dtype = DTYPE[precision]
    _reset_seeds()
    if use_hf_model:
        model, vocab_size = get_hf_model(HF_MODEL_PATH, dtype, tokenizer, ds_config)
    else:
        zero_init_enabled = True if stage==3 else False
        vocab_size = len(tokenizer)
        if zero_init_enabled:
            with deepspeed.zero.Init(config_dict_or_path=ds_config):      
                model = get_dummy_mlp_model(vocab_size, 128, dtype)
        else:
            model = get_dummy_mlp_model(vocab_size, 128, dtype)

    ## optimizer
    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
    optimizer = AdamOptimizer(
        itertools.chain([param for name, param in model.named_parameters() if not 'fc' in name]), 
        lr=learning_rate, 
        betas=(0.9, 0.95)
    )
               
    ## get DS engine (deepspeed.init)
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model = model,
        config = ds_config,
        optimizer = optimizer,
        training_data = train_dataset, # you can make your own dataloader but then, you should compute exact step, batch_size 
        collate_fn=collate_fn,
        # dist_init_required=True,
    ) # return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    device = model_engine.device
    model_engine.train()

    ## get torch profiler
    if use_torch_profiler:
        num_wait_steps=1
        num_warmup_steps=2
        num_active_steps=3
        num_repeat=1
        total_profiling_step = (num_wait_steps + num_warmup_steps + num_active_steps + num_repeat)
        context = [
            get_torch_profiler(
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                num_active_steps=num_active_steps,
                num_repeat=num_repeat,
            )
        ]
    else:
        context = []

    ## training (profiling) loop
    with ContextManagers(context) as p:
        for epoch in range(1, epochs + 1):
            for i, batch in enumerate(train_loader):
                ## allocate tensor to gpu 
                batch = {k: v.to(device) for k, v in batch.items()}

                ## release previous gradient and forward engine  
                optimizer.zero_grad()
                logit = model_engine(**batch)['logits'] if use_hf_model else model_engine(batch['input_ids'])
                logit = logit[:, :-1].contiguous().view(-1, vocab_size)
                target =  batch['input_ids'][:, 1:].contiguous().view(-1)

                ## compute loss
                loss = F.cross_entropy(logit.float(), target, ignore_index=tokenizer.pad_token_id)
                model_engine.backward(loss)
                model_engine.step()

                ## profiler step
                if use_torch_profiler:
                    p.step()
                    if (i+1) == total_profiling_step:
                        exit()

if __name__ == "__main__":
    main()