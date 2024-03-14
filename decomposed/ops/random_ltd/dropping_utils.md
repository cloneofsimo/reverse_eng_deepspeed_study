

### Summary



* `gpt_sample_tokens`: Samples tokens for GPT-style models, returning sampled indices and a new attention mask. Importance: **[High]**
* `bert_sample_tokens`: Samples tokens for BERT-style models, returning sampled indices and a new attention mask. Importance: **[High]**
* `GatherTokens`: A custom PyTorch autograd function for gathering activations based on sorted indices. Importance: **[High]**
* `ScatterTokens`: A custom PyTorch autograd function for scattering gradients based on sorted indices. Importance: **[High]**
* `RandomLTDBuilder`: A class for building the Random LTD (Layer-wise Token Dropping) module, but not defined in this file. Importance: **[Medium]** (mentioned but not defined)
* `random_ltd_module`: A global variable to store the RandomLTDBuilder instance. Importance: **[Low]**

This file is part of the DeepSpeed library and provides utilities for token sampling and manipulation, specifically designed for GPT and BERT-style models. The `gpt_sample_tokens` and `bert_sample_tokens` functions sample tokens for the respective models, applying layer-wise token dropping, and return the sorted indices along with updated attention masks. The custom autograd functions, `GatherTokens` and `ScatterTokens`, are used for efficient computation during the forward and backward passes, handling token gathering and scattering operations. The `RandomLTDBuilder` class, although not defined here, plays a role in building the module for these operations.

### Highlights



1. **Module and Function Definitions**: The code defines two main functions, `gpt_sample_tokens` and `bert_sample_tokens`, which are used to sample tokens for different models (GPT and BERT). Both functions share a similar structure, involving token sampling, sorting, and creating attention masks.
2. **RandomLTDBuilder**: The `RandomLTDBuilder` class is used to build and load an operation module for token manipulation. It is used to perform token sorting and gather/scatter operations, which are essential for the token sampling process.
3. **Global Variable**: The `random_ltd_module` is a global variable that holds the built module from `RandomLTDBuilder`. It is initialized as `None` and loaded when needed, to optimize the execution by avoiding repeated loading.
4. **GatherTokens and ScatterTokens Classes**: These are custom autograd functions for PyTorch, which implement token gathering and scattering operations. They are used to efficiently manipulate activation tensors based on the sorted token indices.
5. **Copyright and Licensing Information**: The code includes a copyright notice and a SPDX-License-Identifier, indicating that it is part of the Microsoft DeepSpeed project and is licensed under the Apache License 2.0.

### Pythonic Pseudocode

```python
# Import necessary modules
import random_module  # Represents the RandomLTDBuilder

# Constants and global variables
random_ltd_module = None

# Function: GPT-style token sampling
def gpt_sample_tokens(reserved_length, seq_length, batch_size, layers=1, device='cpu', attn_mask=None):
    # Create a uniform probability distribution
    prob_dist = uniform_distribution(layers * batch_size, seq_length, device)
    
    # Sample indices from the distribution
    sampled_indices = sample_indices(prob_dist, reserved_length)
    
    # Reshape and sort the sampled indices
    sampled_indices = sort_indices(sampled_indices, seq_length)
    
    # Create or update attention mask
    new_mask = update_mask(attn_mask, reserved_length)
    
    return sampled_indices, new_mask

# Function: BERT-style token sampling
def bert_sample_tokens(reserved_length, seq_length, batch_size, layers=1, device='cpu', attn_mask=None):
    assert attn_mask is not None  # attn_mask is mandatory for BERT
    
    # Perform the same steps as in GPT, with additional mask manipulation
    sampled_indices = gpt_sample_tokens(reserved_length, seq_length, batch_size, layers, device, attn_mask)[0]
    
    # Create a new mask for each layer and batch
    new_mask = create_layered_mask(attn_mask, sampled_indices, layers, batch_size)
    
    return sampled_indices, new_mask

# Custom autograd functions for tensor gathering and scattering
class GatherTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, sorted_indices, batch_first):
        # Initialize or load random_ltd_module
        ctx.save_for_backward(activations, sorted_indices)
        ctx.batch_first = batch_first
        return gather_tokens(activations, sorted_indices, batch_first)

    @staticmethod
    def backward(ctx, a_gradients, g_gradients):
        # Perform gradient scattering
        return scatter_tokens(a_gradients, g_gradients, ctx.saved_tensors, ctx.batch_first), None, None

class ScatterTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, all_activations, layer_activations, sorted_indices, batch_first):
        # Initialize or load random_ltd_module
        scatter_results = scatter_tokens(all_activations, layer_activations, sorted_indices, batch_first)
        ctx.save_for_backward(sorted_indices)
        ctx.batch_first = batch_first
        return scatter_results

    @staticmethod
    def backward(ctx, out_gradients):
        # Perform gradient gathering
        return out_gradients, gather_gradients(out_gradients, ctx.saved_tensors, ctx.batch_first), None, None
```


### import Relationships

Imports found:
import torch
from deepspeed.ops.op_builder import RandomLTDBuilder