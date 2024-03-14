

### Summary



* `RandomLayerTokenDrop`: This is the main class that implements a layer wrapper for Random Layer Token Dropout (LTD) as described in the paper "random-ltd: <https://arxiv.org/abs/2211.11586>". It is designed to work with deep learning models like GPT and BERT. Importance: **[High]**
* `init_config`: Initializes the RandomLayerTokenDrop with configuration, scheduler, and layer ID. Importance: **[Medium]**
* `get_bsh`: Helper function to get the hidden tensor shape for batch-major inputs. Importance: **[Low]**
* `get_sbh`: Helper function to get the hidden tensor shape for sequence-major inputs. Importance: **[Low]**
* `forward`: The forward pass of the RandomLayerTokenDrop, which applies the dropout and handles the masking and gathering/scattering of tokens based on the scheduler's state. Importance: **[High]** 

This file is part of the DeepSpeed library and implements a module for applying Random Layer Token Dropout (LTD) to transformer-based models like GPT and BERT. The module wraps a given layer and dynamically drops tokens during training, following a specific schedule, to improve model efficiency and performance. The class `RandomLayerTokenDrop` manages the token sampling, masking, and the interaction with the scheduler to control the dropout process. The `forward` method is the core of the dropout mechanism, which is called during the forward pass of the model.

### Highlights



1. **Class Definition**: The code defines a class `RandomLayerTokenDrop` which is a subclass of `torch.nn.Module`. This class is designed to wrap a PyTorch layer and implement a specific functionality called "random LTD" (Layer Token Dropout), based on the paper "random-ltd: https://arxiv.org/abs/2211.11586".
2. **Initialization**: The `__init__` method initializes the class with attributes like the wrapped layer (`layer`), reserved length, scheduler, and other configuration-related variables. It also sets up the `batch_first` flag based on the hidden state order.
3. **Configuration Setup**: The `init_config` method sets up the class with the necessary configuration, scheduler, and layer ID. It handles different model types (encoder or decoder) and input dimension orders, and initializes the index generator accordingly.
4. **Helper Methods**: The `get_bsh` and `get_sbh` methods are helper functions to extract sequence and batch sizes from the hidden states tensor, depending on the input dimension order.
5. **Forward Pass**: The `forward` method is the core functionality of the class. It checks if the scheduler is present and if it's in training mode. If so, it performs the random LTD operation, which includes sampling indices, gathering and scattering hidden states, and potentially updating attention masks. If not in training mode, it simply passes the input through the wrapped layer.

### Pythonic Pseudocode

```python
# Import necessary modules and classes
from deepspeed.utils import logger
from torch.nn import Module
from deepspeed.ops.random_ltd.dropping_utils import gpt_sample_tokens, bert_sample_tokens, GatherTokens, ScatterTokens

# Define the RandomLayerTokenDrop class
class RandomLayerTokenDrop(Module):
    def __init__(self, layer: Module):
        # Initialize the class with the wrapped layer and default values
        self.layer = layer
        self.reserved_length = None
        self.random_ltd_scheduler = None
        self.max_length = None
        self.reserved_length = -1
        self.curr_seq = -1
        self.batch_first = False

    def init_config(self, config, scheduler, random_ltd_layer_id):
        # Set up the configuration and scheduler
        self.random_ltd_scheduler = scheduler
        self.random_ltd_layer_id = random_ltd_layer_id
        self.max_length = scheduler.state[RANDOM_LTD_MAX_VALUE]
        self.mask_name = config[RANDOM_LTD_MODEL_MASK_NAME]
        self.micro_bs = config[RANDOM_LTD_MICRO_BATCH_SIZE]
        self.random_ltd_num_layer = scheduler.random_ltd_layer_num
        self.model_type = config[RANDOM_LTD_MODEL_TYPE]
        self.set_hidden_state_order(config[RANDOM_LTD_HIDDEN_STATE_ORDER])

        # Set the index generator based on the model type
        self.set_index_generator(config[RANDOM_LTD_MODEL_TYPE])

    def set_hidden_state_order(self, hidden_state_order):
        # Determine the tensor shape based on the input dimension order
        if hidden_state_order == 'batch_seq_dim':
            self.get_hidden_tensor_shape = self.get_bsh
            self.batch_first = True
        elif hidden_state_order == 'seq_batch_dim':
            self.get_hidden_tensor_shape = self.get_sbh
            self.batch_first = False
        else:
            raise NotImplementedError

    def set_index_generator(self, model_type):
        # Set the token sampling function based on the model type
        if model_type == 'encoder':
            self.index_generator = bert_sample_tokens
        elif model_type == 'decoder':
            self.index_generator = gpt_sample_tokens
        else:
            raise NotImplementedError

    def get_bsh(self, hidden_stats):
        # Get the current sequence and micro-batch dimensions for batch_seq_dim order
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[1], hidden_stats.size()[0]

    def get_sbh(self, hidden_stats):
        # Get the current sequence and micro-batch dimensions for seq_batch_dim order
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[0], hidden_stats.size()[1]

    def forward(self, hidden_states, **kwargs) -> Tensor:
        # Perform the forward pass with random LTD
        if self.random_ltd_scheduler is not None:
            self.reserved_length = self.random_ltd_scheduler.get_current_seq()
            self.get_hidden_tensor_shape(hidden_states)

            if self.training and self.reserved_length < self.curr_seq:
                # Sample indices and attention mask, gather and scatter hidden states
                sampled_indices, part_attention_mask = self.sample_indices_and_mask(hidden_states, kwargs)
                outputs = self.apply_random_ltd_layer(part_hidden_states, **kwargs)
                hidden_states = self.scatter_output(hidden_states, outputs, sampled_indices)

            else:
                # Pass through the wrapped layer without LTD
                outputs = self.layer(hidden_states, **kwargs)
        else:
            # Pass through the wrapped layer without LTD
            outputs = self.layer(hidden_states, **kwargs)

        return outputs

    def sample_indices_and_mask(self, hidden_states, kwargs):
        # Sample indices and attention mask based on the current configuration
        mask = kwargs.get(self.mask_name)
        sampled_indices, part_attention_mask = self.index_generator(self.reserved_length, self.curr_seq, self.curr_micro_batch,
                                                                   self.random_ltd_num_layer, hidden_states.device, mask)
        self.random_ltd_scheduler.state[RANDOM_LTD_SAMPLE_INDEX] = sampled_indices
        self.random_ltd_scheduler.state[RANDOM_LTD_ATTENTION_MASK] = part_attention_mask
        return sampled_indices, part_attention_mask

    def apply_random_ltd_layer(self, part_hidden_states, **kwargs):
        # Apply the wrapped layer to the partially dropped hidden states
        outputs = self.random_ltd_layer(part_hidden_states, **kwargs)
        return outputs

    def scatter_output(self, hidden_states, outputs, sampled_indices):
        # Scatter the output back to the original shape
        if isinstance(outputs, tuple):
            hidden_states = ScatterTokens.apply(hidden_states, outputs[0], sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
            return self.reassemble_tuple(hidden_states, outputs)
        elif isinstance(outputs, Tensor):
            hidden_states = ScatterTokens.apply(hidden_states, outputs, sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
            return hidden_states
        else:
            raise NotImplementedError

    def reassemble_tuple(self, hidden_states, outputs):
        # Reassemble the tuple output
        my_list = list(outputs)
        my_list[0] = hidden_states
        return tuple(my_list)
```


### import Relationships

Imports found:
from deepspeed.utils import logger
from torch import Tensor
from torch.nn import Module
from ..constants import *
from deepspeed.ops.random_ltd.dropping_utils import gpt_sample_tokens, bert_sample_tokens, GatherTokens, ScatterTokens