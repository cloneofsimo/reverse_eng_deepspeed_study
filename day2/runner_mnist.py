import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

class GoodNet(nn.Module):
    def __init__(self):
        super(GoodNet, self).__init__()
        self.fc1 = nn.Linear(784, 4096)
        
        self.A = nn.Linear(4096, 4, bias = False)
        self.B = nn.Linear(4, 4096, bias = False)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)



    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x) + self.B(self.A(x)))
        x = self.fc3(x)
        return x
    
import click
import time
import itertools
@click.command()
@click.option("--local_rank", default=-1, help="Local rank")
def main(local_rank):
    epochs = 5
    learning_rate = 0.001
    per_device_train_batch_size = 128
    train_batch_size = per_device_train_batch_size * 2

    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=per_device_train_batch_size, shuffle=True)

    offload = None

    ds_config = {
            "train_micro_batch_size_per_gpu": per_device_train_batch_size,
            "train_batch_size": train_batch_size,
            "zero_optimization": {
                "stage": 1,
            },
            "fp16": {"enabled": True},
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": True,
        }

    with deepspeed.zero.Init(enabled=False):      
        model = GoodNet()

  
    AdamOptimizer = torch.optim.Adam
    global_rank = torch.distributed.get_rank()


    optimizer = AdamOptimizer(
        itertools.chain([param for name, param in model.named_parameters() if not 'fc' in name]), lr=learning_rate, betas=(0.9, 0.95)
    )


    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        optimizer = optimizer
    )
    device = model_engine.device

    # Train the model
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).to(torch.float16), target.to(device)
            optimizer.zero_grad()

            output = model_engine(data)
            loss = nn.CrossEntropyLoss()(output.float(), target)
            model_engine.backward(loss)
            model_engine.step()

            if global_rank == 0:
                print(f"LOSS:", loss.item())
                
import inspect
import importlib
import types


if __name__ == "__main__":

    # import importlib
    # import inspect
    # import sys
    # import deepspeed  # Ensure DeepSpeed is installed and imported

    # def wrap_function_with_message(original_function, module_name):
    #     """Wrapper function that prints a message."""
    #     def wrapped_function(*args, **kwargs):
    #         print(f"Using function {original_function.__name__} from module {module_name}")
    #         return original_function(*args, **kwargs)
    #     return wrapped_function

    # def wrap_methods_in_class(cls, module_name):
    #     """Wrap all methods in a given class with a print message."""
    #     for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
    #         wrapped_method = wrap_function_with_message(method, module_name)
    #         setattr(cls, name, wrapped_method)

    # def traverse_and_wrap(module, parent_name='', visited=None):
    #     """Recursively traverse a module, wrapping its functions and methods."""
    #     if visited is None:
    #         visited = set()

    #     if module in visited:
    #         return
    #     visited.add(module)
    #     WHITELIST = [
    #         'runtime'
    #     ]
    #     def condition(_name:str):
    #         if _name.startswith('deepspeed') and any([(x in _name) for x in WHITELIST]):
    #             return True
    #         else:
    #             return False

    #     for name, obj in inspect.getmembers(module, lambda member: inspect.ismodule(member) or inspect.isfunction(member) or inspect.isclass(member)):
    #         if inspect.ismodule(obj) and obj.__name__.startswith('deepspeed'):
    #             # Traverse submodules
    #             traverse_and_wrap(obj, parent_name=obj.__name__, visited=visited)
    #         elif inspect.isfunction(obj) and condition(parent_name):
    #             # Wrap functions
    #             setattr(module, name, wrap_function_with_message(obj, parent_name))
    #         elif inspect.isclass(obj) and condition(parent_name):
    #             # Wrap methods within classes
    #             wrap_methods_in_class(obj, parent_name)

    # traverse_and_wrap(deepspeed, parent_name='deepspeed')



    main()