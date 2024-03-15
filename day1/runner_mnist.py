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
        self.fc1 = nn.Linear(784, 512)
        
        self.A = nn.Linear(512, 4, bias = False)
        self.B = nn.Linear(4, 512, bias = False)
        
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)



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
    train_batch_size = per_device_train_batch_size * 2 * 1

    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=per_device_train_batch_size, shuffle=True)

    offload = 'cpu'

    ds_config = {
            "train_micro_batch_size_per_gpu": per_device_train_batch_size,
            "train_batch_size": train_batch_size,
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": offload},
                "offload_optimizer": {"device": offload},
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 3e7,
                "stage3_prefetch_bucket_size": 3e7,
                "memory_efficient_linear": False,
            },
            "fp16": {"enabled": True},
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": True,
        }

    with deepspeed.zero.Init():      
        model = GoodNet()

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
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


            for param in model.parameters():
                print(param.data.shape)
                output_state_dict = {}
                for k, v in model_engine.named_parameters():
                    
                    if hasattr(v, "ds_id"):
                        with deepspeed.zero.GatheredParameters(
                            _z3_params_to_fetch([v]), enabled=True
                        ):
                            v_p = v.data.cpu()
                    else:
                        v_p = v.cpu()

                    output_state_dict[k] = v_p

                for k, v in output_state_dict.items():
                
                    print(f"Rank {local_rank} has {k}, tensor {v.shape}")

            time.sleep(5)

            

            output = model_engine(data)
            loss = nn.CrossEntropyLoss()(output.float(), target)
            model_engine.backward(loss)
            model_engine.step()

            # if global_rank == 0:
            #     print(f"LOSS:", loss.item())



if __name__ == "__main__":
    main()