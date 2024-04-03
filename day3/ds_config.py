'''
og reference : https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/ds_utils.py
all configs  : https://github.com/microsoft/DeepSpeed/blob/master/docs/_pages/config-json.md
offical docs : https://deepspeed.readthedocs.io/en/latest/zero3.html#
'''


def get_train_ds_config(
    precision,
    offload,
    stage,

    train_micro_batch_size_per_gpu,
    gradient_accumulation_steps,

    allgather_partitions=True,
    allgather_bucket_size=5e8,
    reduce_scatter=True,
    reduce_bucket_size=5e8,

    ignore_unused_parameters=True,
    round_robin_gradients=False,

    stage3_param_persistence_threshold=1e4,
    stage3_max_live_parameters=3e7,
    stage3_prefetch_bucket_size=0,
    stage3_max_reuse_distance=1e9,
    memory_efficient_linear=False,

    contiguous_gradients=True,
    overlap_comm=False,

    zero_quantized_weights=False,
    zero_hpz_partition_size=1,
    zero_quantized_gradients=False,    

    wall_clock_breakdown=True,
    memory_breakdown=True,
):

    if precision == 'fp16':
        enable_fp16, enable_bf16 = True, False
    elif precision == 'bf16':
        enable_fp16, enable_bf16 = False, True
    else:
        raise ValueError(f"Invalid precision {precision}")

    device = "cpu" if offload else "none"

    zero_opt_dict = {
        ## stage 1, 2, 3
        "stage": stage,

        ## allgather and reduce hparams
        "allgather_partitions": allgather_partitions,
        "allgather_bucket_size": allgather_bucket_size,
        "reduce_scatter": reduce_scatter,
        "reduce_bucket_size": reduce_bucket_size,

        "contiguous_gradients": contiguous_gradients,
        "overlap_comm": overlap_comm,

        ## cpu or NVMe offloading
        "offload_param": {
            "device": device, # "[cpu|nvme]",
            "nvme_path": "/local_nvme",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9,
        },
        "offload_optimizer": {
            "device": device, # "[cpu|nvme]",
            "nvme_path": "/local_nvme",
            "pin_memory": True,
            "ratio": 0.3,
            "buffer_count": 4,
            "fast_init": False,
        },

        "ignore_unused_parameters": ignore_unused_parameters,
        "round_robin_gradients": round_robin_gradients,

        ## zero-3
        "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
        "stage3_max_live_parameters": stage3_max_live_parameters,
        "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
        "stage3_max_reuse_distance": stage3_max_reuse_distance,
        "memory_efficient_linear": memory_efficient_linear,

        ## zeropp quantization 
        "zero_quantized_weights": zero_quantized_weights,
        "zero_hpz_partition_size": zero_hpz_partition_size,
        "zero_quantized_gradients": zero_quantized_weights,
    }
    output =  {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 10,

        "zero_optimization": zero_opt_dict,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,

        ## dtype
        "fp16": {
            "enabled": enable_fp16,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": enable_bf16,
        },

        ## gradient clipping and scaling
        "gradient_clipping": 1.0,
        "prescale_gradients": False,

        ## for profiling
        "wall_clock_breakdown": wall_clock_breakdown,
        "memory_breakdown": memory_breakdown,

        "data_types": {
            "grad_accum_dtype": None,
        },

        "comms_logger": {
        "enabled": True,
        "verbose": False,
        "prof_all": False,
        "debug": False,
        "prof_ops": ["all_reduce", "all_gather"],
        },
    }
    return output