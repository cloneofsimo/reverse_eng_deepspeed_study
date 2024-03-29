# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import stat
import torch
import hashlib
from collections import defaultdict, OrderedDict, deque
from shutil import copyfile
import gc

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from typing import Callable, Dict, Union, Iterable

import deepspeed

from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage, DummyOptim
from .zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer, ZeRORuntimeException
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.config import ZERO_OPTIMIZATION

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT, ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, \
    MUSGD_OPTIMIZER, LION_OPTIMIZER

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    PLD_THETA, PLD_GAMMA, BFLOAT16, FP16, AMP, GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK
from deepspeed.runtime.zero.config import ZeroStageEnum

# from deepspeed.compression import compression_scheduler
# from deepspeed.compression.constants import \
#     WEIGHT_QUANTIZE_IN_FORWARD_ENABLED, \
#     WEIGHT_QUANTIZATION, SHARED_PARAMETERS, \
#     WEIGHT_QUANTIZE_ENABLED, \
#     WEIGHT_QUANTIZE_GROUPS, \
#     WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE, \
#     WEIGHT_QUANTIZE_CHANGE_RATIO, \
#     WEIGHT_QUANTIZE_TYPE, \
#     WEIGHT_QUANTIZE_ROUNDING, \
#     WEIGHT_QUANTIZE_VERBOSE, \
#     WEIGHT_QUANTIZE_KERNEL

from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, FROZEN_PARAM_FRAGMENTS
from deepspeed.runtime.sparse_tensor import SparseTensor

from deepspeed.runtime import lr_schedules
from deepspeed.utils import groups
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.utils.timer import NoopTimer, ThroughputTimer, SynchronizedWallClockTimer, \
    FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, \
    STEP_MICRO_TIMER, \
    FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_GLOBAL_TIMER
from deepspeed.utils.debug import debug_extract_module_and_param_names, debug_clear_module_and_param_names
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop
from deepspeed.runtime.utils import clip_grad_norm_
# from deepspeed.runtime.eigenvalue import Eigenvalue
from deepspeed.runtime.data_pipeline.constants import DATA_SAMPLING, \
    DATA_ROUTING, DATA_SAMPLING_ENABLED, CURRICULUM_LEARNING, \
    CURRICULUM_LEARNING_ENABLED, DATA_SAMPLING_NUM_WORKERS, RANDOM_LTD, \
    RANDOM_LTD_ENABLED, RANDOM_LTD_LAYER_ID, RANDOM_LTD_LAYER_NUM, \
    RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE, RANDOM_LTD_LAYER_TOKEN_LR_ENABLED, \
    RANDOM_LTD_GLOBAL_BATCH_SIZE, RANDOM_LTD_MICRO_BATCH_SIZE, DATA_EFFICIENCY
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
from deepspeed.runtime.data_pipeline.data_routing.scheduler import RandomLTDScheduler
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict
from deepspeed.runtime.data_pipeline.data_routing.basic_layer import RandomLayerTokenDrop

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from .pipe.module import PipelineModule
from .utils import get_ma_status
from .compiler import CompiledModuleWrapper
from ..ops.adam import FusedAdam

# from ..moe.sharded_moe import TopKGate, MOELayer
# from ..moe.layer import MoE
# from ..moe.utils import is_moe_param

from ..git_version_info import version

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.utils.logging import print_json_dist, print_configuration

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.config import DtypeEnum

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

DeepSpeedOptimizerCallable = \
    Callable[[Union[Iterable[Parameter], Dict[str, Iterable]]], Optimizer]
DeepSpeedSchedulerCallable = Callable[[Optimizer], _LRScheduler]

try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False


def split_half_float_double_sparse(tensors):
    device_type = get_accelerator().device_name()
    supported_types = get_accelerator().supported_dtypes()

    for t in tensors:
        assert t.dtype in supported_types, f"attempting to reduce an unsupported grad type: {t.dtype}"

    sparse_tensor_buckets, dense_tensor_buckets = [], []
    for i, dtype in enumerate(supported_types):
        sparse_bucket, dense_bucket = [], []
        for t in tensors:
            if t.dtype == dtype:
                if isinstance(t, SparseTensor):
                    sparse_bucket.append(t)
                else:
                    dense_bucket.append(t)
        if sparse_bucket:
            sparse_tensor_buckets.append((dtype, sparse_bucket))
        if dense_bucket:
            dense_tensor_buckets.append((dtype, dense_bucket))
    return sparse_tensor_buckets, dense_tensor_buckets


class EngineTimers(object):
    r"""Wallclock timers for DeepSpeedEngine"""

    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []

        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [
                FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER
            ]


from multiprocessing_pdb import MultiprocessingPdb 
Tra = MultiprocessingPdb().set_trace

#############################################################################################################
##########################################      Engine Initialize      ######################################
#############################################################################################################

class DeepSpeedEngine(Module):
    r"""DeepSpeed engine for training."""

    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_class=None,
                 dont_change_device=False):
        super(DeepSpeedEngine, self).__init__()

        ############################################################################################################
        ####################################     essential configurations     ######################################
        ############################################################################################################

        # training data and etc
        self.dont_change_device = dont_change_device
        self.training_data = training_data
        self.collate_fn = collate_fn

        # optimizer, lr scheduler and stats
        self.client_optimizer = optimizer
        self.client_lr_scheduler = lr_scheduler
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True

        # model parallel # https://github.com/microsoft/DeepSpeed/blob/master/docs/_pages/training.md#model-parallelism
        self.mpu = mpu
        self.data_parallel_group = None
        self.all_to_all_group = None

        self.config = config
        self._config = config_class
        self.dist_backend = get_accelerator().communication_backend_name()

        self.enable_backward_allreduce = True

        # GAS -> Gradient Accumulation step
        self.gas_boundary_ctr = 0
        self._step_applied = False
        self._global_grad_norm = None
        self._is_gradient_accumulation_boundary = None
        self.scale_wrt_gas = None
        self.losses = None
        self.use_ds_comm = False  # False --> Use torch.dist, True --> Use ds.comm backend.

        # removed or something (pld, ltd 등은 다 삭제해버렸음)
        self.has_moe_layers = False

        # # for debug purposes - can then debug print: debug_get_module_name(module)
        # debug_extract_module_and_param_names(model)


        ############################################################################################################
        ##########################################      init engine      ###########################################
        ############################################################################################################
        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        see_memory_usage(f"DeepSpeed Engine: After args sanity test", force=self.memory_breakdown())
        if mpu is not None:
            if self.elasticity_enabled():
                if not self.is_elastic_model_parallel_supported():
                    assert not self.elasticity_enabled(), ("Elasticity is not currently supported"
                                                           " with model parallelism.")

        self._set_distributed_vars(args)
        dist.configure(self._config)
        see_memory_usage(f"DeepSpeed Engine: Before configure distributed model",force=self.memory_breakdown())
        # Tra()

        self.pipeline_parallelism = isinstance(model, PipelineModule)
        self._configure_distributed_model(model) # Configure distributed model
        self.param_names = {param: name for name, param in model.named_parameters()} # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        see_memory_usage(f"DeepSpeed Engine: After configure distributed model", force=self.memory_breakdown())

        # Tra()
        '''
        GoodNet(
        (fc1): Linear(in_features=784, out_features=4096, bias=True)
        (fc2): Linear(in_features=4096, out_features=4096, bias=True)
        (out): Linear(in_features=4096, out_features=10, bias=True)
        )
        False

        (Pdb) for k, v in self.param_names.items(): print(k.size(), v);
        torch.Size([4096, 784]) fc1.weight
        torch.Size([4096]) fc1.bias
        torch.Size([4096, 4096]) fc2.weight
        torch.Size([4096]) fc2.bias
        torch.Size([10, 4096]) out.weight
        torch.Size([10]) out.bias
        '''


        ###############################################################################################
        ########################     Configure optimizer and scheduler  ###############################
        ###############################################################################################
        # Configure optimizer and scheduler
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None
        has_optimizer = False

        if optimizer or self.optimizer_name():
            has_optimizer = True
        if model_parameters is None:  # If no parameters given by init default to module parameters
            model_parameters = self.module.parameters()
        if not isinstance(model_parameters, list): # Convert model parameters from generator to list
            model_parameters = list(model_parameters)
        '''
        (Pdb) has_optimizer; self.optimizer_name(); model_parameters;
        True
        'DeepSpeedCPUAdam'
        [Parameter containing:
        tensor([[-0.0171, -0.0322, -0.0339,  ...,  0.0140,  0.0090,  0.0034],
        '''

        ## wallclock and throughput timer
        self.monitor = MonitorMaster(self._config.monitor_config)
        self.timers = SynchronizedWallClockTimer() # Configure wall clock timers
        self.tput_timer = ThroughputTimer( # Throughput timer
            batch_size=self.train_batch_size(),
            steps_per_output=self.steps_per_print(),
            monitor_memory=False,
        )

        if has_optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler(lr_scheduler)
            self._report_progress(0)
        elif self.zero_optimization(): # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None)
        else:
            raise NotImplementedError("fp16 and bf16 optimizer are removed!")
        '''
        (Pdb) self.optimizer
        <deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer object at 0x7f9f886dcd60>
        '''
        # if hasattr(model, 'pruners'): # Hook optimizer for snip_momentum pruning
        #     raise NotImplementedError("removed!")
        
        ###############################################################################################
        ########################     sparse gradient and eigenvalue class  #####################
        ###############################################################################################
        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        for name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info("Will convert {} to sparse tensor during training".format(name))


        ###############################################################################################
        ##########################################      ETC      ######################################
        ###############################################################################################
        ## profiler and Engine timers
        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())
        self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
                                          enable_global_timers=self.wall_clock_breakdown()
                                          or self.flops_profiler_enabled())

        ## dataloader
        self.training_dataloader = self.deepspeed_io(training_data) if training_data else None

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors
        '''
        (Pdb) _flatten_dense_tensors((torch.rand(4,10),torch.rand(4,10))).size()
        torch.Size([80])
        '''

        if self._config.compile_config.enabled:
            self._set_client_model(CompiledModuleWrapper(self.module, self._config.compile_config))


    #############################################################################################################
    ##########################################     Configure optimizer     ######################################
    #############################################################################################################

    def _configure_optimizer(self, client_optimizer, model_parameters):

        ## AdamW optimizer or CPU Adam, and so on  
        if client_optimizer is None:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(f"Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer", ranks=[0])
        else:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])

            if self.zero_use_cpu_optimizer() and not isinstance(basic_optimizer, deepspeed.ops.adam.DeepSpeedCPUAdam):
                if self.zero_force_ds_cpu_optimizer():
                    msg = f'You are using ZeRO-Offload with a client provided optimizer ({type(basic_optimizer)}) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.'
                    raise ZeRORuntimeException(msg)

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]
        log_dist("Removing param_group that has no 'params' in the basic Optimizer", ranks=[0])
        self._check_for_duplicates(basic_optimizer)
        self.basic_optimizer = basic_optimizer
        log_dist("DeepSpeed Basic Optimizer = {}".format(basic_optimizer.__class__.__name__), ranks=[0])
        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)


        ## Initializing ZeRO optimizer
        if optimizer_wrapper == ZERO_OPTIMIZATION:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        else:
            raise NotImplementedError("all removed")
        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer_name()), ranks=[0])

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if optimizer_parameters is None:
            optimizer_parameters = {}
        # print(optimizer_parameters.keys())
        if "max_grad_norm" in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )

        if self.optimizer_name() in [ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name() == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters, **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters, **optimizer_parameters)
            else:
                ## CPU Adam Optimizer or Fused Adam
                if self.zero_use_cpu_optimizer():
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    optimizer = DeepSpeedCPUAdam(model_parameters,
                                                 **optimizer_parameters,
                                                 adamw_mode=effective_adam_w_mode)
                else:
                    from deepspeed.ops.adam import FusedAdam
                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() == ADAGRAD_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                optimizer = DeepSpeedCPUAdagrad(model_parameters, **optimizer_parameters)
            else:
                optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)

        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb
            optimizer = FusedLamb(model_parameters, **optimizer_parameters)

        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam
            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Adam is only verified under FP16")

        elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
            optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f'Currently the convergence of 0/1 Adam is only verified under FP16')

        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb
            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Lamb is only verified under FP16")

        elif self.optimizer_name() == LION_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.lion import DeepSpeedCPULion
                optimizer = DeepSpeedCPULion(model_parameters, **optimizer_parameters)
            else:
                from deepspeed.ops.lion import FusedLion
                optimizer = FusedLion(model_parameters, **optimizer_parameters)

        elif self.optimizer_name() == MUADAM_OPTIMIZER:
            try:
                from mup import MuAdam
            except ImportError:
                logger.error(f"Install mup to use MuAdam optimizer")
            optimizer = MuAdam(model_parameters, **optimizer_parameters)

        elif self.optimizer_name() == MUADAMW_OPTIMIZER:
            try:
                from mup import MuAdamW
            except ImportError:
                logger.error(f"Install mup to use MuAdamW optimizer")
            optimizer = MuAdamW(model_parameters, **optimizer_parameters)

        elif self.optimizer_name() == MUSGD_OPTIMIZER:
            try:
                from mup import MuSGD
            except ImportError:
                logger.error(f"Install mup to use MuSGD optimizer")
            optimizer = MuSGD(model_parameters, **optimizer_parameters)

        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()

        mics_shard_size = self.mics_shard_size()
        model_dtype, gradient_accumulation_dtype = self.get_data_types()

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))
            '''
            class DummyOptim():
                def __init__(self, params):
                    self.param_groups = []
                    self.param_groups.append({'params': params})
            '''

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        ## zero 1, 2
        if zero_stage <= ZeroStageEnum.gradients:
            '''
            class ZeroStageEnum(int, Enum):
                """ Enum class for possible zero stages """
                disabled = 0
                optimizer_states = 1
                gradients = 2
                weights = 3
                max_stage = 3
            '''

            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage {} requires an optimizer".format(zero_stage)

            log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])
            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning("Pipeline parallelism does not support overlapped communication, will be disabled.")
                    overlap_comm = False

            # Tra()
            '''
            (Pdb) optimizer.__class__.__name__; self.zero_offload_optimizer(); contiguous_gradients;
            'DeepSpeedCPUAdam'
            DeepSpeedZeroOffloadOptimizerConfig(device='cpu', nvme_path=None, buffer_count=4, pin_memory=False, pipeline=False, pipeline_read=False, pipeline_write=False, fast_init=False, ratio=1.0)
            True

            (Pdb)self.gradient_accumulation_steps(); overlap_comm; round_robin_gradients;
            4
            True
            False

            (Pdb) self.param_names
            {Parameter containing:
            tensor([[ 0.0210, -0.0076, -0.0283,  ...,  0.0183,  0.0184, -0.0072],
                    [ 0.0260, -0.0310,  0.0128,  ..., -0.0278, -0.0259,  0.0139],
                    [ 0.0181,  0.0330, -0.0342,  ..., -0.0291, -0.0330, -0.0201],
                    ...,
                    [-0.0033, -0.0356,  0.0072,  ...,  0.0308, -0.0046, -0.0050],
                    [ 0.0177,  0.0061, -0.0354,  ..., -0.0208, -0.0198, -0.0266],
                    [-0.0162, -0.0013, -0.0216,  ...,  0.0258,  0.0327, -0.0059]],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True): 'fc1.weight', Parameter containing:
            tensor([ 0.0120, -0.0332,  0.0026,  ..., -0.0277,  0.0248, -0.0251],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True): 'fc1.bias', Parameter containing:
            tensor([[-0.0012, -0.0034, -0.0049,  ...,  0.0044, -0.0153, -0.0018],
                    [ 0.0020,  0.0117,  0.0048,  ...,  0.0001,  0.0051,  0.0028],
                    [-0.0041,  0.0056, -0.0053,  ..., -0.0023, -0.0037,  0.0136],
                    ...,
                    [ 0.0135, -0.0076, -0.0106,  ...,  0.0065,  0.0096,  0.0001],
                    [-0.0075,  0.0030,  0.0112,  ..., -0.0027,  0.0029,  0.0154],
                    [ 0.0120,  0.0001, -0.0126,  ...,  0.0114,  0.0062,  0.0009]],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True): 'fc2.weight', Parameter containing:
            tensor([-0.0082,  0.0090,  0.0061,  ...,  0.0002,  0.0014,  0.0090],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True): 'fc2.bias', Parameter containing:
            tensor([[ 0.0003, -0.0082, -0.0055,  ...,  0.0017, -0.0014,  0.0064],
                    [ 0.0086, -0.0059,  0.0075,  ...,  0.0064,  0.0105,  0.0101],
                    [-0.0148,  0.0055, -0.0134,  ..., -0.0012, -0.0093, -0.0093],
                    ...,
                    [-0.0098, -0.0028, -0.0022,  ...,  0.0100, -0.0034, -0.0055],
                    [ 0.0037, -0.0052, -0.0038,  ...,  0.0019, -0.0065,  0.0082],
                    [-0.0083, -0.0126, -0.0081,  ..., -0.0067,  0.0143,  0.0143]],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True): 'out.weight', Parameter containing:
            tensor([-0.0123, -0.0015, -0.0076, -0.0110,  0.0149,  0.0017,  0.0066, -0.0047,
                    0.0039, -0.0126], device='cuda:1', dtype=torch.bfloat16,
                requires_grad=True): 'out.bias'}

            (Pdb) optimizer.param_groups
            [{'params': [Parameter containing:
            tensor([[ 0.0003, -0.0082, -0.0055,  ...,  0.0017, -0.0014,  0.0064],
                    [ 0.0086, -0.0059,  0.0075,  ...,  0.0064,  0.0105,  0.0101],
                    [-0.0148,  0.0055, -0.0134,  ..., -0.0012, -0.0093, -0.0093],
                    ...,
                    [-0.0098, -0.0028, -0.0022,  ...,  0.0100, -0.0034, -0.0055],
                    [ 0.0037, -0.0052, -0.0038,  ...,  0.0019, -0.0065,  0.0082],
                    [-0.0083, -0.0126, -0.0081,  ..., -0.0067,  0.0143,  0.0143]],
                device='cuda:0', dtype=torch.bfloat16, requires_grad=True), Parameter containing:
            tensor([-0.0123, -0.0015, -0.0076, -0.0110,  0.0149,  0.0017,  0.0066, -0.0047,
                    0.0039, -0.0126], device='cuda:0', dtype=torch.bfloat16,
                requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.95), 'eps': 1e-08, 'weight_decay': 0, 'bias_correction': True, 'amsgrad': False}]
            (Pdb) optimizer.param_groups
            [{'params': [Parameter containing:
            tensor([[ 0.0003, -0.0082, -0.0055,  ...,  0.0017, -0.0014,  0.0064],
                    [ 0.0086, -0.0059,  0.0075,  ...,  0.0064,  0.0105,  0.0101],
                    [-0.0148,  0.0055, -0.0134,  ..., -0.0012, -0.0093, -0.0093],
                    ...,
                    [-0.0098, -0.0028, -0.0022,  ...,  0.0100, -0.0034, -0.0055],
                    [ 0.0037, -0.0052, -0.0038,  ...,  0.0019, -0.0065,  0.0082],
                    [-0.0083, -0.0126, -0.0081,  ..., -0.0067,  0.0143,  0.0143]],
                device='cuda:1', dtype=torch.bfloat16, requires_grad=True), Parameter containing:
            tensor([-0.0123, -0.0015, -0.0076, -0.0110,  0.0149,  0.0017,  0.0066, -0.0047,
                    0.0039, -0.0126], device='cuda:1', dtype=torch.bfloat16,
                requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.95), 'eps': 1e-08, 'weight_decay': 0, 'bias_correction': True, 'amsgrad': False}]
            '''
            optimizer = DeepSpeedZeroOptimizer(
                optimizer, # CPU ADAM OPTIMIZER
                self.param_names,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                use_multi_rank_bucket_allreduce=self.zero_multi_rank_bucket_allreduce(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.seq_data_parallel_group,
                expert_parallel_group= None,
                expert_data_parallel_group= None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                offload_optimizer_config=self.zero_offload_optimizer(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=False,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                gradient_accumulation_dtype=gradient_accumulation_dtype,
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint())

        ## zero-3
        elif zero_stage == ZeroStageEnum.weights:

            assert not self.has_moe_layers, "MoE not supported with Stage 3"

            ## dummy optimizer
            if isinstance(optimizer, DummyOptim):
                log_dist("Creating ZeRO Offload", ranks=[0])
                zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
                if self.zero_hpz_partition_size() > 1 and zero_param_parallel_group is None:
                    self._set_zero_group_parallelism()
                    zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()

                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu,
                    zero_param_parallel_group=zero_param_parallel_group,
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                )

            ## adam and etc optimizer 
            else:
                log_dist(
                    f'Creating fp16 ZeRO stage {zero_stage} optimizer,'
                    f' MiCS is enabled {mics_shard_size>0},'
                    f' Hierarchical params gather {self._config.mics_hierarchial_params_gather}',
                    ranks=[0])

                if mics_shard_size > 0:
                    return self._return_mics_optimizer(optimizer, timers)

                log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                optimizer = DeepSpeedZeroOptimizer_Stage3(
                    self.module,
                    optimizer,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.seq_data_parallel_group,
                    all2all_process_group=self.local_all_to_all_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    offload_ratio=self.zero_partial_offload(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    gradient_accumulation_dtype=gradient_accumulation_dtype,
                    communication_data_type=self.communication_data_type,
                    zero_hpz_partition_size=self.zero_hpz_partition_size(),
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                )

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    def _return_mics_optimizer(self, basic_optimizer, timers):
        from deepspeed.runtime.zero.mics import MiCS_Optimizer
        model_dtype, gradient_accumulation_dtype = self.get_data_types()
        optimizer = MiCS_Optimizer(self.module,
                                   basic_optimizer,
                                   timers=timers,
                                   ds_config=self.config,
                                   static_loss_scale=self.loss_scale(),
                                   dynamic_loss_scale=self.dynamic_loss_scale(),
                                   dynamic_loss_args=self.dynamic_loss_scale_args(),
                                   clip_grad=self.gradient_clipping(),
                                   contiguous_gradients=self.zero_contiguous_gradients(),
                                   reduce_bucket_size=self.zero_reduce_bucket_size(),
                                   prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                                   max_reuse_distance=self.zero_max_reuse_distance(),
                                   max_live_parameters=self.zero_max_live_parameters(),
                                   param_persistence_threshold=self.zero_param_persistence_threshold(),
                                   model_persistence_threshold=self.zero_model_persistence_threshold(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   reduce_scatter=self.zero_reduce_scatter(),
                                   overlap_comm=self.zero_overlap_comm(),
                                   offload_optimizer_config=self.zero_offload_optimizer(),
                                   offload_param_config=self.zero_offload_param(),
                                   sub_group_size=self.zero_sub_group_size(),
                                   mpu=self.mpu,
                                   postscale_gradients=self.postscale_gradients(),
                                   gradient_predivide_factor=self.gradient_predivide_factor(),
                                   gradient_accumulation_steps=self.gradient_accumulation_steps(),
                                   aio_config=self.aio_config(),
                                   gradient_accumulation_dtype=gradient_accumulation_dtype,
                                   communication_data_type=self.communication_data_type)
        return optimizer


    #############################################################################################################
    ##########################################     forward propagation     ######################################
    #############################################################################################################

    @instrument_w_nvtx
    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """

        see_memory_usage("Engine before forward", force=self.memory_breakdown())

        flops_profiler_active = (self.flops_profiler_enabled()
                                 and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0)

        if flops_profiler_active:
            self.flops_profiler.start_profile(ignore_list=None)

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True

        self._start_timers(self.engine_timers.forward_timers)
        if self.training_dataloader is None:
            self.tput_timer.start()

        ## make inputs half, why?
        if self.fp16_auto_cast():
            inputs = self._cast_inputs_half(inputs)

        ## forward
        loss = self.module(*inputs, **kwargs)

        if self.zero_optimization_partition_weights():
            # Disable automated discovery of external parameters
            for module in self.module.modules(): 
                module._parameters._in_forward = False

        self._stop_timers(self.engine_timers.forward_timers)
        if flops_profiler_active:
            self.flops_profiler.stop_profile()

        see_memory_usage("Engine after forward", force=self.memory_breakdown())
        return loss

    def _cast_inputs_half(self, inputs):
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for v in inputs:
                new_inputs.append(self._cast_inputs_half(v))
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = self._cast_inputs_half(v)
            return new_inputs
        elif hasattr(inputs, 'half'):
            return inputs.half()
        else:
            return inputs

    def print_forward_breakdown(self, fwd_time):
        gate_time = 0.0
        moe_time = 0.0
        falltoall = 0.0
        salltoall = 0.0

        # if deepspeed.comm.get_rank() == 0:
        log_dist(
            f"time (ms) | fwd: {fwd_time:.2f} (fwd_moe: {moe_time:.2f}, 1st_a2a: {falltoall:.2f}, 2nd_a2a: {salltoall:.2f}, top_k: {gate_time:.2f})",
            ranks=[0])


    ########################################################################################################
    ##########################################     back propagation     ####################################
    ########################################################################################################

    @instrument_w_nvtx
    def backward(self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
            retain_graph: bool, default: false
                forward on user defined choice of retain_graph
        """

        ## scale loss w.r.t. gradient accumulation if needed
        see_memory_usage("Engine before backward", force=self.memory_breakdown())
        if self.scale_wrt_gas is not None:
            scale_wrt_gas = self.scale_wrt_gas
        if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
            loss = self._scale_loss_by_gas(loss.float()) # gradient accumulation is default fp32

        ## Log training loss
        mean_loss = loss.mean().detach()
        self.losses = mean_loss if self.losses is None else self.losses + mean_loss
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(
                        f"Train/Samples/train_loss",
                        self.losses.item(),
                        self.global_samples,
                    )]
                    self.monitor.write_events(self.summary_events)

        self._start_timers(self.engine_timers.backward_timers)
        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), "must provide optimizer during init in order to use backward"
        self._start_timers(self.engine_timers.backward_inner_timers)


        '''
        https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
        create_graph = for higher order like newton method?
        zero optimization backward perform accumulation?
        '''
        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
            self.optimizer.backward(loss, retain_graph=retain_graph)

        ## other optimizers
        elif self.amp_enabled():
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        elif self.fp16_enabled():
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.bfloat16_enabled():
            self.optimizer.backward(loss)
        else:
            loss.backward(retain_graph=retain_graph)

        ## allreduce gradients after backward
        self._stop_timers(self.engine_timers.backward_inner_timers)
        self._start_timers(self.engine_timers.backward_reduce_timers)
        if allreduce_gradients and self.enable_backward_allreduce:
            self.allreduce_gradients() # Traditional code path that allreduces the module parameter grads
        self._stop_timers(self.engine_timers.backward_reduce_timers)
        self._stop_timers(self.engine_timers.backward_timers)

        if release_loss:
            # loss.data = None
            pass
        see_memory_usage("Engine after backward", force=self.memory_breakdown())

        return loss

    def _scale_loss_by_gas(self, prescaled_loss, eval_micro_batches=None):
        # In pipeline evaluation, there is an option to use different micro-bs, which creates different number of
        # micro batches, thus the training gas, is not valid in this case. need to use the number of eval_micro_batches
        scaling_factor = self.gradient_accumulation_steps() if eval_micro_batches is None else eval_micro_batches

        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / scaling_factor
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / scaling_factor)
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(f"DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}")
                self.warn_unscaled_loss = False

        return scaled_loss
        
    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()

        # ZeRO stage >= 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():

            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and hasattr(self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)

            else:
                grads = None
                if hasattr(self.optimizer, "get_grads_for_reduction"):
                    # This is currently for BF16 optimizer
                    grads = self.optimizer.get_grads_for_reduction()
                self.buffered_allreduce_fallback(grads=grads, elements_per_buffer=bucket_size)

    ########################################################################################################
    ##########################################     optimizer step     ######################################
    ########################################################################################################

    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.
        Returns:
            bool: if the current step is a gradient accumulation boundary.
        """
        if self._is_gradient_accumulation_boundary is None:
            return (self.micro_steps + 1) % \
                self.gradient_accumulation_steps() == 0
        else:
            return self._is_gradient_accumulation_boundary

    def step(self, lr_kwargs=None):
        r"""Execute the weight update step after forward and backward propagationon effective_train_batch."""
        see_memory_usage("Engine before step", force=self.memory_breakdown())

        # Check early because self.global_steps is incremented at some point here.
        # TODO: Delay self.global_steps increment until very end of this function.
        flops_profiler_active = self.flops_profiler_enabled() and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0

        self._start_timers(self.engine_timers.step_timers)
        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), "must provide optimizer during init in order to use step"
        report_progress = False
        self._step_applied = False  # assume False, will flip to True

        # Update the model when we reach gradient accumulation boundaries
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1
            self._take_model_step(lr_kwargs)

            report_progress = self.global_rank == 0 if self.global_rank else True

        self.tput_timer.stop(global_step=self.is_gradient_accumulation_boundary(), report_speed=report_progress)

        self._stop_timers(self.engine_timers.step_timers)

        # Log learning rate
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(f"Train/Samples/lr", self.get_lr()[0], self.global_samples)]

                    if self.fp16_enabled() and hasattr(self.optimizer, "cur_scale"):
                        self.summary_events.append((
                            f"Train/Samples/loss_scale",
                            self.optimizer.cur_scale,
                            self.global_samples,
                        ))
                    self.monitor.write_events(self.summary_events)

        # Check flops profiling
        if flops_profiler_active:
            self.flops_profiler.print_model_profile(
                profile_step=self.global_steps,
                module_depth=self.flops_profiler_module_depth(),
                top_modules=self.flops_profiler_top_modules(),
                detailed=self.flops_profiler_detailed(),
                output_file=self.flops_profiler_output_file(),
            )
            self.flops_profiler.end_profile()

        if self.wall_clock_breakdown():
            # Log micro timing and reset
            self.timers.log(names=self.engine_timers.micro_timers, memory_breakdown=self.memory_breakdown())

        if self.wall_clock_breakdown() or self.flops_profiler_enabled():
            # Log global timing and reset
            if self.is_gradient_accumulation_boundary():
                if self.monitor.enabled:
                    self._write_monitor()

                if self.has_moe_layers:
                    raise NotImplementedError("removed!")

                self.timers.log(self.engine_timers.global_timers)

        self.micro_steps += 1
        see_memory_usage("Engine after step", force=self.memory_breakdown())

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):

        if self.gradient_clipping() > 0.0:
            if not (self.fp16_enabled() or self.bfloat16_enabled() or self.amp_enabled() or self.zero_optimization()):
                # 사실 zero도, 뭣도 아니고 fp32로 deepspeed를 쓸일은 없음.
                # there was a bug (https://github.com/microsoft/DeepSpeed/pull/5150)
                self.clip_fp32_gradients()

            elif self.amp_enabled():
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params, max_norm=self.gradient_clipping(), mpu=self.mpu)

        ## parameter update -> if we are using zero >= 2, gradient should be gathered 
        self.optimizer.step()

        ## gradient is already scaled if we train model with mixed precision. so unscale first, then compute L2 norm
        if hasattr(self.optimizer, '_global_grad_norm'):
            self._global_grad_norm = self.optimizer._global_grad_norm

        # zero grad in basic optimizer could be unreliable and may not exhibit, the behavior that we want
        if self.bfloat16_enabled():
            if self.zero_optimization() and hasattr(self.optimizer, "zero_grad"): # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
                self.optimizer.zero_grad()
            else:
                pass
        elif self.zero_optimization() or self.fp16_enabled() or self.amp_enabled():
            self.optimizer.zero_grad()
        else:
            self.zero_grad()

        report_progress = self.global_rank == 0 if self.global_rank else True

        # Check overflow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        self._step_applied = not overflow # False -> True (flipped)

        if overflow:
            self.skipped_steps += 1
        else:
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step(**(lr_kwargs or {}))
                except TypeError:
                    # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
                    # We don't currently have a way to specify lr_kwargs from
                    # pipe_engine.train_batch()
                    self.lr_scheduler.step(self.train_batch_size())

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

        self.losses = None
        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def clip_fp32_gradients(self):
        clip_grad_norm_(parameters=self.module.parameters(), max_norm=self.gradient_clipping(), mpu=self.mpu)

    def zero_grad(self):
        for param_name, param in self.module.named_parameters():
            param.grad = None


    ########################################################################################################
    #########################################  (sparse)  all reduce   ######################################
    ########################################################################################################

    def allreduce_bucket(self, bucket, dp_group):
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)

        if self.postscale_gradients():
            if self.gradient_predivide_factor() != 1.0:
                tensor_to_allreduce.mul_(1.0 / self.gradient_predivide_factor())

            dist.all_reduce(tensor_to_allreduce, group=dp_group)
            if self.gradient_average:
                if self.gradient_predivide_factor() != dist.get_world_size(group=dp_group):
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() / dist.get_world_size(group=dp_group))
        else:
            tensor_to_allreduce.mul_(1. / dist.get_world_size(group=dp_group))
            dist.all_reduce(tensor_to_allreduce, group=dp_group)

        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket, dp_group):
        allreduced = self.allreduce_bucket(small_bucket, dp_group)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, dp_group, numel_per_bucket=500000000):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, dp_group)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, dp_group)

    def _get_gradients_for_reduction(self):
        non_expert_grads = []
        expert_grads = {}
        if self.has_moe_layers:
            raise NotImplementedError("removed!")

        for param_name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(), dtype=param.dtype, device=param.device)

            grad_data = param.grad.data
            if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                # Call param.grad without data to avoid problem with setting of updated grads
                grad_data = SparseTensor(param.grad)

            # if is_moe_param(param):
            #     expert_grads[param.group_name].append(grad_data)
            # else:
            non_expert_grads.append(grad_data)

        return non_expert_grads, expert_grads

    def _reduce_non_expert_gradients(self, grads, elements_per_buffer):
        split_sparse_tensor_buckets, split_dense_tensor_buckets = split_half_float_double_sparse(grads)
        if self.pipeline_parallelism:
            dp_group = self.mpu.get_data_parallel_group()
        else:
            dp_group = groups._get_sequence_data_parallel_group()

        for _, sparse_bucket_tuple in enumerate(split_sparse_tensor_buckets):
            if sparse_bucket_tuple:
                bucket_type, sparse_bucket = sparse_bucket_tuple
                self.sparse_allreduce_no_retain(sparse_bucket, dp_group=dp_group)

        for _, dense_bucket_tuple in enumerate(split_dense_tensor_buckets):
            if dense_bucket_tuple:
                bucket_type, dense_bucket = dense_bucket_tuple
                self.allreduce_no_retain(dense_bucket, dp_group=dp_group, numel_per_bucket=elements_per_buffer)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        if grads is None:
            non_expert_grads, expert_grads = self._get_gradients_for_reduction()
        else:
            assert not self.has_moe_layers, "attempting to reduce grads in unsupported way w.r.t. MoE"
            non_expert_grads = grads

        self._reduce_non_expert_gradients(non_expert_grads, elements_per_buffer)

        if self.has_moe_layers:
            raise NotImplementedError("removed!")

    def sparse_allreduce_no_retain(self, bucket, dp_group):
        allreduced_sparses = self.sparse_allreduce_bucket(bucket, dp_group)
        # Densify sparse tensor and copy back to original location
        for tensor in allreduced_sparses:
            if tensor.is_sparse:
                tensor.orig_dense_tensor.data = tensor.to_coo_tensor()
            else:
                tensor.orig_dense_tensor.copy_(tensor.to_dense())

    def sparse_allreduce_bucket(self, bucket, dp_group):
        sparse_list = []
        for sparse in bucket:
            sparse_list.append(self.sparse_allreduce(sparse, dp_group))
        return sparse_list

    def sparse_allreduce(self, sparse, dp_group):
        original_data_type = sparse.values.dtype
        if self.communication_data_type != sparse.values.dtype:
            if self.communication_data_type in (torch.float16, torch.bfloat16):
                indices = sparse.indices.to(torch.int32)
            else:
                indices = sparse.indices
            values = sparse.values.to(self.communication_data_type)
        else:
            indices = sparse.indices
            values = sparse.values

        if self.postscale_gradients():
            if self.gradient_average:
                values.mul_(self.gradient_predivide_factor() /
                            (dist.get_world_size(group=dp_group) / float(self.sequence_parallel_size)))
        else:
            values.mul_(1. / (dist.get_world_size(group=dp_group) / float(self.sequence_parallel_size)))

        indices_device_list = self.sparse_all_gather(indices, dp_group)
        values_device_list = self.sparse_all_gather(values, dp_group)

        sparse.indices = torch.cat(indices_device_list).to(torch.long)
        sparse.values = torch.cat(values_device_list).to(original_data_type)
        return sparse


    ########################################################################################################
    #########################################   (sparse) all gather   ######################################
    ########################################################################################################
    def sparse_all_gather(self, value, dp_group):
        my_size = torch.LongTensor([value.size()[0]]).to(self.device)
        all_sizes = self.all_gather_scalar(my_size, dp_group)
        max_size = torch.cat(all_sizes).max()
        fill_size = max_size - my_size

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size)])
            tensor_list = [value.new_empty(max_size) for _ in range(dist.get_world_size(group=dp_group))]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size, value.size()[1])])
            tensor_list = [
                value.new_empty(max_size,
                                value.size()[1]) for _ in range(dist.get_world_size(group=dp_group))
            ]

        dist.all_gather(tensor_list, value, group=dp_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(t.index_select(0, torch.arange(size, dtype=torch.long, device=self.device)))

        return tensors

    def all_gather_scalar(self, value, dp_group):
        tensor_list = [value.new_zeros(value.size()) for _ in range(dist.get_world_size(group=dp_group))]
        dist.all_gather(tensor_list, value, group=dp_group)
        return tensor_list

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled







    ###############################################################################################################
    ####################################### model param info and data related methods  ############################
    ###############################################################################################################

    def __getattr__(self, name):
        """
        Pass through attributes defined in the model if they are not overridden by ds-engine.
        """

        _module = {}
        if "module" in self.__dict__:
            _module = self.__dict__['module']
        if name in dir(self):
            return getattr(self, name)
        elif name in dir(_module):
            return getattr(_module, name)
        elif isinstance(_module, CompiledModuleWrapper):
            try:
                return getattr(_module, name)
            except AttributeError:
                raise AttributeError(
                    f"None of {type(self).__name__}, CompiledModuleWrapper, or the wrapped model has the attribute '{name}'"
                )
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_batch_info(self):
        """Get all training batch related settings.
        Returns:
            train_batch_size (int): The effective training batch size. This is the amount of data
                samples that leads to one step of model update.
            train_micro_batch_size_per_gpu (int): Batch size to be processed by one GPU in one
                step (without gradient accumulation).
            gradient_accumulation_steps (int): Number of training steps to accumulate gradients
                before averaging and applying them.
        """
        return (
            self.train_batch_size,
            self.train_micro_batch_size_per_gpu,
            self.gradient_accumulation_steps,
        )

    def data_efficiency_enabled(self):
        return self._config.data_efficiency_enabled

    def data_efficiency_config(self):
        return self._config.data_efficiency_config

    def data_sampling_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_ENABLED]

    def data_sampling_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING]

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        if train_batch_size % (self.train_micro_batch_size_per_gpu() * self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError(f'Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() * self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

    def set_train_micro_batch_size(self, micro_batch_size):
        """Adjust the micro batch size(i.e., the micro batch size in every data parallel group),
        while keep the gradient accumulation steps the same.
        Args:
            micro_batch_size (int): The new micro batch size for training.
        """
        # overwrite config
        new_global_batch_size = micro_batch_size * self._config.gradient_accumulation_steps * self.dp_world_size
        self._config.train_batch_size = new_global_batch_size
        self._config.train_micro_batch_size_per_gpu = micro_batch_size

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu


    ########################################################################################################
    ####################################### get optimizer stats and get lr #################################
    ########################################################################################################

    def _get_optimizer_param(self, param_name):
        result = []
        if not self.optimizer:
            return result
        for group in self.optimizer.param_groups:
            if param_name in group:
                result.append(group[param_name])
            else:
                result.append(0.0)
        return result

    def get_lr(self):
        return self._get_optimizer_param("lr")

    def get_type(self):
        return self._get_optimizer_param("type")

    def get_mom(self):
        if self.optimizer_name() in ["SGD", "RMSprop"]:
            return self._get_optimizer_param("momentum")
        else:
            return self._get_optimizer_param("betas")

    def _report_progress(self, step):
        lr = self.get_lr()
        mom = self.get_mom()
        log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr}, mom={mom}", ranks=[0])

    ##################################################################################
    ####################################### elastic  #################################
    ##################################################################################

    def elasticity_enabled(self):
        return self._config.elasticity_enabled

    def is_elastic_model_parallel_supported(self):
        if self.elasticity_enabled():
            # Add code for finding number of GPUs per node automatically
            if self._config.num_gpus_per_node % self._config.elastic_model_parallel_size == 0:
                return True
            else:
                return False



    ##############################################################################################
    ####################################### timer and profilers  #################################
    ##############################################################################################

    def _start_timers(self, timer_names):
        for name in timer_names:
            self.timers(name).start()

    def _stop_timers(self, timer_names):
        record = self.is_gradient_accumulation_boundary() and \
            self.flops_profiler_enabled() and \
                (self.global_steps >= self.flops_profiler_profile_step())
        for name in timer_names:
            self.timers(name).stop(record=record)

    def _write_monitor(self):
        if self.global_rank == 0:
            self.summary_events = [
                (
                    f"Train/Samples/elapsed_time_ms_forward",
                    self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward",
                    self.timers(BACKWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_inner",
                    self.timers(BACKWARD_INNER_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_allreduce",
                    self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_step",
                    self.timers(STEP_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
            ]
            self.monitor.write_events(self.summary_events)

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def flops_profiler_enabled(self):
        return self._config.flops_profiler_config.enabled

    def flops_profiler_recompute_fwd_factor(self):
        return self._config.flops_profiler_config.recompute_fwd_factor

    def flops_profiler_profile_step(self):
        step = self._config.flops_profiler_config.profile_step
        return step

    def flops_profiler_module_depth(self):
        return self._config.flops_profiler_config.module_depth

    def flops_profiler_top_modules(self):
        return self._config.flops_profiler_config.top_modules

    def flops_profiler_detailed(self):
        return self._config.flops_profiler_config.detailed

    def flops_profiler_output_file(self):
        return self._config.flops_profiler_config.output_file

    def memory_breakdown(self):
        return self._config.memory_breakdown


    ##################################################################################################################
    ####################################### optimizer and scheduler related mtehods  #################################
    ##################################################################################################################

    def optimizer_name(self):
        return (self.client_optimizer.__class__.__name__ if self.client_optimizer else self._config.optimizer_name)

    def optimizer_params(self):
        return self._config.optimizer_params

    def optimizer_legacy_fusion(self):
        return self._config.optimizer_legacy_fusion

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def get_global_grad_norm(self) -> float:
        """Return the 2-norm of all gradients. If there is model parallelism,
        the norm will be global.
        The computed norm will be cached and reused until the next step() pass.
        .. note::
            In the presence of model parallelism, this is a collective call
            and acts as a barrier among ``mpu.get_model_parallel_group()``.
        Returns:
            float: norm
        """
        return self._global_grad_norm

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def set_gradient_accumulation_boundary(self, is_boundary):
        """
        Manually overrides the DeepSpeed engine's gradient accumulation boundary state, this is an optional
        feature and should be used with care. The state should be set before to the intended
        value before each forward/backward. The final forward/backward should have the
        boundary state set to True. This style allows client code to only call engine.step() once after all
        the gradient accumulation passes are complete. See example below:
        .. code-block:: python
        engine.set_gradient_accumulation_boundary(False)
        for _ in range(gradient_accumulation_steps - 1):
            micro_batch = next(data_loader)
            loss = engine(micro_batch)
            engine.backward(loss)
        engine.set_gradient_accumulation_boundary(True)
        micro_batch = next(data_loader)
        loss = engine(micro_batch)
        engine.backward(loss)
        engine.step()
        Arguments:
            is_boundary (bool): are we at a gradient accumulation boundary or not?
        """
        self._is_gradient_accumulation_boundary = is_boundary
        self.optimizer.is_gradient_accumulation_boundary = is_boundary

    #########################################################################################################
    ####################################### zero optimizer related methods  #################################
    #########################################################################################################

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_force_ds_cpu_optimizer(self):
        return self._config.zero_force_ds_cpu_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_offload_optimizer(self):
        return self._config.zero_config.offload_optimizer

    def zero_offload_param(self):
        return self._config.zero_config.offload_param

    def zero_use_cpu_optimizer(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme]
        return False

    def zero_cpu_offload(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device == OffloadDeviceEnum.cpu
        return False

    def zero_partial_offload(self):
        return getattr(self._config.zero_config.offload_optimizer, "ratio", 1.0)

    def zero_sub_group_size(self):
        return self._config.zero_config.sub_group_size

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def mics_shard_size(self):
        return self._config.mics_shard_size

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_multi_rank_bucket_allreduce(self):
        return self._config.zero_config.use_multi_rank_bucket_allreduce

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.gradients

    def zero_optimization_partition_weights(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.weights

    def is_first_weights_partition_group(self):
        ret = True if self.mics_shard_size() < 0 \
            and self.zero_optimization_partition_weights() else False
        if self.mics_shard_size() > 0 and self.global_rank < self.mics_shard_size():
            ret = True
        return ret

    def zero_contiguous_gradients(self):
        return self._config.zero_config.contiguous_gradients

    def zero_load_from_fp32_weights(self):
        return self._config.zero_config.load_from_fp32_weights

    def zero_elastic_checkpoint(self):
        return self._config.zero_config.elastic_checkpoint

    def zero_has_nvme_offload(self):
        if not hasattr(self.optimizer, "swap_optimizer"):
            return False
        return self.optimizer.swap_optimizer or self.optimizer.params_in_nvme_and_cpu

    def zero_max_live_parameters(self):
        return self._config.zero_config.max_live_parameters

    def zero_max_reuse_distance(self):
        return self._config.zero_config.max_reuse_distance

    def zero_prefetch_bucket_size(self):
        return self._config.zero_config.prefetch_bucket_size

    def zero_param_persistence_threshold(self):
        return self._config.zero_config.param_persistence_threshold

    def zero_model_persistence_threshold(self):
        return self._config.zero_config.model_persistence_threshold

    def zero_gather_16bit_weights_on_model_save(self):
        return self._config.zero_config.gather_16bit_weights_on_model_save

    def zero_grad_hooks(self):
        return self._config.zero_config.grad_hooks

    def zero_legacy_stage1(self):
        return self._config.zero_config.legacy_stage1

    def zero_ignore_unused_parameters(self):
        return self._config.zero_config.ignore_unused_parameters

    def graph_harvesting(self):
        return self._config.graph_harvesting



    def get_data_types(self):
        model_dtype = torch.float32
        if self.fp16_enabled():
            model_dtype = torch.float16
        elif self.bfloat16_enabled():
            model_dtype = torch.bfloat16

        if self._config.grad_accum_dtype is None:
            if model_dtype == torch.bfloat16 and not self.zero_optimization():
                grad_accum_dtype = torch.float32
            else:
                grad_accum_dtype = model_dtype
        else:
            grad_accum_dtype = DtypeEnum(self._config.grad_accum_dtype).value

        return (model_dtype, grad_accum_dtype)
    
    def fp16_enabled(self):
        return self._config.fp16_enabled

    def bfloat16_enabled(self):
        return self._config.bfloat16_enabled

    def fp16_master_weights_and_gradients(self):
        return self._config.fp16_master_weights_and_gradients

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def fp16_auto_cast(self):
        return self._config.fp16_auto_cast

    def loss_scale(self):
        return self._config.loss_scale


    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res

        if self.fp16_enabled():
            return torch.float16

        if self.bfloat16_enabled():
            return torch.bfloat16

        return torch.float32

    @communication_data_type.setter
    def communication_data_type(self, value):
        self._config.communication_data_type = value

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def zero_allgather_partitions(self):
        return self._config.zero_config.allgather_partitions

    def zero_round_robin_gradients(self):
        return self._config.zero_config.round_robin_gradients

    def zero_hpz_partition_size(self):
        return self._config.zero_config.zero_hpz_partition_size

    def zero_quantized_weights(self):
        return self._config.zero_config.zero_quantized_weights

    def zero_quantized_nontrainable_weights(self):
        return self._config.zero_config.zero_quantized_nontrainable_weights

    def zero_quantized_gradients(self):
        return self._config.zero_config.zero_quantized_gradients

    def dump_state(self):
        return self._config.dump_state

    def gradient_clipping(self):
        return self._config.gradient_clipping

    def dynamic_loss_scale(self):
        return self._config.loss_scale == 0

    def initial_dynamic_scale(self):
        return self._config.initial_dynamic_scale

    def dynamic_loss_scale_args(self):
        return self._config.dynamic_loss_scale_args

    def swap_tensor_config(self):
        return self._config.swap_tensor_config



    ##################################################################################################################
    ####################################### do configure, sanity check... and so on  #################################
    ##################################################################################################################

    def _configure_lr_scheduler(self, client_lr_scheduler):
        # First check for scheduler in json configuration
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        if lr_scheduler:
            log_dist(f"DeepSpeed using configured LR scheduler = {self.scheduler_name()}", ranks=[0])
            self.lr_scheduler = lr_scheduler
        else:
            if isinstance(client_lr_scheduler, Callable):
                log_dist('DeepSpeed using client callable to create LR scheduler', ranks=[0])
                self.lr_scheduler = client_lr_scheduler(self.basic_optimizer)
            else:
                log_dist('DeepSpeed using client LR scheduler', ranks=[0])
                self.lr_scheduler = client_lr_scheduler

        log_dist(f'DeepSpeed LR Scheduler = {self.lr_scheduler}', ranks=[0])

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler,
                               scheduler_name), f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _set_distributed_vars(self, args):
        device_rank = args.device_rank if args is not None and hasattr(args, 'device_rank') else self.local_rank
        if device_rank >= 0:
            get_accelerator().set_device(device_rank)
            self.device = torch.device(get_accelerator().device_name(), device_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device(get_accelerator().device_name())

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatibility with scripts relying on [args|self].local_rank containing
        # the correct local rank info. _do_args_sanity_check will ensure this is the case.

        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            ompi_local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            local_rank = os.environ.get('LOCAL_RANK', ompi_local_rank)
            assert ompi_local_rank == local_rank, f"LOCAL_RANK ({local_rank}) != OMPI_COMM_WORLD_LOCAL_RANK ({ompi_local_rank}), " \
                "not sure how to proceed as we're seeing conflicting local rank info."
            os.environ['LOCAL_RANK'] = local_rank

        self.local_rank = int(os.environ['LOCAL_RANK'])
        if hasattr(args, 'local_rank'):
            args.local_rank = self.local_rank

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch's launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank is not None:
            assert isinstance(args.local_rank,
                              int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert (
                    env_local_rank == args.local_rank
                ), f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

    def _is_supported_optimizer(self, optimizer_name):
        return (optimizer_name in DEEPSPEED_OPTIMIZERS or getattr(torch.optim, optimizer_name, None) is not None)

    def _supported_optims(self):
        FairseqOptimizer = None
        try:
            from fairseq.optim.fairseq_optimizer import FairseqOptimizer
        except ImportError:
            pass

        expected_optim_types = [Optimizer]
        if FairseqOptimizer:
            # fairseq optims are not torch.optim objects
            expected_optim_types.append(FairseqOptimizer)
        return expected_optim_types

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        if self.fp16_enabled() and not get_accelerator().is_fp16_supported():
            raise ValueError("Type fp16 is not supported.")

        expected_optim_types = self._supported_optims()
        expected_optim_types += [type(None), Callable]
        assert isinstance(self.client_optimizer, tuple(expected_optim_types)), \
            f'Client Optimizer is of unexpected type {type(self.client_optimizer)}'

        if not self.client_optimizer:
            if self.optimizer_name() is not None:
                assert self._is_supported_optimizer(
                    self.optimizer_name()), "{} is not a supported DeepSpeed Optimizer".format(self.optimizer_name())

        if (self.optimizer_name() == LAMB_OPTIMIZER or self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER):
            assert (self.dynamic_loss_scale()), "DeepSpeed {} optimizer requires dynamic loss scaling".format(
                self.optimizer_name())

        # Detect invalid combinations of client optimizer and client scheduler
        if isinstance(self.client_lr_scheduler, _LRScheduler):
            assert isinstance(self.client_optimizer, Optimizer), \
                f'Client Optimizer (type = {type(self.client_optimizer)} is not instantiated but Client LR Scheduler is instantiated'

    def _broadcast_model(self):
        def is_replicated(p):
            if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
                return False
            return True

        for p in self.module.parameters():
            # Broadcast the model for different parameters
            # if is_moe_param(p):
            #     raise NotImplementedError("removed!")
            # else:
            if torch.is_tensor(p) and is_replicated(p):
                dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)

    @staticmethod
    def __check_params(model: Module, dtype: torch.dtype) -> None:
        return
        if not all(param.dtype == dtype for param in model.parameters()) and dist.get_rank() == 0:
            raise ValueError(f"{dtype} is enabled but the following parameters have dtype that is "
                             f"not {dtype}: "
                             f"{[(n, p.dtype) for n, p in model.named_parameters() if p.dtype != dtype]}")

    def _set_client_model(self, model):
        # register client model in _modules so that nn.module methods work correctly
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        # register module attribute in engine but avoid getattr
        self.__dict__['module'] = model

    def _configure_distributed_model(self, model):
        self._set_client_model(model)
        is_zero_init_model = self.zero_optimization_partition_weights() and any(
            [hasattr(param, "ds_id") for param in self.module.parameters()]
        )

        if self.fp16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.half)
            self.module.half()
        elif self.bfloat16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.bfloat16)
            self.module.bfloat16()
        else:
            self.__check_params(self.module, torch.float)

        # zero.Init() handles device placement of model
        if not (self.dont_change_device or is_zero_init_model):
            self.module.to(self.device)

        # # MoE related initialization
        # for _, module in self.module.named_modules():
        #     if isinstance(module, MoE):
        #         raise NotImplementedError("removed!")

        if self.has_moe_layers:
            raise NotImplementedError("removed!")

        # Pass the mpu from here to groups. For subsequent use, just query groups
        if self.mpu is not None:
            groups.mpu = self.mpu

        # Set deepspeed parallelism spec. for the model including expert parallelism
        for _, module in self.module.named_modules():
            if hasattr(module, 'set_deepspeed_parallelism'):
                module.set_deepspeed_parallelism(self._config.use_data_before_expert_parallel_)

        # Query the groups module to get information about various parallel groups
        self.local_all_to_all_group = None
        if self.zero_quantized_gradients():
            log_dist("Using quantized gradients", ranks=[0])
            self.local_all_to_all_group = groups._get_local_all_to_all_group()
        self.data_parallel_group = groups._get_data_parallel_group()
        self.dp_world_size = groups._get_data_parallel_world_size()
        self.seq_data_parallel_group = groups._get_sequence_data_parallel_group()
        self.seq_dp_world_size = groups._get_sequence_data_parallel_world_size()
        self.mp_world_size = groups._get_model_parallel_world_size()
        # Tra()
        '''
        (Pdb) groups; self.data_parallel_group; self.dp_world_size; self.seq_data_parallel_group; self.seq_dp_world_size; self.mp_world_size;
        <module 'deepspeed.utils.groups' from '/path/to/dir/deepspeed_profiling/pruned_deepspeed/DeepSpeed/deepspeed/utils/groups.py'>
        <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fe87ce31d30>
        2
        <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fe87ce31d30>
        2
        1
        '''

        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        if self.sequence_parallel_size > 1:
            self.communication_data_type = self._config.seq_parallel_communication_data_type

        if not (self.amp_enabled() or is_zero_init_model):
            self._broadcast_model()
        # Tra()
        '''
        (Pdb) self.sequence_parallel_size
        1
        '''

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id) if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behavior."

    def _do_optimizer_sanity_check(self, basic_optimizer):
        model_dtype, grad_accum_dtype = self.get_data_types()
        zero_enabled = self.zero_optimization()
        amp_enabled = self.amp_enabled()
        # config based assertions
        assert (
            not (amp_enabled and zero_enabled)
        ), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
        if zero_enabled:
            if not is_zero_supported_optimizer(basic_optimizer):
                assert (
                    self.zero_allow_untested_optimizer()
                ), 'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning("**** You are using ZeRO with an untested optimizer, proceed with caution *****")
            if model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32 and self.zero_optimization_stage(
            ) == 1 and not self.zero_cpu_offload():
                return BFLOAT16
            return ZERO_OPTIMIZATION
        elif amp_enabled:
            if model_dtype != grad_accum_dtype:
                raise NotImplementedError(
                    "Model data type and gradient accumulation data type must be equal to use Amp")
            if model_dtype == torch.bfloat16 or model_dtype == torch.float16:
                raise NotImplementedError("Cannot enable both amp with (legacy) fp16 or bfloat16 mode")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError("Unable to import apex/amp, please make sure it is installed")
            return AMP
        # data type checks
        elif model_dtype == grad_accum_dtype:
            if model_dtype == torch.bfloat16:
                if self.pipeline_parallelism:
                    logger.warning(
                        "**** BF16 gradient accumulation is not safe numerically with large number of accumulation steps, proceed with caution *****"
                    )
                    return BFLOAT16
                else:
                    raise NotImplementedError(
                        "Bfloat16 wrapper must use a gradient accumulation type of fp32, enable ZeRO to use Bfloat16 gradient accumulation"
                    )
            if model_dtype == torch.float16:
                return FP16
            # else optimizer_wrapper = None
        elif model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32:
            return BFLOAT16
        else:
            raise NotImplementedError("unsupported mix of model dtype and gradient accumulation type")

        return None



    #######################################################################################
    #######################################      ETC      #################################
    #######################################################################################

    @staticmethod
    def is_map_style_dataset(obj):
        return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

    @staticmethod
    def is_iterable_style_dataset(obj):
        return isinstance(obj, torch.utils.data.IterableDataset)  # hasattr(obj, "__iter__") should work as well

    def dataloader_drop_last(self):
        return self._config.dataloader_drop_last

    def was_step_applied(self) -> bool:
        """Returns True if the latest ``step()`` produced in parameter updates.
        Note that a ``False`` return is not an error condition. Steps are frequently
        no-ops, such as between gradient accumulation boundaries or when overflows
        occur.
        Returns:
            bool: Whether the latest ``step()`` modified model parameters.
        """
        return self._step_applied

    def deepspeed_io(self,
                     dataset,
                     batch_size=None,
                     route=ROUTE_TRAIN,
                     pin_memory=True,
                     data_sampler=None,
                     collate_fn=None,
                     num_local_io_workers=None):
        if not (self.is_map_style_dataset(dataset) or self.is_iterable_style_dataset(dataset)):
            raise ValueError("Training data must be a torch Dataset")

        if batch_size is None:
            batch_size = self.train_micro_batch_size_per_gpu()

        if collate_fn is None:
            collate_fn = self.collate_fn

        # Currently we only use timer in train route
        deepspeed_io_timer = None
        if route == ROUTE_TRAIN:
            deepspeed_io_timer = self.tput_timer

        # If mpu is provided, forward world size and parallel rank to sampler.
        data_parallel_world_size = self.dp_world_size
        data_parallel_rank = self.global_rank
        if self.mpu is not None:
            data_parallel_world_size = self.mpu.get_data_parallel_world_size()
            data_parallel_rank = self.mpu.get_data_parallel_rank()

        if data_sampler is None and (route == ROUTE_PREDICT or route == ROUTE_EVAL):
            data_sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=data_parallel_world_size,
                rank=data_parallel_rank,
                shuffle=False,
            )

        deepspeed_dataloader_config = {}
        # if self.curriculum_learning_enabled():
        #     deepspeed_dataloader_config = {
        #         CURRICULUM_LEARNING: self.curriculum_learning_enabled(),
        #         DATA_EFFICIENCY: self.data_efficiency_config(),
        #         DATA_PARALLEL_GROUP: self.data_parallel_group,
        #         GRADIENT_ACCUMULATION_STEPS: self.gradient_accumulation_steps(),
        #         GLOBAL_RANK: self.global_rank,
        #         DATA_SAMPLING_NUM_WORKERS: self.data_sampling_config()[DATA_SAMPLING_NUM_WORKERS]
        #     }

        return DeepSpeedDataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn,
                                   local_rank=self.local_rank,
                                   tput_timer=deepspeed_io_timer,
                                   num_local_io_workers=num_local_io_workers,
                                   data_sampler=data_sampler,
                                   data_parallel_world_size=data_parallel_world_size,
                                   data_parallel_rank=data_parallel_rank,
                                   dataloader_drop_last=self.dataloader_drop_last(),
                                   deepspeed_dataloader_config=deepspeed_dataloader_config)

    def aio_config(self):
        return self._config.aio_config

    def train(self, mode=True):
        r""""""
        self.warn_unscaled_loss = True
        self.module.train(mode)

    def eval(self):
        r""""""
        self.warn_unscaled_loss = True
        self.module.train(False)