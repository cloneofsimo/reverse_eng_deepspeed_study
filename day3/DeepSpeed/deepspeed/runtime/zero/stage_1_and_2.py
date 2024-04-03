# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.utils import (bwc_tensor_model_parallel_rank, empty_cache, see_memory_usage, inf,
                                     is_model_parallel_parameter, align_dense_tensors, all_gather_dp_groups)

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger

# from deepspeed.moe.utils import is_moe_param

from deepspeed.git_version_info import version

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.constants import (DS_VERSION, GROUP_PADDINGS, PARTITION_COUNT, LOSS_SCALER,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, BASE_OPTIMIZER_STATE,
                                            BASE_OPTIMIZER_STATE_STEP, CLIP_GRAD, ZERO_STAGE, PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params, lazy_init_hp_params_optimizer_state, map_to_flat_opt_states
from deepspeed.checkpoint import enable_universal_checkpoint

from deepspeed.utils import groups
# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TIMERS = [OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER]


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

class DeepSpeedZeroOptimizer(ZeROOptimizer):

    def __init__(self,
                 init_optimizer,
                 param_names,
                 timers,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 use_multi_rank_bucket_allreduce=True,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 mpu=None,
                 clip_grad=0.0,
                 gradient_accumulation_dtype=torch.float32,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):

        ## whether cpu offloading optimizer state or gradients
        if offload_optimizer_config is not None and offload_optimizer_config.device != OffloadDeviceEnum.none:
            self.cpu_offload = True
            self.cpu_offload_pin_memory = offload_optimizer_config.pin_memory
        else:
            self.cpu_offload = False
            self.cpu_offload_pin_memory = False

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {allgather_bucket_size}")
            logger.info(f"CPU Offload: {self.cpu_offload}")
            logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu
        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Accelerator is not detected, cannot perform low precision training (e.g., fp16, bf16).")
        self.optimizer = init_optimizer

        ## Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        ## ZeRO stage 1 (False) or 2 (True)
        self.partition_gradients = partition_grads
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

        self.timers = timers

        self.reduce_scatter = reduce_scatter
        self.overlap_comm = overlap_comm
        self.deepspeed_adam_offload = self.cpu_offload

        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'
        self.dp_process_group = dp_process_group
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        dp_size = dist.get_world_size(group=self.dp_process_group) #data parallel size for non-experts

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        self.contiguous_gradients = contiguous_gradients or self.cpu_offload

        self._global_grad_norm = 0.
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.overflow = False
        self.clip_grad = clip_grad
        self.postscale_gradients = postscale_gradients
        self.gradient_predivide_factor = gradient_predivide_factor

        self.micro_step_id = 0
        self.ignore_unused_parameters = ignore_unused_parameters
        self.extra_large_param_to_reduce = None

        self.communication_data_type = communication_data_type
        self.round_robin_gradients = round_robin_gradients

        ## who use fp16 as master copy?
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
            f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32."\
            f"Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}." \
            f"Either disable fp16_master_weights_and_gradients or enable {self.zero_stage_string} Offload with DeepSpeedCPUAdam."

        if self.reduce_scatter and self.partition_gradients:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, f"gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
            assert self.postscale_gradients, f"pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"

        # param flattened by groups
        self.bit16_groups = []
        self.bit16_groups_flat = []

        # param partitioned by data parallel degree
        # this will contain a list of equal sized tensors
        # each of which will be updated by a different process
        self.parallel_partitioned_bit16_groups = []

        self.single_partition_of_fp32_groups = [] # this process will update, a single 32-bit partition of the parallel partitioned parameters

        # param partition info
        self.params_not_in_partition = [] # These are the parameters in each group that will not be updated by this process directly
        self.params_in_partition = [] # These are the parameters that will be updated by this process directly

        # Offset from the first parameter in the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        self.first_offset = []

        self.partition_size = [] # number of elements per partition in each group
        self.nccl_start_alignment_factor = 2  # align nccl all-gather send buffers to 4-byte boundary, 4-byte alignment/sizeof(fp16) = 2
        assert (allgather_bucket_size % self.nccl_start_alignment_factor == 0), \
        f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor}"

        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.gradient_accumulation_dtype = gradient_accumulation_dtype

        self.use_separate_grad_accum = True if self.dtype != self.gradient_accumulation_dtype else False
        self.use_grad_accum_attribute = True if self.use_separate_grad_accum and not self.partition_gradients else False

        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []
        self.round_robin_bit16_meta = []


        ########################################################################################
        #################### for MP, model class should have this functions ####################
        ########################################################################################
        # https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed
        # https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/finetune_hf_llama
        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)


        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        self.groups_padding = []

        ## temporary for debugging (2gpu no MP)
        rank = dist.get_rank(self.real_dp_process_group[0])
        world_size = dist.get_world_size(group=self.real_dp_process_group[0])

        '''
        (Pdb) self.real_dp_process_group; dist.get_rank(self.real_dp_process_group[0]);
        [<torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd0964802b0>]
        0
        (Pdb) self.real_dp_process_group; dist.get_rank(self.real_dp_process_group[0]);
        [<torch.distributed.distributed_c10d.ProcessGroup object at 0x7facdba23430>]
        1
        '''
        # Tra()

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    trainable_parameters.append(param)
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer

            # Create temp CPU param copies, free accelerator tensors
            orig_group_numel = 0
            for param in self.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)

            empty_cache()
            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], 
                    dist.get_world_size(group=self.real_dp_process_group[i])
                )
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # Create meta tensors list, ordered according to round_robin_tensors
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            self.round_robin_bit16_meta.append(meta_tensors)

            # create flat buffer in CPU
            flattened_buffer = self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]),
                use_cpu_data=True)

            # free temp CPU params
            for param in self.bit16_groups[i]:
                del param.cpu_data

            # Move CPU flat tensor to the accelerator memory.
            self.bit16_groups_flat.append(flattened_buffer.to(get_accelerator().current_device_name()))
            del flattened_buffer

            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

            # Record padding required for alignment
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bit16_groups_flat[i].numel() - orig_group_numel
            else:
                padding = 0
            self.groups_padding.append(padding)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

            # set model bit16 weight to slices of flattened buffer
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            if not fp16_master_weights_and_gradients:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(self.device).clone().float().detach())
            else:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(self.device).clone().half().detach())

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            self.single_partition_of_fp32_groups[i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(self.round_robin_bit16_groups[i], partition_size, partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        # Tra()
        '''
        self.bit16_groups이 나뉘는 경우는 optimizer state가 여러개를 관리할 때 말곤 없는듯
        (Pdb) rank; world_size; len(self.bit16_groups)
        0
        2
        1
        (Pdb) rank; world_size; len(self.bit16_groups)
        1
        2
        1

        (Pdb) rank; world_size; partition_size; len(self.bit16_groups_flat[i]); self.bit16_groups_flat; self.parallel_partitioned_bit16_groups; param_group['params']
        1
        2
        4112000.0
        8224000
        [tensor([ 1.9297,  1.4844,  0.9023,  ...,  0.0271,  0.0708, -0.0801],
            device='cuda:1', dtype=torch.bfloat16)]
        [[tensor([ 1.9297,  1.4844,  0.9023,  ..., -0.0640, -0.0713,  0.0850],
            device='cuda:1', dtype=torch.bfloat16), tensor([-0.0874, -0.0093,  0.0513,  ...,  0.0271,  0.0708, -0.0801],
            device='cuda:1', dtype=torch.bfloat16)]]
        [tensor([-0.0874, -0.0093,  0.0513,  ...,  0.0271,  0.0708, -0.0801],
            device='cuda:1', requires_grad=True)]
        (Pdb) rank; world_size; partition_size; len(self.bit16_groups_flat[i]); self.bit16_groups_flat; self.parallel_partitioned_bit16_groups; param_group['params']
        0
        2
        4112000.0
        8224000
        [tensor([ 1.9297,  1.4844,  0.9023,  ...,  0.0271,  0.0708, -0.0801],
            device='cuda:0', dtype=torch.bfloat16)]
        [[tensor([ 1.9297,  1.4844,  0.9023,  ..., -0.0640, -0.0713,  0.0850],
            device='cuda:0', dtype=torch.bfloat16), tensor([-0.0874, -0.0093,  0.0513,  ...,  0.0271,  0.0708, -0.0801],
            device='cuda:0', dtype=torch.bfloat16)]]
        [tensor([ 1.9297,  1.4844,  0.9023,  ..., -0.0640, -0.0713,  0.0850],
            device='cuda:0', requires_grad=True)]
        '''

        self.reduce_bucket_size = int(reduce_bucket_size)
        self.use_multi_rank_bucket_allreduce = use_multi_rank_bucket_allreduce
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()

        #self.copy_grad_stream = get_accelerator().Stream()
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None

        # simplified param id
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel, device=self.device, dtype=self.dtype)
            if self.cpu_offload_pin_memory:
                self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(self.temp_grad_buffer_for_cpu_offload)
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel, device=get_accelerator().current_device_name(), dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])


        self.param_to_partition_ids = {} # mapping from parameter to partition that it belongs to
        self.is_partition_reduced = {} # stores if a partition has been reduced in this step
        self.remaining_grads_in_partition = {} # number of grads in partition that still need to be computed
        self.total_grads_in_partition = {} # total number of grads in partition
        self.is_grad_computed = {} # stores if a grad in a partition has been computed or not
        self.grad_partition_insertion_offset = {} # stores the offset at which a parameter gradient needs to be inserted in a partition
        self.grad_start_offset = {} # the offset in the gradient at which it must be inserted at the beginning of the partition
        self.averaged_gradients = {} # will store the averaged gradients required by this partition
        self.offload_gradient_dict = {} # For cpu_offload, will store the averaged gradients required by this partition
        self.first_param_index_in_partition = {} # store index of first parameter in each partition


        self.initialize_gradient_partitioning_data_structures() # initializes all data structures for implementing gradient partitioning
        # Tra()
        self.reset_partition_gradient_structures() # resets the data structure value for the next backward propagation
        # Tra()

        self._grad_acc_hooks = [] # creates backward hooks for gradient partitioning
        if self.partition_gradients or self.overlap_comm:
            self.create_reduce_and_remove_grad_hooks()

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        # we may have a way of fusing dynamic scale. Do not support for now
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic
        if self.dtype != torch.float16: # Only fp16 should use dynamic loss scaling
            assert self.loss_scaler.cur_scale == 1.0
            assert not self.dynamic_loss_scale

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

        self._link_all_hp_params()
        self._hp_optimizer_states_linked = False




    def _link_all_hp_params(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        if self.cpu_offload:
            self._get_offload_gradient_dict()

        for i, _ in enumerate(self.optimizer.param_groups):
            # Link bit16 and fp32 params in partition
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bit16_groups_flat[i].numel() // dp_world_size

            flat_hp_partition = self.single_partition_of_fp32_groups[i]

            link_hp_params(lp_param_list=self.bit16_groups[i],
                           flat_hp_partition=flat_hp_partition,
                           gradient_dict=self.averaged_gradients,
                           offload_gradient_dict=self.offload_gradient_dict,
                           use_offload=self.cpu_offload,
                           param_group_index=i,
                           partition_start=partition_id * partition_size,
                           partition_size=partition_size,
                           dp_group=self.real_dp_process_group[i])

    def _lazy_init_hp_params_optimizer_state(self):
        if not self._hp_optimizer_states_linked:
            for i, _ in enumerate(self.optimizer.param_groups):
                lazy_init_hp_params_optimizer_state(
                    self.bit16_groups[i], 
                    self.single_partition_of_fp32_groups[i],
                    self.optimizer.state
                )
            self._hp_optimizer_states_linked = True

    def _update_model_bit16_weights(self, group_index):
        updated_params = self.unflatten(self.bit16_groups_flat[group_index], self.round_robin_bit16_meta[group_index])
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data

        # set model fp16 weight to slices of reordered flattened buffer
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    def _round_robin_reorder(self, tensor_list, num_partitions):

        # disable round robin if need to debug something
        # return tensor_list, list(range(len(tensor_list)))

        partition_tensors = {}

        for i, tensor in enumerate(tensor_list):
            j = i % num_partitions
            if not j in partition_tensors:
                partition_tensors[j] = []
            partition_tensors[j].append((i, tensor))

        reordered_tensors = []
        reordered_indices = {}

        for partition_index in partition_tensors.keys():
            for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                reordered_indices[original_index] = len(reordered_tensors)
                reordered_tensors.append(tensor)

        return reordered_tensors, reordered_indices

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.bit16_groups):
            Tra()
            single_grad_partition = torch.zeros(int(self.partition_size[i]),
                                                dtype=self.single_partition_of_fp32_groups[i].dtype,
                                                device=self.device)
            self.single_partition_of_fp32_groups[i].grad = get_accelerator().pin_memory(single_grad_partition) if self.cpu_offload_pin_memory else single_grad_partition

        # Initialize the optimizer states with the flattened fp32 partition.
        # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
        # which do lazy initialization of the state at the first call to step.
        if isinstance(self.optimizer, torch.optim.Adagrad):
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)

        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  #class init

        return


    #############################################################################################
    ################################## Backward and Step methods ################################
    #############################################################################################
    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            self.ipg_buffer.append(buf_0)
            '''
            (Pdb) len(self.ipg_buffer); buf_0.size()
            1
            torch.Size([500000000])
            '''

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=get_accelerator().current_device_name())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        # Tra()
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            # just scale loss, perform backward (not unscale, it would be performed later)
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        # Only for Stage 1, Mode 2
        if self.use_grad_accum_attribute:
            self.fill_grad_accum_attribute()

    def _optimizer_step(self, group_no):
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]

        # Disabling this as the C++ side copy & synchronize is not working correctly
        #from deepspeed.ops.adam import DeepSpeedCPUAdam
        #if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
        #    self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        #else:
        #    self.optimizer.step()

        # Tra()
        '''
        (Pdb) self.optimizer.param_groups; group_no       
        [{'params': [tensor([ 1.9297,  1.4844,  0.9023,  ..., -0.0640, -0.0713,  0.0850],
            requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.95), 'eps': 1e-08, 'weight_decay': 0, 'bias_correction': True, 'amsgrad': False}]
        0

        (Pdb) self.optimizer.param_groups[0]['params'][0].dtype
        torch.float32
        '''
        self.optimizer.step()
        self.optimizer.param_groups = original_param_groups

        # We need to link optimizer state after the first step() call
        self._lazy_init_hp_params_optimizer_state()

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = -1

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            for timer in OPTIMIZER_TIMERS:
                self.timers(timer).start()
                self.timers(timer).stop()
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i) # group number

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                self.timers(OPTIMIZER_STEP_TIMER).stop()
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])
                    ).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(self.single_partition_of_fp32_groups[i].dtype)
                
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                # Step 3:- run the optimizer if no offloading
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i) # group number
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.timers.log(OPTIMIZER_TIMERS)
        see_memory_usage('After zero_optimizer step')

        return

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        invalid_grad_count = torch.zeros([1], dtype=torch.float, device=get_accelerator().current_device_name())
        for p in params:
            if p.grad is not None:
                invalid_grad_count += self._has_inf_or_nan(p.grad)
        return invalid_grad_count.bool()

    def has_overflow_partitioned_grads_serial(self):
        invalid_grad_count = torch.zeros([1], dtype=torch.float, device=get_accelerator().current_device_name())
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None:
                    invalid_grad_count += self._has_inf_or_nan(grad)
        return invalid_grad_count.bool()

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = get_accelerator().ByteTensor([overflow]) if self.cpu_offload else overflow.byte().to(
                get_accelerator().current_device_name())
            '''This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process'''
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            params = []
            for group in self.bit16_groups:
                for param in group:
                    params.append(param)
            overflow_gpu = self.has_overflow_serial(params).byte().to(get_accelerator().current_device_name())

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        float_x = x.float()
        nan = float_x.isnan()
        inf = float_x.isinf()
        inf_or_nan = nan.logical_or(inf)
        return inf_or_nan.float().max()

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)


    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################

    def reduce_gradients(self, pipeline_parallel=False):
        world_size = dist.get_world_size(self.dp_process_group)
        my_rank = dist.get_rank(self.dp_process_group)

        # with PP we must create ipg buffer, since backward is handled outside zero
        # Tra()
        if pipeline_parallel and self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            self.ipg_buffer.append(buf_0)
            self.ipg_index = 0

        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    grad_reduc = self.get_gradient_for_reduction(param)
                    if grad_reduc is not None:
                        self.reduce_ready_partitions_and_remove_grads(param, i)

        # reduce any pending grads in either hook/non-hook case
        self.overlapping_partition_gradients_reduce_epilogue()




    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):

        # Tra()
        for i, param_group in enumerate(self.round_robin_bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.total_grads_in_partition[i][partition_id] = 0

                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False

                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(i, param_group, partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        # if dist.get_rank() == 0:
        #    logger.info("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        if self.overlap_comm:
            if not get_accelerator().resolves_data_dependency():
                get_accelerator().synchronize()
            # It is safe to clear previously reduced grads of other partitions
            self._clear_previous_reduced_grads()

        if self.cpu_offload is False:
            for i, _ in enumerate(self.bit16_groups):

                if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
                    self.averaged_gradients[i] = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.gradient_accumulation_dtype,
                        device=get_accelerator().current_device_name(),
                        return_tensor_list=True)
                else:
                    avg_new = self.get_flat_partition(self.params_in_partition[i],
                                                      self.first_offset[i],
                                                      self.partition_size[i],
                                                      dtype=self.gradient_accumulation_dtype,
                                                      device=get_accelerator().current_device_name(),
                                                      return_tensor_list=True)

                    for accumulated_grad, new_avg_grad in zip(self.averaged_gradients[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)

        self._release_ipg_buffers()

        # No need to keep the gradients anymore.
        # All gradients required by the step
        # are in self.averaged_gradients
        self.zero_grad(set_to_none=True)
        see_memory_usage(f"End ipg_epilogue")

    def reduce_ipg_grads(self):
        if self.contiguous_gradients:
            if self.extra_large_param_to_reduce is not None:
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"

                _, _, param_id = self.params_in_ipg_bucket[0]
                assert self.get_param_id(self.extra_large_param_to_reduce) == param_id, "param in ipg bucket does not match extra-large param"
                
                extra_large_grad_reduc = self.get_gradient_for_reduction(self.extra_large_param_to_reduce)
                self.average_tensor(extra_large_grad_reduc.view(-1))

                self.extra_large_param_to_reduce = None
            else:
                self.average_tensor(self.ipg_buffer[self.ipg_index].narrow(0, 0, self.elements_in_ipg_bucket))
        else:
            self.buffered_reduce_fallback(None,
                                          self.grads_in_ipg_bucket,
                                          elements_per_buffer=self.elements_in_ipg_bucket)

        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            get_accelerator().synchronize()
            #            stream = self.copy_grad_stream
            stream = get_accelerator().current_stream()
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:

                assert self.params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"

                self.params_already_reduced[param_id] = True

                if self.partition_gradients:
                    if not self.is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.previous_reduced_grads is None:
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            self.clear_grad_attribute(param)

                    elif self.contiguous_gradients:
                        self.copy_grads_in_partition(param)

                else:  # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                        self.copy_grads_in_partition(param)

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def reset_partition_gradient_structures(self):
        for i, _ in enumerate(self.bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]

                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):

        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1

        partition_size = self.partition_size[i]

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for param in param_group:

            param_size = param.numel()
            param_id = self.get_param_id(param)

            if start_index <= current_index < end_index:
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif current_index < start_index < (current_index + param_size):
                assert (first_offset == 0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def fill_grad_accum_attribute(self):
        for group in self.bit16_groups:
            for param in group:
                if param.grad is not None:
                    if param.grad_accum is None:
                        param.grad_accum = param.grad.to(self.gradient_accumulation_dtype)
                    else:
                        param.grad_accum.add_(
                            param.grad.to(self.gradient_accumulation_dtype).view(param.grad_accum.shape))
                    param.grad = None

    def get_gradient_for_reduction(self, param):
        if self.use_grad_accum_attribute:
            return param.grad_accum.to(self.dtype) if param.grad_accum is not None else None
        else:
            return param.grad

    def get_param_gradient_attribute(self, param):
        return param.grad_accum if self.use_grad_accum_attribute else param.grad

    # Clear the tensor the reduction gradient attribute is pointing to
    def clear_grad_attribute(self, param):
        if self.use_grad_accum_attribute:
            param.grad_accum = None
        else:
            param.grad = None

    def create_reduce_and_remove_grad_hooks(self):
        self.grad_accs = []
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        self._grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
                        self.grad_accs.append(grad_acc)

                    wrapper(param, i)

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}"
        )

    # create a flat tensor aligned at the alignment boundary
    def flatten_dense_tensors_aligned(self, tensor_list, alignment, use_cpu_data=False):
        tensor_list = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list
        return self.flatten(align_dense_tensors(tensor_list, alignment))




    #################################################################################
    ######################### Independent Partition Gradient ########################
    #################################################################################

    def average_tensor(self, tensor):
        if self.overlap_comm:
            stream = self.reduction_stream
            if not get_accelerator().resolves_data_dependency():
                stream.wait_stream(get_accelerator().current_stream())
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id, prev_process_group = -1, None

            process_group = self.dp_process_group
            # count = 0
            for i, param, param_id in self.params_in_ipg_bucket:

                process_group = self.dp_process_group
                grad_reduc = self.get_gradient_for_reduction(param)

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids
                            ]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]

                # Get all partition ids + their offsets
                partition_ids_w_offsets = []

                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # if dist.get_rank() == 0 and count < 100:
                    #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                    # count += 1

                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                        
                    else:
                        # Set numel to next partition's offset
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    if partition_id == prev_id and process_group == prev_process_group:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id, prev_process_group = partition_id, process_group

            buckets = {}
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                    dst, real_dp_process_group[i])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                if self.use_multi_rank_bucket_allreduce:
                    buckets[bucket_key].append((dst, grad_slice))
                else:
                    buckets[bucket_key].append(grad_slice)

            for bucket_key in buckets:
                if self.use_multi_rank_bucket_allreduce:
                    self.allreduce_and_scatter(buckets[bucket_key],
                                               numel_per_bucket=self.reduce_bucket_size,
                                               divide=False,
                                               process_group=bucket_key)
                else:
                    dst, process_group = bucket_key
                    self.allreduce_no_retain(buckets[bucket_key],
                                             numel_per_bucket=self.reduce_bucket_size,
                                             rank=dst,
                                             divide=False,
                                             process_group=process_group)

    def gradient_reduction_w_predivide(self, tensor):

        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        tensor_to_allreduce = tensor

        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            if self.gradient_predivide_factor != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor /
                                         (dp_world_size / float(self.sequence_parallel_size)))
        else:
            tensor_to_allreduce.div_(dp_world_size / float(self.sequence_parallel_size))
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy_with_multiple_ranks(self,
                                               small_bucket,
                                               log=None,
                                               divide=True,
                                               process_group=None,
                                               bucket_ranks=None):
        process_group = self.dp_process_group if process_group is None else process_group
        allreduced = self.allreduce_bucket(small_bucket, log=log, divide=divide, process_group=process_group)
        for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
            if dist.get_rank(group=process_group) == bucket_rank:
                buf.copy_(synced)

    def allreduce_and_scatter(self, bucket, numel_per_bucket=500000000, log=None, divide=True, process_group=None):
        small_bucket = []
        small_bucket_ranks = []
        numel = 0
        allreduce_sizes = []

        for i, bucket_elem in enumerate(bucket):
            rank, tensor = bucket_elem
            small_bucket.append(tensor)
            small_bucket_ranks.append(rank)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                            log=None,
                                                            divide=divide,
                                                            process_group=process_group,
                                                            bucket_ranks=small_bucket_ranks)
                small_bucket = []
                small_bucket_ranks = []
                numel = 0

        if len(small_bucket) > 0:
            self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                        log=None,
                                                        divide=divide,
                                                        process_group=process_group,
                                                        bucket_ranks=small_bucket_ranks)

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):

        grad_reduc = self.get_gradient_for_reduction(param)
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
            self.reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        if self.contiguous_gradients:
            if param.numel() > self.reduce_bucket_size:
                self.extra_large_param_to_reduce = param
            else:
                # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
                new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
                new_grad_tensor.copy_(grad_reduc.view(-1))
                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        self.elements_in_ipg_bucket += param.numel()

        assert grad_reduc is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        self.grads_in_ipg_bucket.append(grad_reduc)
        self.params_in_ipg_bucket.append((i, param, param_id))

        self.report_ipg_memory_usage("End ipg_remove_grads", 0)




    ##############################################################################
    ############################# CPU Offload Methods#############################
    ##############################################################################

    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        current_offset = 0

        for i, tensor in enumerate(tensor_list):
            param_id = self.get_param_id(tensor)
            param_start_offset = 0

            num_elements = tensor.numel()

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                param_start_offset = first_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_offset):
                num_elements = partition_size - current_offset

            self.grad_position[param_id] = [
                int(group_id), int(param_start_offset),
                int(current_offset), int(num_elements)
            ]
            current_offset += num_elements

    def update_overflow_tracker_for_param_grad(self, param):
        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is not None and self._has_inf_or_nan(grad_accum.data):
            self.local_overflow = True

    def _get_offload_gradient_dict(self):
        for param_group_index, _ in enumerate(self.optimizer.param_groups):
            self.offload_gradient_dict[param_group_index] = []
            for lp_param in self.params_in_partition[param_group_index]:
                param_id = self.get_param_id(lp_param)
                [_, _, dest_offset, num_elements] = self.grad_position[param_id]
                dest_tensor = self.single_partition_of_fp32_groups[param_group_index].grad.view(-1).narrow(
                    0, dest_offset, num_elements)
                self.offload_gradient_dict[param_group_index].append(dest_tensor)

    def set_norm_for_param_grad(self, param):
        param_id = self.get_param_id(param)
        grad_accum = self.get_param_gradient_attribute(param)
        accumulated_grad = self.accumulated_grads_in_cpu[
            param_id] if self.gradient_accumulation_steps > 1 else grad_accum

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is None:
            accumulated_grad = param.grad
        else:
            accumulated_grad = grad_accum

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
                # so they have no norm_for_param_grads
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item()**2
                else:
                    # As unused parameters in modules may not be expected sometimes,
                    # add an explicit error msg when it occurred and an option to
                    # avoid the error
                    assert self.ignore_unused_parameters, """
                        This assert indicates that your module has parameters that
                        were not used in producing loss.
                        You can avoid this assert by
                        (1) enable ignore_unused_parameters option in zero_optimization config;
                        (2) making sure all trainable parameters and `forward` function
                            outputs participate in calculating loss.
                    """

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm




    #######################################################################################################
    ################################## copy gradients in partition (async) ################################
    #######################################################################################################

    def copy_grads_in_partition(self, param):
        if self.cpu_offload:

            if self.gradient_accumulation_steps > 1:
                self.async_accumulate_grad_in_cpu_via_gpu(param)

            if self.is_gradient_accumulation_boundary:
                self.set_norm_for_param_grad_in_gpu(param)
                self.update_overflow_tracker_for_param_grad(param)
                self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)

            return
        #print(f"ID {self.get_param_id(param)} grad norm {param.grad.norm()}")
        if self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()

            see_memory_usage(f"before copying {total_size} gradients into partition")
            self.grads_in_partition = torch.empty(int(total_size),
                                                  dtype=self.dtype,
                                                  device=get_accelerator().current_device_name())
            see_memory_usage(f"after copying {total_size} gradients into partition")

        grad_reduc = self.get_gradient_for_reduction(param)
        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())
        new_grad_tensor.copy_(grad_reduc.view(-1))
        grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)
        #print(f"Grad norm after copy to contiguous_buffer {param.grad.data.norm()}")
        self.grads_in_partition_offset += param.numel()

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is None:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        else:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        if not self.fp16_master_weights_and_gradients:
            src_tensor = src_tensor.float()

        dest_tensor.copy_(src_tensor, non_blocking=True)
        param.grad = None  #offload only

    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # copy to a preexisiting buffer to avoid memory allocation penalty
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(0, 0, param.numel())

        #buffer for storing gradients for this parameter in CPU
        def buffer_to_accumulate_to_in_cpu():
            if not self.fp16_master_weights_and_gradients:
                buffer = torch.zeros(param.numel(), dtype=param.dtype, device=self.device)
                return get_accelerator().pin_memory(buffer) if self.cpu_offload_pin_memory else buffer
            else:
                return self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        #accumulate gradients into param.grad_accum or parts of it that belongs to this partition
        def accumulate_gradients():
            grad_accum = self.get_param_gradient_attribute(param)
            if not self.fp16_master_weights_and_gradients:
                dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)
                grad_accum.data.view(-1).add_(dest_buffer)
            else:
                dest_buffer.narrow(0, source_offset,
                                   num_elements).copy_(self.accumulated_grads_in_cpu[param_id].view(-1),
                                                       non_blocking=True)
                grad_accum.data.view(-1).narrow(0, source_offset,
                                                num_elements).add_(dest_buffer.narrow(0, source_offset, num_elements))

        #move accumulated gradients back to CPU
        def copy_gradients_to_cpu():
            grad_accum = self.get_param_gradient_attribute(param)
            if not self.fp16_master_weights_and_gradients:
                self.accumulated_grads_in_cpu[param_id].data.copy_(grad_accum.data.view(-1), non_blocking=True)
            else:
                self.accumulated_grads_in_cpu[param_id].data.copy_(grad_accum.data.view(-1).narrow(
                    0, source_offset, num_elements),
                                                                   non_blocking=True)

        if param_id not in self.accumulated_grads_in_cpu:
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()

        if self.micro_step_id > 0:
            accumulate_gradients()

        # at the boundary we will send 32bit directly
        if not self.is_gradient_accumulation_boundary:
            copy_gradients_to_cpu()



    #############################################################################################
    ################################## Reduction Related Methods ################################
    #############################################################################################

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):

        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None  # dead code

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = self.flatten(tensors)
        def print_func():
            logger.info(flatten_tensor.contiguous().view(-1).narrow(0, start, n))
        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):

        def get_reducible_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(total_elements - start,
                               self.partition_size[i] - self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0, int(start), int(num_elements))
            else:
                if num_elements == total_elements:
                    return grad.clone()
                else:
                    return grad.clone().contiguous().view(-1).narrow(0, int(start), int(num_elements))

        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducible_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def sequential_execution(self, function, message, group=None):
        if group is None:
            group = self.dp_process_group
        if dist.get_rank(group=group) == 0:
            logger.info(message)
        for id in range(dist.get_world_size(group=group)):
            if id == dist.get_rank(group=group):
                function()
            dist.barrier(group=group)

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zero_like(param)

    def allreduce_bucket(self, bucket, rank=None, log=None, divide=True, process_group=None):
        tensor = self.flatten(bucket)

        process_group = self.dp_process_group if process_group is None else process_group

        tensor_to_allreduce = tensor

        if pg_correctness_test or self.sequence_parallel_size > 1:
            communication_data_type = torch.float32
        else:
            communication_data_type = self.communication_data_type

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        if divide:
            tensor_to_allreduce.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=process_group)
        else:
            global_rank = dist.get_global_rank(process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    def _clear_previous_reduced_grads(self):
        if self.previous_reduced_grads is not None:
            for param in self.previous_reduced_grads:
                self.clear_grad_attribute(param)
            self.previous_reduced_grads = None

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None, divide=True, process_group=None):
        process_group = self.dp_process_group if process_group is None else process_group
        if self.overlap_comm:
            if not get_accelerator().resolves_data_dependency():
                get_accelerator().synchronize()
            # It is safe to clear the previously reduced grads of other partitions
            self._clear_previous_reduced_grads()
            stream = self.reduction_stream
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            allreduced = self.allreduce_bucket(
                small_bucket,
                rank=rank,
                log=log,
                divide=divide,
                process_group=process_group,
            )
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(
        self,
        bucket,
        numel_per_bucket=500000000,
        rank=None,
        log=None,
        divide=True,
        process_group=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None, divide=divide, process_group=process_group)
                small_bucket = []
                numel = 0

        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log, divide=divide, process_group=process_group)

    # allows using reduction of gradients instead of using all_reduce
    def buffered_reduce_fallback(self, rank, grads, elements_per_buffer=500000000, log=None):
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer, rank=rank, log=log)


    #############################################################################
    #############################################################################
    #############################################################################
    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor, group_id):
        partitions = []

        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        # dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)

            elif current_index < start_index < (current_index + tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        # zero all pointers to grad tensors
        for group in self.bit16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None  # epilogue and in step
                    p.grad_accum = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass
        else:
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        all_norms = []
        if norm_type == inf:
            for g in gradients:
                all_norms.append(g.data.abs().max().float())
            total_norm = torch.stack(all_norms).max()
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.MAX)
        else:
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    all_norms.append(
                        torch.norm(g.data.double().detach(), norm_type).to(get_accelerator().current_device_name()))
            if len(all_norms) > 0:
                total_norm = torch.stack(all_norms).square().sum().float()
            else:
                total_norm = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            # Sum across all model parallel Device.
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)

            total_norm = total_norm.pow(1. / norm_type)

        norm_is_inf = total_norm.isinf()
        norm_is_nan = total_norm.isnan()
        inf_or_nan = norm_is_nan.logical_or(norm_is_inf)

        err = torch.tensor(-1.0, device=self.device, dtype=torch.float)
        total_norm = inf_or_nan * err + inf_or_nan.logical_not() * total_norm
        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype, device, return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0

        for i, tensor in enumerate(tensor_list):
            grad_accum = self.get_param_gradient_attribute(tensor)
            if grad_accum is None:
                grad_accum = torch.zeros_like(tensor, dtype=dtype)

            tensor = grad_accum
            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(torch.zeros(int(partition_size - current_size), dtype=dtype, device=device))

        if return_tensor_list:
            return flat_tensor_list

        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None  # in step
            p.grad_accum = None

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def scaled_global_norm(self, norm_type=2):
        assert norm_type == 2, "only L2 norm supported"
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            if self.cpu_offload:
                # complete complete_grad_norm_calculation_for_cpu_offload return python float, moving back to
                # torch.tensor as else statement returns tensor as well
                norm = torch.tensor(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]),
                                    device=self.device)
                norm_groups.append(norm)
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))

        # calculating L2 norm
        return torch.norm(torch.stack(norm_groups), p=norm_type)

    def get_bit16_param_group(self, group_no):
        bit16_partitions = self.parallel_partitioned_bit16_groups[group_no]
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_no])
        return [bit16_partitions[dist.get_rank(group=self.real_dp_process_group[group_no])]]
