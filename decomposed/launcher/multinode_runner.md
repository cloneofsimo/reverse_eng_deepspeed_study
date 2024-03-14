

### Summary



* `MultiNodeRunner`: Abstract base class for multi-node runners. Defines the interface for launching distributed training across multiple nodes. Importance: **[High]**
* `PDSHRunner`: Implements the `MultiNodeRunner` for the pdsh backend, which uses pdsh for parallel command execution on multiple nodes. Importance: **[Medium]**
* `OpenMPIRunner`: Implements the `MultiNodeRunner` for the OpenMPI backend, using mpirun for launching distributed training. Importance: **[Medium]**
* `MPICHRunner`: Implements the `MultiNodeRunner` for the MPICH backend, using mpirun for launching distributed training. Importance: **[Medium]**
* `IMPIRunner`: Implements the `MultiNodeRunner` for the Intel MPI (IMPI) backend, using mpirun for launching distributed training. Importance: **[Medium]**

### Highlights



1. **Inheritance and Abstract Base Class (ABC)**: The code defines a base class `MultiNodeRunner` which is an abstract class (using `ABC` and `abstractmethod`). This class provides a common interface for different distributed computing backends (e.g., `PDSHRunner`, `OpenMPIRunner`, `MPICHRunner`, `IMPIRunner`, `SlurmRunner`, `MVAPICHRunner`). Each subclass implements the `backend_exists` and `get_cmd` methods specific to their respective backend.
2. **Command Execution**: The `get_cmd` method in each subclass constructs the command to execute the user script on the distributed nodes. It handles environment variables, arguments, and backend-specific flags and configurations.
3. **Argument Handling**: The `__init__` method initializes the runner with arguments and provides utility methods like `parse_user_args` to process user-provided arguments. The `add_export` method allows adding environment variables for the command execution.
4. **Backend Detection**: Each subclass has a `backend_exists` method that checks if the corresponding backend (e.g., `pdsh`, `ompi_info`, `mpirun`) is installed and available on the system.
5. **Error Handling and Validation**: Some subclasses have additional validation methods like `validate_args` to ensure the provided arguments are compatible with the specific backend.

### Pythonic Pseudocode

```python
# Define a base class for multi-node runners
class MultiNodeRunner(ABC):
    def __init__(self, args, world_info_base64):
        self.args = args
        self.validate_args()
        self.user_arguments = self.parse_user_args()
        self.user_script = args.user_script
        self.world_info_base64 = world_info_base64
        self.exports = {}

    # Abstract method to check if the backend exists
    @abstractmethod
    def backend_exists(self):
        pass

    # Abstract method to get the command to execute on a node
    @abstractmethod
    def get_cmd(self, environment, active_resources):
        pass

    # Add an environment variable to the runner
    def add_export(self, key, var):
        self.exports[key.strip()] = var.strip()

    # Parse user arguments
    def parse_user_args(self):
        return self.args.user_args

    # Return the backend name
    @property
    def name(self):
        return self.__class__.__name__

    # Validate the runner's arguments
    def validate_args(self):
        pass


# Define a runner for Parallel Distributed Shell (pdsh)
class PDSHRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64):
        super().__init__(args, world_info_base64)

    # Check if pdsh is installed
    def backend_exists(self):
        return is_tool_installed('pdsh')

    # Parse user arguments for pdsh compatibility
    def parse_user_args(self):
        return process_user_args(self.args.user_args)

    # Return the pdsh-specific command
    def get_cmd(self, environment, active_resources):
        # Set environment variables and prepare pdsh command
        prepare_pdsh_environment(environment, self.args)
        prepare_deepspeed_launch(environment, self.exports, self.args)
        return build_pdsh_command(environment, active_resources, self.args, self.user_script, self.user_arguments)


# Define a runner for OpenMPI
class OpenMPIRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    # Check if OpenMPI is installed
    def backend_exists(self):
        return is_tool_installed('ompi_info')

    # Validate OpenMPI-specific arguments
    def validate_args(self):
        super().validate_args()
        validate_openmpi_args(self.args)

    # Return the OpenMPI-specific command
    def get_cmd(self, environment, active_resources):
        # Prepare mpirun command and environment
        prepare_mpirun_command(environment, self.args, self.resource_pool)
        return build_mpirun_command(environment, self.exports, self.args, self.user_script, self.user_arguments)


# Define a runner for MPICH
class MPICHRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    # Check if MPICH is installed
    def backend_exists(self):
        return is_tool_installed('mpirun')  # mpich_info

    # Validate MPICH-specific arguments
    def validate_args(self):
        super().validate_args()
        validate_mpich_args(self.args)

    # Return the MPICH-specific command
    def get_cmd(self, environment, active_resources):
        # Prepare mpirun command and environment
        prepare_mpich_command(environment, self.args, self.resource_pool)
        return build_mpich_command(environment, self.exports, self.args, self.user_script, self.user_arguments)


# Define a runner for Intel MPI (IMPI)
class IMPIRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    # Check if IMPI is installed
    def backend_exists(self):
        return is_tool_installed('mpirun')  # mpich_info

    # Validate IMPI-specific arguments
    def validate_args(self):
        super().validate_args()
        validate_impi_args(self.args)

    # Return the IMPI-specific command
    def get_cmd(self, environment, active_resources):
        # Prepare mpirun command and environment
        prepare_impi_command(environment, self.args, self.resource_pool)
        return build_impi_command(environment, self.exports, self.args, self.user_script, self.user_arguments)


# Define a runner for Slurm
class SlurmRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    # Check if Slurm is installed
    def backend_exists(self):
        return is_tool_installed('sinfo')

    # Return the Slurm-specific command
    def get_cmd(self, environment, active_resources):
        # Prepare srun command and environment
        prepare_slurm_command(environment, self.args, self.resource_pool)
        return build_slurm_command(environment, self.exports, self.args, self.user_script, self.user_arguments)


# Define a runner for MVAPICH
class MVAPICHRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool
        self.configure_mvapich_environment()

    # Check if MVAPICH is installed
    def backend_exists(self):
        return is_tool_installed('mpiname') and is_mvapich2_gdr_installed()

    # Configure MVAPICH environment
    def configure_mvapich_environment(self):
        disable_cma()
        enable_debug_backtrace()
        enable_cuda_aware_communication()
        enable_dl_support()
        enable_mpi_thread_multiple()

    # Return the MVAPICH-specific command
    def get_cmd(self, environment, active_resources):
        # Prepare mpirun command and environment
        prepare_mvapich_command(environment, self.args, self.resource_pool)
        return build_mvapich_command(environment, self.exports, self.args, self.user_script, self.user_arguments)
```


### import Relationships

Imports found:
import os
import sys
import shutil
import subprocess
import warnings
from shlex import split
from abc import ABC, abstractmethod
from deepspeed.accelerator import get_accelerator
from ..utils import logger, get_numactl_cmd
from .constants import PDSH_MAX_FAN_OUT, MVAPICH_TMP_HOSTFILE