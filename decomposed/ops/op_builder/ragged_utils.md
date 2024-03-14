

### Summary

<|im_end|>

* `RaggedUtilsBuilder`: A class that extends `CUDAOpBuilder` for building Ragged Operations. Importance: **[High]**
* `is_compatible`: Checks if the system is compatible for building Ragged Operations, considering the presence of PyTorch and compatible CUDA versions. Importance: **[High]**
* `absolute_name`: Returns the fully qualified name for the Ragged Operations module. Importance: **[Medium]**
* `filter_ccs`: Filters out compute capabilities that are not compatible with NVIDIA Inference (Pascal and newer architectures). Importance: **[Medium]**
* `get_prefix`: Determines the correct path prefix for the DeepSpeed sources. Importance: **[Low]**

### Highlights

<|im_end|>

1. **Inheritance and Class Definition**: The code defines a class `RaggedUtilsBuilder` which inherits from `CUDAOpBuilder`. This class is responsible for building operations related to ragged tensors for the DeepSpeed library.
2. **Module Import**: The code imports necessary modules, including `os` and `torch`, for file operations and checking CUDA compatibility with PyTorch.
3. **Compatibility Check**: The `is_compatible` method checks if the system has the required dependencies, specifically `torch` and a compatible CUDA version, for building the ragged operations. It also logs warnings if the system is not compatible.
4. **Filtering Compute Capabilities**: The `filter_ccs` method filters out compute capabilities that are less than 6, as NVIDIA inference is only supported on Pascal and newer architectures.
5. **File Paths and Sources**: The `get_prefix`, `sources`, `extra_ldflags`, and `include_paths` methods are responsible for managing file paths and specifying source files, include directories, and linker flags needed for building the ragged operations.

### Pythonic Pseudocode

```python
# Define a class for building RaggedUtils with CUDAOpBuilder as the base class
class RaggedUtilsBuilder(CUDAOpBuilder):
    # Class constants
    BUILD_VAR = "DS_BUILD_RAGGED_OPS"
    NAME = "ragged_ops"

    # Constructor
    def __init__(self, name=None):
        # Set the class name if not provided
        name = self.NAME if name is None else name
        # Initialize the base class with the provided name
        super().__init__(name=name)

    # Get the fully qualified class name
    def absolute_name(self):
        return f'deepspeed.inference.v2.{self.NAME}'

    # Check if the system is compatible for building RaggedUtils
    def is_compatible(self, verbose=True):
        # Check if torch is installed
        try:
            import torch
        except ImportError:
            # Print a warning and return False if torch is not installed
            self.warning("Please install torch if trying to pre-compile inference kernels")
            return False

        # Check CUDA compatibility
        cuda_okay = True
        if not self.is_rocm_pytorch() and torch.cuda.is_available():
            # Get installed and torch CUDA versions
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            # Get CUDA device properties (ignore CUDA-related errors)
            try:
                cuda_capability = torch.cuda.get_device_properties(0).major
            except:
                pass
            else:
                # Check compatibility based on CUDA capability and versions
                if cuda_capability < 6:
                    self.warning("NVIDIA Inference is only supported on Pascal and newer architectures")
                    cuda_okay = False
                if cuda_capability >= 8 and (torch_cuda_major < 11 or sys_cuda_major < 11):
                    self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False

        # Return compatibility status
        return super().is_compatible(verbose) and cuda_okay

    # Filter compute capabilities (CCs) based on a minimum threshold
    def filter_ccs(self, ccs):
        # Create lists for retained and pruned CCs
        ccs_retained = []
        ccs_pruned = []

        # Iterate over CCs
        for cc in ccs:
            # Check if CC is greater than or equal to 6
            if int(cc[0]) >= 6:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)

        # Print a warning if CCs are pruned
        if ccs_pruned:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")

        # Return retained CCs
        return ccs_retained

    # Get the path prefix for source files
    def get_prefix(self):
        # Check if the "deepspeed" directory exists
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    # Get the list of source files
    def sources(self):
        # Define the source file paths
        sources = [
            "inference/v2/ragged/csrc/fast_host_buffer.cu",
            "inference/v2/ragged/csrc/ragged_ops.cpp",
        ]

        # Add the prefix to each source file path
        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]

        # Return the list of source files
        return sources

    # Get any extra linker flags
    def extra_ldflags(self):
        # Return an empty list
        return []

    # Get the list of include paths
    def include_paths(self):
        # Define the include directories
        include_dirs = ['inference/v2/ragged/includes', 'inference/v2/kernels/includes']
        # Add the prefix to each include directory
        prefix = self.get_prefix()
        includes = [os.path.join(prefix, include_dir) for include_dir in include_dirs]

        # Return the list of include paths
        return includes
```


### import Relationships

Imports found:
import os
from .builder import CUDAOpBuilder, installed_cuda_version