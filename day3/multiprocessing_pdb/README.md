## Install

```
cd /path/to/dir/multiprocessing_pdb &&\
pip install -e .
```

## Usage

```
from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

def dummy_code_block(...):
    Tra()
```