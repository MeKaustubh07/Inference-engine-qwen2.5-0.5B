"""The loading dock: a .safetensors file on disk -> named bf16 tensors in memory, zero-copy.

File layout:  [8-byte little-endian header length][JSON catalog][raw tensor bytes, back-to-back]
"""
import json
import mmap
import struct

import torch

# safetensors dtype string -> torch dtype (only the ones we expect to meet)
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class SafetensorsFile:
    def __init__(self, path: str):
        self._file = open(path, "rb")
        # Memory-map the file: the OS maps its bytes into our address space and
        # pulls pages from the SSD only when first touched. Nothing is read up front.
        # ACCESS_COPY = private copy-on-write mapping: readable, writable in our
        # process only, never written back to disk.
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)

        header_len = struct.unpack("<Q", self._mm[:8])[0]           # '<Q' = little-endian uint64
        header = json.loads(self._mm[8 : 8 + header_len])
        self._catalog = {name: info for name, info in header.items() if name != "__metadata__"}
        self._data_start = 8 + header_len                           # where the warehouse begins

    def tensor_names(self) -> list[str]:
        return list(self._catalog)

    def info(self, name: str) -> dict:
        if name not in self._catalog:
            raise KeyError(f"no tensor named {name!r}")
        return self._catalog[name]

    def get(self, name: str) -> torch.Tensor:
        """Return a tensor that VIEWS the file bytes directly. No copy, no dtype conversion."""
        t = self.info(name)
        dtype = DTYPES[t["dtype"]]
        start, end = t["data_offsets"]
        numel = (end - start) // torch.tensor([], dtype=dtype).element_size()
        flat = torch.frombuffer(self._mm, dtype=dtype, count=numel, offset=self._data_start + start)
        return flat.view(*t["shape"])

    def integrity_check(self) -> bool:
        """The catalog must account for every byte: last end offset == end of file."""
        max_end = max(t["data_offsets"][1] for t in self._catalog.values())
        return self._data_start + max_end == len(self._mm)
