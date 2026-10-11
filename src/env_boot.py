"""Import-order guard. Must be the first import of every entry point.

On Windows, pandas 3.0 imports pyarrow eagerly, and pyarrow loads its own
Intel OpenMP runtime (libiomp5md.dll). If that happens before torch is
imported, torch's c10.dll fails to load with WinError 1114. Importing torch
first claims the OpenMP runtime and both libraries then coexist.
"""
import torch as _torch  # noqa: F401

__all__ = []
