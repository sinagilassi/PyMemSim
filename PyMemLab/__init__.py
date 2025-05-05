# settings
from .config import __package__, __version__, __description__, __author__
from .app import hfm_module
from .docs import HFM

__all__ = [
    "__package__",
    "__version__",
    "__description__",
    "__author__",
    "hfm_module",
    "HFM",
]
