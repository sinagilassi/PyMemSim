# import libs

# locals
from .gas_hfm import GasHFM


class GasHFMCoCurrent(GasHFM):
    """
    Gas HFM for co-current flow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
