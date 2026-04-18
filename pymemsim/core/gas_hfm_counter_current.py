# import libs

# locals
from .gas_hfm import GasHFM


class GasHFMCounterCurrent(GasHFM):
    """
    Gas HFM counter-current flow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
