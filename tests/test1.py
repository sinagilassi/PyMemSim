# import libs
from rich import print
import pymemsim as pml

# check version
print(pml.__version__)

# NOTE: create a HFM module
hfm = pml.hfm_module("HFM1", "co-current", True)
print(hfm)
print(type(hfm))
