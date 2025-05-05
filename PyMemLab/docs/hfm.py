# import libs
from typing import Dict, List, Literal, Optional, Tuple, Union


class HFM:
    """
    Hollow Fiber Membrane (HFM) class for PyMemLab.
    """
    # NOTE: variables
    # unit name
    unit_name: str
    # unit type
    unit_type: str
    # energy analysis
    energy_analysis: bool

    def __init__(self,
                 unit_name: str,
                 unit_type: str,
                 energy_analysis: bool) -> None:
        """
        Initialize the Unit class.
        """
        # sets
        self.__unit_name = unit_name
        self.__unit_type = unit_type
        self.__energy_analysis = energy_analysis

    def __str__(self):
        """
        String representation of the Unit class.
        """
        return "Unit class for PyMemLab"

    @property
    def unit_name(self) -> str:
        """
        Get the unit name.
        """
        return self.__unit_name

    @unit_name.setter
    def unit_name(self, unit_name: str) -> None:
        """
        Set the unit name.
        """
        self.__unit_name = unit_name

    @property
    def unit_type(self) -> str:
        """
        Get the unit type.
        """
        return self.__unit_type

    @unit_type.setter
    def unit_type(self, unit_type: str) -> None:
        """
        Set the unit type.
        """
        self.__unit_type = unit_type

    @property
    def energy_analysis(self) -> bool:
        """
        Get the energy analysis.
        """
        return self.__energy_analysis
