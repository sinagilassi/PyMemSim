# import libs
import logging
import numpy as np
from typing import Union
# locals
from ..models import GasModel


class ThermoCalc:
    # SECTION: EOS related methods
    # ! Calculate total pressure using ideal gas law
    def calc_tot_pressure(
            self,
            n_total: float,
            temperature: float,
            reactor_volume_value: float,
            R: float,
            gas_model: GasModel
    ) -> float:
        """
        Total pressure [Pa].
        Default: ideal gas
            P = N_total * R * T / V

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        reactor_volume_value : float
            Volume of the reactor [m3].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Total pressure of the gas in the reactor [Pa].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / float(reactor_volume_value)

    # ! Calculate volume
    def calc_gas_volume(
        self,
        n_total: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: GasModel
    ) -> float:
        """
        Calculate the volume of the gas in the reactor using the ideal gas law.
            V = N_total * R * T / P

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        pressure : float
            Pressure of the gas in the reactor [Pa].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Volume of the gas in the reactor [m3].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / pressure

    def calc_liquid_volume(
            self,
            n: np.ndarray,
            molecular_weights: np.ndarray,
            density: np.ndarray
    ) -> float:
        """
        Calculate the volume of the liquid in the reactor using the formula:
            V = sigma_i (n_i * MW_i) / density_i

        Parameters
        ----------
        n : np.ndarray
            An array of moles of each component in the liquid phase.
        molecular_weights : np.ndarray
            An array of molecular weights for each component in the liquid phase [g/mol].
        density : np.ndarray
            An array of densities for each component in the liquid phase [g/m3].
        """
        # calculate volume for each component
        # n [mol], MW [g/mol], density [g/m3] => volume [m3]
        volumes = n * molecular_weights / density

        # total volume is the sum of the volumes of each component
        total_volume = np.sum(volumes)

        return total_volume

    def calc_molar_flow_rate_from_volumetric_flow_rate(
        self,
        volumetric_flow_rate: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: GasModel
    ) -> float:
        """
        Calculate the molar flow rate from the volumetric flow rate using the ideal gas law.
            F = V_dot * P / (R * T)

        Parameters
        ----------
        volumetric_flow_rate : float
            Volumetric flow rate of the gas [m3/s].
        temperature : float
            Temperature of the gas in the reactor [K].
        pressure : float
            Pressure of the gas in the reactor [Pa].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Molar flow rate of the gas [mol/s].
        """
        if gas_model == "real":
            return 0

        # ideal gas model
        return volumetric_flow_rate * pressure / (R * temperature)

    def calc_molar_flow_rate_from_total_concentration(
        self,
        total_concentration: float,
        volumetric_flow_rate: float,
    ) -> float:
        """
        Calculate the molar flow rate from the total concentration.
            F = C_total * V_dot

        Parameters
        ----------
        total_concentration : float
            Total concentration in the reactor [mol/m3].
        volumetric_flow_rate : float
            Volumetric flow rate of the gas [m3/s].

        Returns
        -------
        float
            Molar flow rate of the gas [mol/s].
        """
        return total_concentration * volumetric_flow_rate

    def calc_gas_volumetric_flow_rate(
            self,
            molar_flow_rate: float,
            temperature: float,
            pressure: float,
            R: float,
            gas_model: GasModel
    ):
        """
        Calculate the volumetric flow rate of the gas using the ideal gas law or real gas model.
            V_dot = F * R * T / P

        Parameters
        ----------
        molar_flow_rate : float
            Molar flow rate of the gas [mol/s].
        temperature : float
            Temperature of the gas in the reactor [K].
        pressure : float
            Pressure of the gas in the reactor [Pa].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Volumetric flow rate of the gas [m3/s].
        """
        if gas_model == "real":
            return 0.0

        # ideal gas model
        return molar_flow_rate * R * temperature / pressure

    def calc_liquid_volumetric_flow_rate(
            self,
            molar_flow_rates: np.ndarray,
            molecular_weights: np.ndarray,
            density: np.ndarray
    ) -> float:
        """
        Calculate the volumetric flow rate of the liquid using the formula:
            V_dot = sigma_i (F_i * MW_i) / density_i

        Parameters
        ----------
        molar_flow_rates : np.ndarray
            An array of molar flow rates for each component in the liquid phase [mol/s].
        molecular_weights : np.ndarray
            An array of molecular weights for each component in the liquid phase [g/mol].
        density : np.ndarray
            An array of densities for each component in the liquid phase [g/m3].

        Returns
        -------
        float
            Volumetric flow rate of the liquid [m3/s].
        """
        # calculate volumetric flow rate for each component
        # F [mol/s], MW [g/mol], density [g/m3] => volumetric flow rate [m3/s]
        volumetric_flow_rates = molar_flow_rates * molecular_weights / density

        # total volumetric flow rate is the sum of the volumetric flow rates of each component
        total_volumetric_flow_rate = np.sum(volumetric_flow_rates)

        return total_volumetric_flow_rate
