# import libs
import logging
import numpy as np
from typing import List, Dict, Any, cast
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, CustomProp, ComponentKey
from pyreactlab_core import build_rxns_stoichiometry
from pyThermoLinkDB.models import ModelSource
from pythermocalcdb.reactions import build_hsg_reaction
from pythermocalcdb.core import HSGReaction
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from ..utils.reaction_tools import stoichiometry_mat

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoReaction:
    """
    A class representing the thermodynamic and kinetic information for a chemical reaction system.
    """

    def __init__(
        self,
        components: List[Component],
        model_source: ModelSource,
        thermo_inputs: Dict[str, Any],
        reaction_rates: List[ReactionRateExpression],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoReaction instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        model_source : ModelSource
            A ModelSource object containing the source of the model to be used in the simulation.
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the inputs for the batch reactor simulation, such as volume, heat transfer properties, etc.
        reaction_rates : List[ReactionRateExpression]
            A list of reaction rate expressions, where each expression is represented as a ReactionRateExpression object containing information about the reaction and its rate expression.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
        # NOTE: Set attributes
        self.components = components
        self.model_source = model_source
        self.thermo_inputs = thermo_inputs
        self.reaction_rates = reaction_rates
        self.component_key = component_key

        # NOTE: Build reactions and stoichiometry matrix
        self.reactions = self.build_reactions()

        # NOTE: Build stoichiometry matrix
        # self.stoichiometry_matrix = self.build_stoichiometry_matrix()
        self.stoichiometry_matrix = None

        # SECTION: build hsg reactions
        self.hsg_reactions = self._config_hsg_reactions()

    # SECTION: Reaction and stoichiometry related methods
    # ! Extract all reactions

    def build_reactions(self):
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        reactions = []
        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            reactions.append(rxn)
        return reactions

    # ! Build stoichiometry matrix
    def build_stoichiometry(self):
        """
        Build the stoichiometry matrix for the reactions in the gas-phase batch reactor using the provided reaction rates and components.
        """
        # >> extract reactions from reaction rates
        reactions = []

        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            reactions.append(rxn)

        # >> build stoichiometry matrix
        mat = stoichiometry_mat(
            reactions=reactions,
            components=self.components,
            component_key=cast(ComponentKey, self.component_key)
        )

        return mat

    # ! Build stoichiometry matrix using pythermolinkdb
    def build_stoichiometry_matrix(self):
        """
        Build the stoichiometry matrix for the reactions in the gas-phase batch reactor using the provided reaction rates and components.
        """
        # >> extract reactions from reaction rates
        res = build_rxns_stoichiometry(
            reactions=self.build_reactions(),
            components=self.components,
            component_key=cast(ComponentKey, self.component_key)
        )

        # >> check
        if res is None:
            raise ValueError(
                "Failed to build stoichiometry matrix. No valid reactions or components found."
            )

        # check matrix
        if 'matrix' not in res:
            raise ValueError(
                "Failed to build stoichiometry matrix. No matrix found in the result."
            )

        # to numpy array
        mat = res['matrix']

        # res
        return np.array(mat, dtype=float)

    # ! Get reaction names and indices
    def get_reaction_names(self) -> List[str]:
        """
        Get the list of reaction names for the reactions in the gas-phase batch reactor using the provided reaction rates.

        Returns
        -------
        List[str]
            A list of reaction names for the reactions in the gas-phase batch reactor.
        """
        reaction_names = []
        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            reaction_names.append(rxn.name)
        return reaction_names

    # ! Get reaction index mapping
    def get_reaction_index(self) -> Dict[str, int]:
        """
        Get the reaction index for the reactions in the reactor using the provided reaction rates.
        Returns
        -------
        Dict[str, int]
            A dictionary where the keys are reaction names and the values are the reaction orders for the reactions in the gas-phase batch reactor.
        """
        res = {}
        for index, rate_exp in enumerate(self.reaction_rates):
            rxn = rate_exp.reaction
            res[rxn.name] = index
        return res

    # NOTE: build hsg reactions
    def _config_hsg_reactions(self) -> List[HSGReaction]:
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        hsg_reactions: list[HSGReaction] = []

        # iterate through reaction rates and build hsg reactions
        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            hsg_rxn = build_hsg_reaction(
                reaction=rxn,
                model_source=self.model_source,
            )

            # >> check
            if hsg_rxn is None:
                logger.warning(
                    f"Failed to build HSG reaction for reaction: {rxn.name}. Skipping this reaction."
                )
                raise ValueError(
                    f"Failed to build HSG reaction for reaction: {rxn.name}. Skipping this reaction."
                )

            # add
            hsg_reactions.append(hsg_rxn)
        return hsg_reactions
