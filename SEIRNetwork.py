import json
import random
from typing import Optional

import networkx as nx

from lib.networks_edit import *
from models import SEIRConfigModel


class SEIRNetwork(SEIRConfigModel):
    def __init__(self, population=4000000, numNodes=10000, config_path: str = "seir_config.json", generator_type=None) -> None:
        super().__init__(config_path)
        self.population = population
        self.numNodes = numNodes
        self.age_group_list = None
        self.G_baseline = None
        self.G_complete_quarantine = None
        self.G_household = None
        self.G_social_gathering_limit = {}
        self.age_grp_distrbution = None
        self.household_size = None

        self._generate_household_data()
        self._generate_baseline_network(generator_type)
        self._generate_complete_quarantine_network()
        self._generate_household_network()
        self._set_seir_params()
        self._generate_seir_network_model()

    def _generate_household_data(self) -> None:
        populaton_data = json.load(open("population_data.json", "r"))
        self.age_grp_distrbution = populaton_data["age_grp_distrbution"]
        self.household_size = populaton_data["household_size"]
        self.household_data = {
            "age_distn": self.age_grp_distrbution,
            "household_size_distn": {
                1: float(self.household_size['1']),
                2: float(self.household_size['2']),
                3: float(self.household_size['3']),
                4: float(self.household_size['4']),
                5: float(self.household_size['5']),
                6: float(self.household_size['6']),
            },
            "household_stats": {
                "pct_with_under20": 0.524,  # percent of households with at least one member under 20
                "pct_with_over65": 0.344,  # percent of households with at least one member over 60
                # percent of households with at least one member under 20 and at least one member over 60
                "pct_with_under20_over65": 0.076,
                # percent of households with a single-occupant that is over 60
                "pct_with_over65_givenSingleOccupant": 0.110,
                # number of people under 20 in households with at least one member under 20
                "mean_num_under20_givenAtLeastOneUnder20": 1.91,
            },
        }

    def _generate_baseline_network(self, generator_type=None):
        (
            graphs,
            self.individualAgeBracketLabels,
            self.households,
        ) = generate_demographic_contact_network(
            N=self.numNodes,
            demographic_data=self.household_data,
            distancing_scales=[
                0.7
            ],  # ~95% of individuals have no more than a single out-of-household contact
            layer_generator='FARZ' if not generator_type else generator_type
        )
        self.G_baseline = graphs["baseline"]
        self._set_household_id(self.G_baseline)
        self._set_age_group(self.G_baseline)

    def _generate_complete_quarantine_network(self):
        self.G_complete_quarantine = nx.Graph()
        self.G_complete_quarantine.add_nodes_from(self.G_baseline.nodes)
        self._set_household_id(self.G_complete_quarantine)
        self._set_age_group(self.G_complete_quarantine)

    def _generate_household_network(self):
        self.G_household = nx.Graph()
        for house in self.households:
            complete_network = nx.complete_graph(house["indices"])
            self.G_household.add_nodes_from(complete_network.nodes)
            self.G_household.add_edges_from(complete_network.edges)
        self._set_household_id(self.G_household)
        self._set_age_group(self.G_household)

    def _set_age_group(self, network):
        try:
            age_grp_dict = {
                i: self.individualAgeBracketLabels[i]
                for i in range(len(self.individualAgeBracketLabels))
            }
            nx.set_node_attributes(network, age_grp_dict, name="age_grp")
        except:
            raise "ERROR: FAILED TO SET THE AGE GROUP!!!"

    def _set_household_id(self, network):
        try:
            house_dict = {}
            for i, house in enumerate(self.households):
                for indice in house["indices"]:
                    house_dict[indice] = i
            nx.set_node_attributes(network, house_dict, name="household_id")
        except:
            raise "ERROR: FAILED TO SET THE HOUSEHOLD ID!!!"

    def _set_seir_params(self):
        self.SIGMA = 1 / self.latent_period
        assert (self.SIGMA is not None)
        self.sigma_q = ((1 / self.latent_period_q)
                        if self.latent_period_q else self.SIGMA)
        self._lambda = 1 / self.presymptomatic_period
        assert (self._lambda is not None)
        self.gamma = 1 / self.infectious_period
        assert (self.gamma is not None)
        self.gamma_asym = (
            (1 / self.infectious_period_asym)
            if self.infectious_period_asym
            else self.gamma
        )
        self.gamma_h = (
            (1 / self.infectious_period_hospital)
            if self.infectious_period_hospital
            else self.gamma
        )
        self.gamma_q_sym = (
            (1 / self.infectious_period_q_sym)
            if self.infectious_period_q_sym
            else self.gamma
        )
        self.gamma_q_asym = (
            (1 / self.infectious_period_q_asym)
            if self.infectious_period_q_asym
            else self.gamma
        )
        self.r0 = self.r0
        self.h = self.hopsital_prob
        self.f = self.death_rate_given_hospital
        self.mu_h = (
            (1 / self.admission_to_death_period)
            if self.admission_to_death_period
            else 0.0
        )
        self.eta = (
            (1 / self.onset_to_admission_period)
            if self.onset_to_admission_period
            else 0.0
        )
        self.lambda_q = 1 / self.presymptomatic_period
        self.base_beta = self.beta = 1 / (1 / self.gamma) * self.r0
        self.beta_age_grp = self.beta_by_age_group_rate
        if self.beta_age_grp:
            self.beta = [
                self.base_beta *
                self.beta_age_grp[self.individualAgeBracketLabels[i]]
                for i in range(self.numNodes)
            ]
        assert (self.beta is not None)
        self.base_alpha = self.alpha = self.rate_of_susceptibility
        self.alpha_age_grp = self.alpha_by_age_group_rate
        if self.alpha_age_grp:
            self.alpha = [
                self.base_alpha *
                self.alpha_age_grp[self.individualAgeBracketLabels[i]]
                for i in range(self.numNodes)
            ]
        self.p = self.global_infectiotion_rate
        self.q = self.global_infectiotion_rate_q
        self.theta_s = self.testing_rate_s
        self.theta_e = self.testing_rate_e
        self.theta_pre = self.testing_rate_presym
        self.theta_sym = self.testing_rate_sym
        self.theta_asym = self.testing_rate_asym
        self.phi_s = self.contact_tracing_testing_rate_s
        self.phi_e = self.contact_tracing_testing_rate_e
        self.phi_pre = self.contact_tracing_testing_rate_s
        self.phi_sym = self.contact_tracing_testing_rate_sym
        self.phi_asym = self.contact_tracing_testing_rate_asym
        self.psi_s = self.positive_test_rate_s
        self.psi_e = self.positive_test_rate_e
        self.psi_pre = self.positive_test_rate_e_presym
        self.psi_sym = self.positive_test_rate_e_sym
        self.psi_asym = self.positive_test_rate_e_asym

    def _generate_seir_network_model(self) -> None:
        """
        Generate the seir network model.
        """
        self.model = ExtSEIRSNetworkModel(
            population=self.population,
            G=self.G_baseline,
            G_Q=self.G_complete_quarantine,
            sigma=self.SIGMA,
            sigma_Q=self.sigma_q,
            lamda=self._lambda,
            gamma=self.gamma,
            gamma_asym=self.gamma_asym,
            gamma_H=self.gamma_h,
            gamma_Q_sym=self.gamma_q_sym,
            gamma_Q_asym=self.gamma_q_asym,
            h=self.h,
            f=self.f,
            mu_H=self.mu_h,
            lamda_Q=self.lambda_q,
            beta=self.beta,
            alpha=self.alpha,
            p=self.p,
            q=self.q,
            theta_S=self.theta_s,
            theta_E=self.theta_e,
            theta_pre=self.theta_pre,
            theta_sym=self.theta_sym,
            theta_asym=self.theta_asym,
            phi_S=self.phi_s,
            phi_E=self.phi_e,
            phi_pre=self.phi_pre,
            phi_sym=self.phi_sym,
            phi_asym=self.phi_asym,
            psi_S=self.psi_s,
            psi_E=self.psi_e,
            psi_pre=self.psi_pre,
            psi_sym=self.psi_sym,
            psi_asym=self.psi_asym,
            eta=self.eta,
            initE=2,
            initI_pre=2,
            initI_sym=2,
            initI_asym=0,
            isolation_time=14
        )

    def _generate_network_social_gathering_limit(self, group_limit: int) -> None:
        """
        Generates the network with social gathering limit.

        Parameters
        ----------
        group_limit: int
            The social gathering limit.
        """
        self.G_social_gathering_limit[group_limit] = nx.Graph()
        self.G_social_gathering_limit[group_limit].add_nodes_from(
            self.G_baseline)
        for i in range(self.numNodes):
            current_degree = self.G_social_gathering_limit[group_limit].degree[i]
            num_interaction = int(random.uniform(0, group_limit - 1))
            degree_to_add = max(0, num_interaction - current_degree)
            edges_list = list(self.G_baseline.edges(i))
            if degree_to_add and edges_list:
                edges_to_add = random.choices(edges_list, k=degree_to_add)
                self.G_social_gathering_limit[group_limit].add_edges_from(
                    edges_to_add)
        for i in range(self.numNodes):
            current_degree = self.G_social_gathering_limit[group_limit].degree[i]
            if current_degree > group_limit - 1:
                num_interaction = int(random.uniform(0, group_limit - 1))
                degree_to_remove = current_degree - num_interaction
                self.G_social_gathering_limit[group_limit].remove_edges_from(
                    random.choices(
                        list(
                            self.G_social_gathering_limit[group_limit].edges(i)),
                        k=degree_to_remove,
                    )
                )
        # Add back graph that represents household as interactions with memebers within household are independent of social distancing
        self.G_social_gathering_limit[group_limit].add_edges_from(
            self.G_household.edges)

    def get_network(self, type: str, group_limit: Optional[int] = None) -> nx.graph:
        """
        Get the network of type.

        :param type: The type of network to get.
        :param group_limit: The number of social distancing edges to add to the network.
        :return networkx Graph of type
        """
        if type == 'baseline':
            if not self.G_baseline:  # This should never be triggered as G_baseline will be created during init
                self._generate_baseline_network()
            return self.G_baseline

        if type == "quarantine":
            if not self.G_complete_quarantine:  # This should never be triggered as G_quarantine will be created during init
                self._generate_complete_quarantine_network()
            return self.G_complete_quarantine

        if type == "household":
            if not self.G_household:  # This should never be triggered as G_household will be created during init
                self._generate_household_network()
            return self.G_household

        if type == "social_distancing" or type == "social_gathering":
            if group_limit not in self.G_social_gathering_limit:
                self._generate_network_social_gathering_limit(group_limit)
            return self.G_social_gathering_limit[group_limit]

        raise ("Unknown type: %s" % type)
