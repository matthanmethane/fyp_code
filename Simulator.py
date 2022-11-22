from tkinter import W
from typing import Optional, Dict

import numpy as np
from matplotlib import pyplot as plt

from models import SimulatorConfigModel
from SEIRNetwork import SEIRNetwork
from Simulation import Simulation
import networkx as nx


class Simulator(SimulatorConfigModel):
    def __init__(self, seir_network: SEIRNetwork, config_path: Optional[str] = "simulator_config.json") -> None:
        super().__init__(config_path)

        self.simulation = None
        self.SEIRModel = seir_network
        self.model = self.SEIRModel.model
        self.numNodes = self.SEIRModel.numNodes
        self.intervention_history = []
        self._update_parameters()

    def _update_parameters(self):
        self.testing_compliance_random = (
            np.random.rand(
                self.numNodes) < self.testing_compliance_rate_random
        )
        self.testing_compliance_traced = (
            np.random.rand(
                self.numNodes) < self.testing_compliance_rate_traced
        )
        self.testing_compliance_symptomatic = (
            np.random.rand(self.numNodes)
            < self.testing_compliance_rate_symptomatic
        )
        self.tracing_compliance = (
            np.random.rand(
                self.numNodes) < self.tracing_compliance_rate
        )
        self.isolation_compliance_symptomatic_individual = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_symptomatic_individual
        )
        self.isolation_compliance_symptomatic_groupmate = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_symptomatic_groupmate
        )
        self.isolation_compliance_positive_individual = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_positive_individual
        )
        self.isolation_compliance_positive_groupmate = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_positive_groupmate
        )
        self.isolation_compliance_positive_contact = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_positive_contact
        )
        self.isolation_compliance_positive_contactgroupmate = (
            np.random.rand(self.numNodes)
            < self.isolation_compliance_rate_positive_contactgroupmate
        )

    def set_checkpoint(self, new_checkpoint: Dict) -> None:
        """
        Set the new checkpoint.
        """
        pass

    def set_trace_together(
            self,
            time: float,
            tracing_lag: int,
            tracing_compliance_rate: float,
    ):
        """
        Set TraceTogether.
        """
        # self.tracing_lag = tracing_lag
        # self.tracing_compliance_rate = tracing_compliance_rate
        # self._update_parameters()
        self.simulation.set_params(
            time=time,
            tracing_lag=tracing_lag,
            tracing_compliance=(
                np.random.rand(self.numNodes) < tracing_compliance_rate)
        )
        self.intervention_history.append({'t': time, 'type': 'TraceTogether'})

    def set_imported_case(self, time: float, average_introductions_per_day: float):
        self.simulation.set_seir_network_params(
            time=time, average_introductions_per_day=average_introductions_per_day)

    def set_social_distancing(self, time: float, global_rate: float):
        self.simulation.set_seir_network_params(time=time, p=global_rate)

    def set_social_gathering_limit(self, time: float, group_size: int):
        self.simulation.set_seir_network_params(
            time=time, G=self.SEIRModel.get_network("social_distancing", group_size))

    def generate_simulation(self, T=90):
        self.simulation = Simulation(
            model=self.model,
            T=T,
            intervention_start_pct_infected=self.intervention_start_pct_infected,
            intervention_start_time=self.intervention_start_time,
            average_introductions_per_day=self.average_introductions_per_day,
            testing_cadence=self.testing_cadence,
            pct_tested_per_day=self.pct_tested_per_day,
            test_falseneg_rate=self.test_falseneg_rate,
            testing_compliance_symptomatic=self.testing_compliance_symptomatic,
            max_pct_tests_for_symptomatic=self.max_pct_tests_for_symptomatic,
            testing_compliance_traced=self.testing_compliance_traced,
            max_pct_tests_for_traces=self.max_pct_tests_for_traces,
            testing_compliance_random=self.testing_compliance_random,
            random_testing_degree_bias=self.random_testing_degree_bias,
            tracing_compliance=self.tracing_compliance,
            pct_contacts_to_trace=self.pct_contacts_to_trace,
            tracing_lag=self.tracing_lag,
            isolation_compliance_symptomatic_individual=self.isolation_compliance_symptomatic_individual,
            isolation_compliance_symptomatic_groupmate=self.isolation_compliance_symptomatic_groupmate,
            isolation_compliance_positive_individual=self.isolation_compliance_positive_individual,
            isolation_compliance_positive_groupmate=self.isolation_compliance_positive_groupmate,
            isolation_compliance_positive_contact=self.isolation_compliance_positive_contact,
            isolation_compliance_positive_contactgroupmate=self.isolation_compliance_positive_contactgroupmate,
            isolation_lag_symptomatic=self.isolation_lag_symptomatic,
            isolation_lag_positive=self.isolation_lag_positive,
        )

    def run(self):
        if not self.simulation:
            self.generate_simulation()
        while self.simulation.running:
            self.run_iteration()

    def run_iteration(self):
        if not self.simulation:
            self.generate_simulation()
        self.simulation.run()


# SEIR_network = SEIRNetwork()
# # pos = nx.drawing.layout.spring_layout(SEIR_network.G_household)
# # nx.draw_networkx(SEIR_network.G_household, pos,
# #                  with_labels=False, node_size=100)
# # plt.show()
# # print("NETOWRK PRINTED HAHAHAHAHAHAHAHA")

# simulator = Simulator(SEIR_network)
# simulator.generate_simulation()


# # =================================================================================================
# # +++++ Feb 1st ~ May 1st +++++
# simulator.set_imported_case(time=1, average_introductions_per_day=0.1)
# simulator.set_trace_together(
#     time=49, tracing_lag=1, tracing_compliance_rate=0.8)
# simulator.set_social_gathering_limit(time=57, group_size=10)
# simulator.set_imported_case(time=25, average_introductions_per_day=0.4)
# simulator.set_imported_case(time=40, average_introductions_per_day=1.0)
# simulator.set_social_distancing(time=68, global_rate=0.01)
# simulator.run()
# ax1 = plt.subplot(2, 1, 1)
# ax1.bar(simulator.simulation.numPosTseries.keys(),
#         simulator.simulation.numPosTseries.values())

# ax2 = plt.subplot(2, 1, 2)
# simulator.model.figure_infections(
#     ax=ax2,
#     plot_S=False,
#     plot_E=False,
#     plot_I_pre=False,
#     plot_I_sym='stacked',
#     plot_Q_sym='stacked',
#     plot_Q_pre=False,
#     plot_Q_asym=False,
#     plot_I_asym=False,
#     show=False,
#     combine_Q_infected=False,
#     plot_percentages=False,
#     scaled=True
# )

# # plt.title("Effect of Dynamic Scaling")
# #
# # ax1 = plt.subplot(2,2,1)
# # plt.title("Non-Scaled Non-S")
# # simulator.model.figure_infections(
# #     plot_S=False,
# #     plot_E="stacked",
# #     plot_I_pre="stacked",
# #     plot_I_sym="stacked",
# #     plot_I_asym="stacked",
# #     plot_H="stacked",
# #     plot_R="stacked",
# #     plot_F="stacked",
# #     plot_Q_E="stacked",
# #     plot_Q_pre="stacked",
# #     plot_Q_sym="stacked",
# #     plot_Q_asym="stacked",
# #     plot_Q_S=False,
# #     plot_Q_R="stacked",
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     show=False,
# #     scaled=False,
# #     ax=ax1
# # )
# # ax2 = plt.subplot(2,2,2)
# # plt.title("Scaled Non-S")
# # simulator.model.figure_infections(
# #     plot_S=False,
# #     plot_E="stacked",
# #     plot_I_pre="stacked",
# #     plot_I_sym="stacked",
# #     plot_I_asym="stacked",
# #     plot_H="stacked",
# #     plot_R="stacked",
# #     plot_F="stacked",
# #     plot_Q_E="stacked",
# #     plot_Q_pre="stacked",
# #     plot_Q_sym="stacked",
# #     plot_Q_asym="stacked",
# #     plot_Q_S=False,
# #     plot_Q_R="stacked",
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     show=False,
# #     scaled=True,
# #     ax=ax2
# # )
# #
# # ax3 = plt.subplot(2,2,3)
# # plt.title("Non-Scaled SEIR")
# # simulator.model.figure_basic(
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     plot_S='stacked',
# #     plot_Q_S='stacked',
# #     show=False,
# #     scaled=False,
# #     ax=ax3
# # )
# # ax4 = plt.subplot(2,2,4)
# # plt.title("Scaled SEIR")
# # simulator.model.figure_basic(
# #     combine_Q_infected=False,
# #     plot_percentages=False,
# #     plot_S='stacked',
# #     plot_Q_S='stacked',
# #     plot_Q_R='line',
# #     show=False,
# #     scaled=True,
# #     ax=ax4
# # )
# #
# #
# # plt.suptitle("Effect of Dynamic Scaling")
# plt.show()
