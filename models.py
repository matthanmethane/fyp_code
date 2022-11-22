from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Dict, Optional


@dataclass
class SimulatorConfigModel:
    intervention_start_pct_infected: float
    intervention_start_time: int
    average_introductions_per_day: float
    testing_cadence: str
    pct_tested_per_day: float
    test_falseneg_rate: str
    max_pct_tests_for_symptomatic: float
    max_pct_tests_for_traces: float
    random_testing_degree_bias: int
    pct_contacts_to_trace: float
    tracing_lag: int
    isolation_lag_symptomatic: int
    isolation_lag_positive: int
    isolation_lag_contact: int
    testing_compliance_rate_symptomatic: float
    testing_compliance_rate_traced: float
    testing_compliance_rate_random: float
    tracing_compliance_rate: float
    isolation_compliance_rate_symptomatic_individual: float
    isolation_compliance_rate_symptomatic_groupmate: float
    isolation_compliance_rate_positive_individual: float
    isolation_compliance_rate_positive_groupmate: float
    isolation_compliance_rate_positive_contact: float
    isolation_compliance_rate_positive_contactgroupmate: float
    num_contacts_to_trace: Optional[int] = None

    def __init__(self, config_path="simulator_config.json"):
        config_dict = json.load(open(config_path, 'r'))
        for key, value in config_dict.items():
            setattr(self, key, value)


@dataclass
class SEIRConfigModel:
    latent_period: float
    infectious_period: int
    presymptomatic_period: int
    r0: Optional[float] = None
    beta_by_age_group_rate: Optional[Dict[str, float]] = None
    beta_asym_rate: Optional[float] = None
    beta_local_rate: Optional[float] = None
    beta_local_asym_rate: Optional[float] = None
    asymp_prob: Optional[float] = None
    latent_period_q: Optional[float] = None
    rate_of_susceptibility: Optional[float] = 1.0
    infectious_period_asym: Optional[int] = None
    hopsital_prob: Optional[float] = None
    death_rate_given_hospital: Optional[float] = None
    onset_to_admission_period: Optional[float] = None
    infectious_period_hospital: Optional[int] = None
    admission_to_death_period: Optional[int] = None
    rate_of_resusceptibility: Optional[float] = None
    global_infectiotion_rate: Optional[float] = None
    global_infectiotion_rate_q: Optional[float] = None
    infectious_period_q_sym: Optional[int] = None
    infectious_period_q_asym: Optional[int] = None
    alpha_by_age_group_rate: Optional[Dict[str, float]] = None
    alpha_q_rate: Optional[float] = None
    beta_q_rate: Optional[float] = None
    beta_q_local_rate: Optional[float] = None
    testing_rate_s: Optional[float] = None
    testing_rate_e: Optional[float] = None
    testing_rate_presym: Optional[float] = None
    testing_rate_sym: Optional[float] = None
    testing_rate_asym: Optional[float] = None
    contact_tracing_testing_rate_s: Optional[float] = None
    contact_tracing_testing_rate_e: Optional[float] = None
    contact_tracing_testing_rate_presym: Optional[float] = None
    contact_tracing_testing_rate_sym: Optional[float] = None
    contact_tracing_testing_rate_asym: Optional[float] = None
    positive_test_rate_s: Optional[float] = None
    positive_test_rate_e: Optional[float] = None
    positive_test_rate_e_presym: Optional[float] = None
    positive_test_rate_e_sym: Optional[float] = None
    positive_test_rate_e_asym: Optional[float] = None
    quarantine_days: Optional[int] = 14

    def __init__(self, config_path="seir_config.json"):
        config_dict = json.load(open(config_path, 'r'))
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_param(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
