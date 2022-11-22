import json
from typing import Optional

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from SEIRNetwork import SEIRNetwork
from Simulator import Simulator


@st.experimental_singleton(suppress_st_warning=True)
def get_seir_network(num_population):
    seir_model = SEIRNetwork(population=num_population)
    return seir_model


@st.experimental_singleton(suppress_st_warning=True)
def get_simulator(seir_model):
    sim = Simulator(seir_model)
    return sim


@st.experimental_singleton(suppress_st_warning=True)
def plot_distribution(network_type, social_gathering_limit: Optional[int] = None):
    if social_gathering_limit:
        graph = get_seir_network(population).get_network(network_type, social_gathering_limit)
    else:
        graph = get_seir_network(population).get_network(network_type)
    nodeDegrees = [d[1] for d in graph.degree()]
    meanDegree = np.mean(nodeDegrees)
    fig, ax = plt.subplots()
    ax.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.75, color='tab:blue',
            label=('mean degree = %.1f' % meanDegree))
    plt.xlim(0, max(nodeDegrees))
    ax.legend(loc='upper right')
    return fig


if 'gen_net' not in st.session_state:
    st.session_state['gen_net'] = False

st.set_page_config(
    page_title="COVID-19 Singapore Simulator",
    page_icon='🇸🇬',
    layout='centered'
)

st.title("COVID-19 Simulator")


def set_seir_config(default=False):
    if not default and config_file:
        seir_config_json = json.load(config_file)
    else:
        seir_config_json = json.load(open('seir_config.json', 'r'))
    st.session_state.seir_config_json = seir_config_json


BASIC_PARAM = ['latent_period', 'infectious_period', 'presymptomatic_period', 'r0']


def init_seir_config_ui():
    for key, value in st.session_state.seir_config_json.items():
        if not isinstance(value, dict):
            if key in BASIC_PARAM:
                st.session_state.seir_config_dict[key] = basic_params.number_input(key, value=float(value))
            else:
                st.session_state.seir_config_dict[key] = advanced_params.number_input(key, value=float(value))

        else:
            age_grp_params.text(key)
            st.session_state.seir_config_dict[key] = {}
            for age_grp, rate in value.items():
                st.session_state.seir_config_dict[key][age_grp] = age_grp_params.number_input(age_grp, key=key+age_grp,value=float(rate))


pop, f_upload = st.columns([1, 3])
population = pop.number_input("Enter population", min_value=10000, value=4000000)
config_file = f_upload.file_uploader("Import Network Configuration", type='json')
st.write("Configuration")

basic_params = st.expander("Basic Parameters", expanded=True)
advanced_params = st.expander("Advanced Parameters", expanded=False)
age_grp_params = st.expander("Age Group Parameters", expanded=False)

if 'seir_config_json' not in st.session_state:
    st.session_state.seir_config_dict = {}
    set_seir_config(True)
    init_seir_config_ui()

if config_file:
    set_seir_config()

if st.button("Load Default Config"):
    set_seir_config(True)

st.markdown("""---""")

if st.button('Generate Network') or st.session_state['gen_net']:
    init_seir_config_ui()
    st.session_state['gen_net'] = True
    network_generating = st.text("Generating Network of Singapore...")
    get_seir_network(population).update_param(st.session_state.seir_config_dict)

    baseline_tab, quarantine_tab, household_tab, social_gathering_tab = st.tabs(
        ["Baseline", "Quarantine", "Household", "Social Gathering Limit"])
    with baseline_tab:
        st.subheader('Network information')
        st.text(get_seir_network(population).get_network('baseline'))
        st.pyplot(plot_distribution('baseline'))

    with quarantine_tab:
        st.subheader('Network information')
        st.text(get_seir_network(population).get_network('quarantine'))
        st.pyplot(plot_distribution('quarantine'))

    with household_tab:
        st.subheader('Network information')
        st.text(get_seir_network(population).get_network('household'))
        st.pyplot(plot_distribution('household'))

    with social_gathering_tab:
        social_gathering_val = social_gathering_tab.slider('Gathering Size Limit', min_value=0, max_value=50, value=0)
        if social_gathering_val:
            st.subheader('Network information')
            st.text(get_seir_network(population).get_network('social_distancing', social_gathering_val))
            st.pyplot(plot_distribution('social_distancing', social_gathering_val))

    network_generating.text("Complete!")

    st.markdown("""---""")

    st.header("Add restrictions")

    st.file_uploader("Import Simulator Configuration", type='json')

    restrict1, restrict2, restrict3, submit_restrict = st.columns(4)
    if restrict1.checkbox('Set TraceTogether', value=False):
        trace_tgt_col1, trace_tgt_col2 = st.columns([1, 1])
        tracing_lag = trace_tgt_col1.slider("Tracing Lag", 0, 7, 1)
        tracing_compliance_rate = trace_tgt_col2.slider("Tracing Compliance Rate", 0.0, 1.0, 0.8)
    else:
        tracing_lag = None
        tracing_compliance_rate = None

    if restrict2.checkbox('Set social gathering limit', value=False):
        social_gathering_limit_size = st.slider('Gathering Size Limit', 0, 10, 10)
    else:
        social_gathering_limit_size = None

    if restrict3.checkbox('Set social distancing', value=False):
        social_distancing_size = st.slider("Social Distancing Rate", 0.0, 1.0, 1.0)
    else:
        social_distancing_size = None

    if submit_restrict.button('Set restriction'):
        pass

    num_days = st.number_input("Number of Days to simulate", value=120)

    intervene1, intervene2, intervene3 = st.columns([1, 2, 1])

    intervention_time = intervene1.number_input("Time of Intervention", min_value=0, max_value=num_days)
    intervention_param = intervene2.selectbox(
        'Parameter Type',
        [
            'average_introductions_per_day',
            'tracing_lag',
            'tracing_compliance_rate'
        ]
    )
    intervention_value = intervene3.number_input("Rate of intervention", min_value=0.0, max_value=1.0, value=0.5)

    sim = get_simulator(seir_model=get_seir_network(population))
    sim.generate_simulation(T=num_days)

    if st.button("Add Intervention"):
        st.write(f"{intervention_time} {intervention_param} {intervention_value}")
        sim.simulation.set_params(time=intervention_time, **{intervention_param: intervention_value})

    st.markdown("""---""")

    if st.button("Start Simulation"):
        with st.spinner("Simulating..."):
            sim = get_simulator(seir_model=get_seir_network(population))
            sim.model.reset()
            sim.generate_simulation(T=num_days)
            # sim.simulation.set_params(time=30, average_introductions_per_day=0.0)
            # sim.set_social_gathering_limit(time=30, group_size=1)
            # sim.simulation.set_params(time=80, average_introductions_per_day=0.9)
            # sim.simulation.set_seir_network_params(time=80, G=get_seir_network(population).get_network("baseline"))
            sim.run()
            fig, ax = sim.model.figure_infections(
                plot_S=False,
                plot_E=False,
                plot_I_pre=False,
                plot_I_sym="stacked",
                plot_I_asym=False,
                plot_H=False,
                plot_R=False,
                plot_F=False,
                plot_Q_E=False,
                plot_Q_pre=False,
                plot_Q_sym="stacked",
                plot_Q_asym=False,
                plot_Q_S=False,
                plot_Q_R=False,
                combine_Q_infected=False,
                plot_percentages=False,
            )
        st.pyplot(fig, clear_figure=True)
