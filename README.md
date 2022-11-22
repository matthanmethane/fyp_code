# Code for the Final Year Project

##
'''
Simulator.py
'''
This file is where you can run the simulator

# Sample Code
'''

SEIR_network = SEIRNetwork()
simulator = Simulator(SEIR_network)
simulator.generate_simulation()
simulator.set_imported_case(time=1, average_introductions_per_day=0.1)
simulator.set_trace_together(time=49, tracing_lag=1, tracing_compliance_rate=0.8)
simulator.set_social_gathering_limit(time=57, group_size=10)
simulator.set_imported_case(time=25, average_introductions_per_day=0.4)
simulator.set_imported_case(time=40, average_introductions_per_day=1.0)
simulator.set_social_distancing(time=68, global_rate=0.01)
simulator.run()

'''

IPYNB files are ran to gather results and screenshots for the Report 
