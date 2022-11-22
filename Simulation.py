from __future__ import division

import heapq

import numpy


class Simulation:
    def __init__(
            self,
            model,
            T,
            max_dt=None,
            checkpoints=None,
            intervention_start_pct_infected=0,
            intervention_start_time=0,
            average_introductions_per_day=0,
            testing_cadence='everyday',
            pct_tested_per_day=1.0,
            test_falseneg_rate='temporal',
            testing_compliance_symptomatic=[None],
            max_pct_tests_for_symptomatic=1.0,
            testing_compliance_traced=[None],
            max_pct_tests_for_traces=1.0,
            testing_compliance_random=[None],
            random_testing_degree_bias=0,
            tracing_compliance=[None],
            num_contacts_to_trace=None,
            pct_contacts_to_trace=1.0,
            tracing_lag=1,
            isolation_compliance_symptomatic_individual=[None],
            isolation_compliance_symptomatic_groupmate=[None],
            isolation_compliance_positive_individual=[None],
            isolation_compliance_positive_groupmate=[None],
            isolation_compliance_positive_contact=[None],
            isolation_compliance_positive_contactgroupmate=[None],
            isolation_lag_symptomatic=1,
            isolation_lag_positive=1,
            isolation_lag_contact=0,
            isolation_groups=None,
            cadence_testing_days=None,
            cadence_cycle_length=28,
            temporal_falseneg_rates=None,
            backlog_skipped_intervals=False
    ):
        self.interventionStartTime = None
        self.interventionOn = None
        self.timeOfLastIntroduction = None
        self.running = None
        self.timeOfLastIntervention = None
        self.checkpointTime = None
        self.model = model
        self.T = T
        self.max_dt = max_dt
        self.checkpoints = checkpoints
        self.intervention_start_pct_infected = intervention_start_pct_infected
        self.intervention_start_time = intervention_start_time
        self.average_introductions_per_day = average_introductions_per_day
        self.testing_cadence = testing_cadence
        self.pct_tested_per_day = pct_tested_per_day
        self.test_falseneg_rate = test_falseneg_rate
        self.testing_compliance_symptomatic = testing_compliance_symptomatic
        self.max_pct_tests_for_symptomatic = max_pct_tests_for_symptomatic
        self.testing_compliance_traced = testing_compliance_traced
        self.max_pct_tests_for_traces = max_pct_tests_for_traces
        self.testing_compliance_random = testing_compliance_random
        self.random_testing_degree_bias = random_testing_degree_bias
        self.tracing_compliance = tracing_compliance
        self.num_contacts_to_trace = num_contacts_to_trace
        self.pct_contacts_to_trace = pct_contacts_to_trace
        self.tracing_lag = tracing_lag
        self.isolation_compliance_symptomatic_individual = isolation_compliance_symptomatic_individual
        self.isolation_compliance_symptomatic_groupmate = isolation_compliance_symptomatic_groupmate
        self.isolation_compliance_positive_individual = isolation_compliance_positive_individual
        self.isolation_compliance_positive_groupmate = isolation_compliance_positive_groupmate
        self.isolation_compliance_positive_contact = isolation_compliance_positive_contact
        self.isolation_compliance_positive_contactgroupmate = isolation_compliance_positive_contactgroupmate
        self.isolation_lag_symptomatic = isolation_lag_symptomatic
        self.isolation_lag_positive = isolation_lag_positive
        self.isolation_lag_contact = isolation_lag_contact
        self.isolation_groups = isolation_groups
        self.cadence_testing_days = cadence_testing_days
        self.cadence_cycle_length = cadence_cycle_length
        self.temporal_falseneg_rates = temporal_falseneg_rates
        self.backlog_skipped_intervals = backlog_skipped_intervals

        self.update_param_heap = []  # Heap to store [TIME,PARAM,VALUE] for TTI
        self.checkpoints_heap = []  # Heap to store [TIME,PARAM,VALUE] for SEIR Network Model

        self._init_simulation()

    def set_params(self, time: int, **kwargs) -> None:
        if time >= (self.model.t - 1):
            for key, value in kwargs.items():
                heapq.heappush(self.update_param_heap, [time, key, value])

    def _init_simulation(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Testing cadences involve a repeating 28 day cycle starting on a Monday
        # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
        # For each cadence, testing is done on the day numbers included in the associated list.

        if self.cadence_testing_days is None:
            self.cadence_testing_days = {
                'everyday': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27],
                'workday': [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25],
                'semiweekly': [0, 3, 7, 10, 14, 17, 21, 24],
                'weekly': [0, 7, 14, 21],
                'biweekly': [0, 14],
                'monthly': [0],
                'cycle_start': [0]
            }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.temporal_falseneg_rates is None:
            self.temporal_falseneg_rates = {
                self.model.E: {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                self.model.I_pre: {0: 0.25, 1: 0.25, 2: 0.22},
                self.model.I_sym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34,
                                   9: 0.38,
                                   10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76,
                                   18: 0.79,
                                   19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96,
                                   27: 0.97,
                                   28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                self.model.I_asym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34,
                                    9: 0.38,
                                    10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76,
                                    18: 0.79,
                                    19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96,
                                    27: 0.97,
                                    28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                self.model.Q_E: {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
                self.model.Q_pre: {0: 0.25, 1: 0.25, 2: 0.22},
                self.model.Q_sym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34,
                                   9: 0.38,
                                   10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76,
                                   18: 0.79,
                                   19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96,
                                   27: 0.97,
                                   28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
                self.model.Q_asym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34,
                                    9: 0.38,
                                    10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76,
                                    18: 0.79,
                                    19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96,
                                    27: 0.97,
                                    28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
            }

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Custom simulation loop:
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.interventionOn = False
        self.interventionStartTime = None

        self.timeOfLastIntervention = -1
        self.timeOfLastIntroduction = -1

        self.testingDays = self.cadence_testing_days[self.testing_cadence]
        self.cadenceDayNumber = 0

        self.tests_per_day = int(self.model.numNodes * self.pct_tested_per_day)
        self.max_tracing_tests_per_day = int(
            self.tests_per_day * self.max_pct_tests_for_traces)
        self.max_symptomatic_tests_per_day = int(
            self.tests_per_day * self.max_pct_tests_for_symptomatic)

        self.tracingPoolQueue = [[] for i in range(self.tracing_lag)]
        self.isolationQueue_symptomatic = [
            [] for i in range(self.isolation_lag_symptomatic)]
        self.isolationQueue_positive = [[]
                                        for i in range(self.isolation_lag_positive)]
        self.isolationQueue_contact = [[]
                                       for i in range(self.isolation_lag_contact)]

        self.numPosTseries = {0: 0}
        self.model.tmax = self.T
        self.running = True

        # if self.checkpoints:
        #     self.numCheckpoints = len(self.checkpoints['t'])
        #     for chkpt_param, chkpt_values in self.checkpoints.items():
        #         assert (isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values) ==
        #                 self.numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times (" + str(
        #             self.numCheckpoints) + ") for each checkpoint parameter."
        #     # Finds 1st index in list greater than given val
        #     self.checkpointIdx = numpy.searchsorted(self.checkpoints['t'], self.model.t)
        #     if self.checkpointIdx >= self.numCheckpoints:
        #         # We are out of checkpoints, stop checking them:
        #         self.checkpoints = None
        #     else:
        #         self.checkpointTime = self.checkpoints['t'][self.checkpointIdx]

    def set_seir_network_params(self, time: int, **kwargs) -> None:
        if time >= (self.model.t-1):
            for param, value in kwargs.items():
                if param in list(self.model.parameters.keys()):
                    heapq.heappush(self.checkpoints_heap, [time, param, value])

    def run(self):
        self.running = self.model.run_iteration(max_dt=self.max_dt)
        # Handle TTI Parameter Update
        while self.update_param_heap and self.update_param_heap[0][0] <= self.model.t:
            _, key, value = heapq.heappop(self.update_param_heap)
            setattr(self, key, value)
            print(f"[UPDATE @ t = {int(self.model.t)} ({key}-> {value})]")

        while self.checkpoints_heap and self.checkpoints_heap[0][0] <= self.model.t:
            _, param, value = heapq.heappop(self.checkpoints_heap)
            self.model.parameters[param] = value
            self.model.update_parameters()
            print(f"[UPDATE @ t = {int(self.model.t)} ({param}-> {value})]")

        # # Handle checkpoints if applicable:
        # if self.checkpoints:
        #     if self.model.t >= self.checkpointTime:
        #         print("[Checkpoint: Updating parameters]")
        #         # A checkpoint has been reached, update param values:
        #         for param in list(self.model.parameters.keys()):
        #             if param in list(self.checkpoints.keys()):
        #                 self.model.parameters.update(
        #                     {param: self.checkpoints[param][self.checkpointIdx]})
        #         # Update parameter data structures and scenario flags:
        #         self.model.update_parameters()
        #         # Update the next checkpoint time:
        #         # Finds 1st index in list greater than given val
        #         checkpointIdx = numpy.searchsorted(self.checkpoints['t'], self.model.t)
        #         if checkpointIdx >= self.numCheckpoints:
        #             # We are out of checkpoints, stop checking them:
        #             self.checkpoints = None
        #         else:
        #             self.checkpointTime = self.checkpoints['t'][checkpointIdx]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Introduce exogenous exposures randomly:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if int(self.model.t) != int(self.timeOfLastIntroduction):

            self.timeOfLastIntroduction = self.model.t

            numNewExposures = numpy.random.poisson(
                lam=self.average_introductions_per_day)

            self.model.introduce_exposures(num_new_exposures=numNewExposures)

            if numNewExposures > 0:
                print("[NEW EXPOSURE @ t = %.2f (%d exposed)]" %
                      (self.model.t, numNewExposures))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute testing policy at designated intervals:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if int(self.model.t) != int(self.timeOfLastIntervention):

            cadenceDayNumbers = [int(self.model.t % self.cadence_cycle_length)]

            if self.backlog_skipped_intervals:
                cadenceDayNumbers = [int(i % self.cadence_cycle_length) for i in numpy.arange(
                    start=self.timeOfLastIntervention, stop=int(self.model.t), step=1.0)[1:]] + cadenceDayNumbers

            self.timeOfLastIntervention = self.model.t

            for cadenceDayNumber in cadenceDayNumbers:

                currentNumInfected = self.model.total_num_infected()[
                    self.model.tidx]
                currentPctInfected = self.model.total_num_infected()[
                    self.model.tidx] / self.model.numNodes

                if currentPctInfected >= self.intervention_start_pct_infected and not self.interventionOn:
                    self.interventionOn = True
                    self.interventionStartTime = self.model.t

                if self.model.t >= self.intervention_start_time and not self.interventionOn:
                    self.interventionOn = True
                    self.interventionStartTime = self.model.t

                if self.interventionOn:

                    print("[INTERVENTIONS @ t = %.2f (%d (%.2f%%) infected)]" %
                          (self.model.t, currentNumInfected, currentPctInfected * 100))

                    nodeStates = self.model.X.flatten()
                    nodeTestedStatuses = self.model.tested.flatten()
                    nodeTestedInCurrentStateStatuses = self.model.testedInCurrentState.flatten()
                    nodePositiveStatuses = self.model.positive.flatten()

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # tracingPoolQueue[0] = tracingPoolQueue[0]Queue.pop(0)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    newIsolationGroup_symptomatic = []
                    newIsolationGroup_contact = []

                    # ----------------------------------------
                    # Isolate SYMPTOMATIC cases without a test:
                    # ----------------------------------------
                    numSelfIsolated_symptoms = 0
                    numSelfIsolated_symptomaticGroupmate = 0

                    if any(self.isolation_compliance_symptomatic_individual):
                        symptomaticNodes = numpy.argwhere(
                            (nodeStates == self.model.I_sym)).flatten()
                        for symptomaticNode in symptomaticNodes:
                            if self.isolation_compliance_symptomatic_individual[symptomaticNode]:
                                if self.model.X[symptomaticNode] == self.model.I_sym:
                                    numSelfIsolated_symptoms += 1
                                    newIsolationGroup_symptomatic.append(
                                        symptomaticNode)

                                # ----------------------------------------
                                # Isolate the GROUPMATES of this SYMPTOMATIC node without a test:
                                # ----------------------------------------
                                if (
                                        self.isolation_groups is not None and any(
                                            self.isolation_compliance_symptomatic_groupmate)):
                                    isolationGroupmates = next(
                                        (group for group in self.isolation_groups if symptomaticNode in group), None)
                                    for isolationGroupmate in isolationGroupmates:
                                        if isolationGroupmate != symptomaticNode:
                                            if self.isolation_compliance_symptomatic_groupmate[isolationGroupmate]:
                                                numSelfIsolated_symptomaticGroupmate += 1
                                                newIsolationGroup_symptomatic.append(
                                                    isolationGroupmate)

                    # ----------------------------------------
                    # Isolate the CONTACTS of detected POSITIVE cases without a test:
                    # ----------------------------------------
                    numSelfIsolated_positiveContact = 0
                    numSelfIsolated_positiveContactGroupmate = 0

                    if (any(self.isolation_compliance_positive_contact) or any(
                            self.isolation_compliance_positive_contactgroupmate)):
                        for contactNode in self.tracingPoolQueue[0]:
                            if self.isolation_compliance_positive_contact[contactNode]:
                                newIsolationGroup_contact.append(contactNode)
                                numSelfIsolated_positiveContact += 1

                            # ----------------------------------------
                            # Isolate the GROUPMATES of this self-isolating CONTACT without a test:
                            # ----------------------------------------
                            if (
                                    self.isolation_groups is not None and any(
                                        self.isolation_compliance_positive_contactgroupmate)):
                                isolationGroupmates = next(
                                    (group for group in self.isolation_groups if contactNode in group), None)
                                for isolationGroupmate in isolationGroupmates:
                                    # if(isolationGroupmate != contactNode):
                                    if self.isolation_compliance_positive_contactgroupmate[isolationGroupmate]:
                                        newIsolationGroup_contact.append(
                                            isolationGroupmate)
                                        numSelfIsolated_positiveContactGroupmate += 1

                    # ----------------------------------------
                    # Update the nodeStates list after self-isolation updates to model.X:
                    # ----------------------------------------
                    nodeStates = self.model.X.flatten()

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # ----------------------------------------
                    # Allow SYMPTOMATIC individuals to self-seek tests
                    # regardless of cadence testing days
                    # ----------------------------------------
                    symptomaticSelection = []

                    if any(self.testing_compliance_symptomatic):

                        symptomaticPool = numpy.argwhere((self.testing_compliance_symptomatic == True)
                                                         & (nodeTestedInCurrentStateStatuses == False)
                                                         & (nodePositiveStatuses == False)
                                                         & ((nodeStates == self.model.I_sym) | (
                                                             nodeStates == self.model.Q_sym))
                                                         ).flatten()

                        numSymptomaticTests = min(
                            len(symptomaticPool), self.max_symptomatic_tests_per_day)

                        if len(symptomaticPool) > 0:
                            symptomaticSelection = symptomaticPool[numpy.random.choice(len(
                                symptomaticPool), min(numSymptomaticTests, len(symptomaticPool)), replace=False)]

                    # ----------------------------------------
                    # Test individuals randomly and via contact tracing
                    # on cadence testing days:
                    # ----------------------------------------

                    tracingSelection = []
                    randomSelection = []

                    if cadenceDayNumber in self.testingDays:

                        # ----------------------------------------
                        # Apply a designated portion of this day's tests
                        # to individuals identified by CONTACT TRACING:
                        # ----------------------------------------

                        tracingPool = self.tracingPoolQueue.pop(0)

                        if any(self.testing_compliance_traced):

                            numTracingTests = min(len(tracingPool), min(
                                self.tests_per_day - len(symptomaticSelection), self.max_tracing_tests_per_day))

                            for trace in range(numTracingTests):
                                traceNode = tracingPool.pop()
                                if ((nodePositiveStatuses[traceNode] == False)
                                        and (self.testing_compliance_traced[traceNode] == True)
                                        and (self.model.X[traceNode] != self.model.R)
                                        and (self.model.X[traceNode] != self.model.Q_R)
                                        and (self.model.X[traceNode] != self.model.H)
                                        and (self.model.X[traceNode] != self.model.F)):
                                    tracingSelection.append(traceNode)

                        # ----------------------------------------
                        # Apply the remainder of this day's tests to random testing:
                        # ----------------------------------------

                        if any(self.testing_compliance_random):

                            testingPool = numpy.argwhere((self.testing_compliance_random == True)
                                                         & (nodePositiveStatuses == False)
                                                         & (nodeStates != self.model.R)
                                                         & (nodeStates != self.model.Q_R)
                                                         & (nodeStates != self.model.H)
                                                         & (nodeStates != self.model.F)
                                                         ).flatten()

                            numRandomTests = max(min(
                                self.tests_per_day -
                                len(tracingSelection) -
                                len(symptomaticSelection),
                                len(testingPool)), 0)

                            testingPool_degrees = self.model.degree.flatten()[
                                testingPool]
                            testingPool_degreeWeights = numpy.power(testingPool_degrees,
                                                                    self.random_testing_degree_bias) / numpy.sum(
                                numpy.power(testingPool_degrees, self.random_testing_degree_bias))

                            if len(testingPool) > 0:
                                randomSelection = testingPool[numpy.random.choice(
                                    len(testingPool), numRandomTests, p=testingPool_degreeWeights, replace=False)]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # ----------------------------------------
                    # Perform the tests on the selected individuals:
                    # ----------------------------------------

                    selectedToTest = numpy.concatenate(
                        (symptomaticSelection, tracingSelection, randomSelection)).astype(int)

                    numTested = 0
                    numTested_random = 0
                    numTested_tracing = 0
                    numTested_symptomatic = 0
                    numPositive = 0
                    numPositive_random = 0
                    numPositive_tracing = 0
                    numPositive_symptomatic = 0
                    numIsolated_positiveGroupmate = 0

                    newTracingPool = []

                    newIsolationGroup_positive = []

                    for i, testNode in enumerate(selectedToTest):

                        self.model.set_tested(testNode, True)

                        numTested += 1
                        if i < len(symptomaticSelection):
                            numTested_symptomatic += 1
                        elif i < len(symptomaticSelection) + len(tracingSelection):
                            numTested_tracing += 1
                        else:
                            numTested_random += 1

                        # If the node to be tested is not infected, then the test is guaranteed negative,
                        # so don't bother going through with doing the test:
                        if self.model.X[testNode] == self.model.S or self.model.X[testNode] == self.model.Q_S:
                            pass
                        # Also assume that latent infections are not picked up by tests:
                        elif self.model.X[testNode] == self.model.E or self.model.X[testNode] == self.model.Q_E:
                            pass
                        elif (self.model.X[testNode] == self.model.I_pre or self.model.X[testNode] == self.model.Q_pre
                              or self.model.X[testNode] == self.model.I_sym or self.model.X[
                                  testNode] == self.model.Q_sym
                              or self.model.X[testNode] == self.model.I_asym or self.model.X[
                                  testNode] == self.model.Q_asym):

                            if self.test_falseneg_rate == 'temporal':
                                testNodeState = self.model.X[testNode][0]
                                testNodeTimeInState = self.model.timer_state[testNode][0]
                                if testNodeState in list(self.temporal_falseneg_rates.keys()):
                                    falseneg_prob = self.temporal_falseneg_rates[testNodeState][int(min(
                                        testNodeTimeInState, max(list(
                                            self.temporal_falseneg_rates[testNodeState].keys()))))]
                                else:
                                    falseneg_prob = 1.00
                            else:
                                falseneg_prob = self.test_falseneg_rate

                            if numpy.random.rand() < (1 - falseneg_prob):
                                # +++++++++++++++++++++++++++++++++++++++++++++
                                # The tested node has returned a positive test
                                # +++++++++++++++++++++++++++++++++++++++++++++
                                numPositive += 1
                                if i < len(symptomaticSelection):
                                    numPositive_symptomatic += 1
                                elif i < len(symptomaticSelection) + len(tracingSelection):
                                    numPositive_tracing += 1
                                else:
                                    numPositive_random += 1

                                # Update the node's state to the appropriate detected case state:
                                self.model.set_positive(testNode, True)

                                # ----------------------------------------
                                # Add this positive node to the isolation group:
                                # ----------------------------------------
                                if self.isolation_compliance_positive_individual[testNode]:
                                    newIsolationGroup_positive.append(testNode)

                                # ----------------------------------------
                                # Add the groupmates of this positive node to the isolation group:
                                # ----------------------------------------
                                if (self.isolation_groups is not None and any(
                                        self.isolation_compliance_positive_groupmate)):
                                    isolationGroupmates = next(
                                        (group for group in self.isolation_groups if testNode in group), None)
                                    for isolationGroupmate in isolationGroupmates:
                                        if isolationGroupmate != testNode:
                                            if self.isolation_compliance_positive_groupmate[isolationGroupmate]:
                                                numIsolated_positiveGroupmate += 1
                                                newIsolationGroup_positive.append(
                                                    isolationGroupmate)

                                # ----------------------------------------
                                # Add this node's neighbors to the contact tracing pool:
                                # ----------------------------------------
                                if (any(self.tracing_compliance) or any(
                                        self.isolation_compliance_positive_contact) or any(
                                        self.isolation_compliance_positive_contactgroupmate)):
                                    if self.tracing_compliance[testNode]:
                                        testNodeContacts = list(
                                            self.model.G[testNode].keys())
                                        numpy.random.shuffle(testNodeContacts)
                                        if self.num_contacts_to_trace is None:
                                            numContactsToTrace = int(
                                                self.pct_contacts_to_trace * len(testNodeContacts))
                                        else:
                                            numContactsToTrace = self.num_contacts_to_trace
                                        newTracingPool.extend(
                                            testNodeContacts[0:numContactsToTrace])

                    # Add the nodes to be isolated to the isolation queue:
                    self.isolationQueue_positive.append(
                        newIsolationGroup_positive)
                    self.isolationQueue_symptomatic.append(
                        newIsolationGroup_symptomatic)
                    self.isolationQueue_contact.append(
                        newIsolationGroup_contact)

                    # Add the nodes to be traced to the tracing queue:
                    self.tracingPoolQueue.append(newTracingPool)

                    print("\t" + str(numTested_symptomatic) + "\ttested due to symptoms  [+ " + str(
                        numPositive_symptomatic) + " positive (%.2f %%) +]" % (
                        numPositive_symptomatic / numTested_symptomatic * 100 if numTested_symptomatic > 0 else 0))
                    print("\t" + str(numTested_tracing) + "\ttested as traces        [+ " + str(
                        numPositive_tracing) + " positive (%.2f %%) +]" % (
                        numPositive_tracing / numTested_tracing * 100 if numTested_tracing > 0 else 0))
                    print("\t" + str(numTested_random) + "\ttested randomly         [+ " + str(
                        numPositive_random) + " positive (%.2f %%) +]" % (
                        numPositive_random / numTested_random * 100 if numTested_random > 0 else 0))
                    print("\t" + str(numTested) + "\ttested TOTAL            [+ " + str(
                        numPositive) + " positive (%.2f %%) +]" % (
                        numPositive / numTested * 100 if numTested > 0 else 0))

                    print("\t" + str(numSelfIsolated_symptoms) + " will isolate due to symptoms         (" +
                          str(numSelfIsolated_symptomaticGroupmate) + " as groupmates of symptomatic)")
                    print("\t" + str(numPositive) + " will isolate due to positive test    (" +
                          str(numIsolated_positiveGroupmate) + " as groupmates of positive)")
                    print("\t" + str(numSelfIsolated_positiveContact) + " will isolate due to positive contact (" +
                          str(numSelfIsolated_positiveContactGroupmate) + " as groupmates of contact)")

                    # ----------------------------------------
                    # Update the status of nodes who are to be isolated:
                    # ----------------------------------------

                    numIsolated = 0

                    isolationGroup_symptomatic = self.isolationQueue_symptomatic.pop(
                        0)
                    for isolationNode in isolationGroup_symptomatic:
                        self.model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    isolationGroup_contact = self.isolationQueue_contact.pop(0)
                    for isolationNode in isolationGroup_contact:
                        self.model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    isolationGroup_positive = self.isolationQueue_positive.pop(
                        0)
                    for isolationNode in isolationGroup_positive:
                        self.model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    print("\t" + str(numIsolated) + " entered isolation")

                    self.numPosTseries[self.model.t] = numPositive * \
                        self.model.cur_scale

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        interventionInterval = (self.interventionStartTime, self.model.t)

        return interventionInterval
