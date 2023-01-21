# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:29:59 2021

@author: lfl
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_NIST_Q2 import config
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################


with program() as play_pulses:
    with infinite_loop_():
        play("readout", "rr", duration=100)
        play("const", 'qubit',duration=100)

    

######################################
# Open Communication with the Server #
######################################
qmm = QuantumMachinesManager()

####################
# Simulate Program #
####################
# simulation_config = SimulationConfig(
#                     duration=90000,
#                     simulation_interface=LoopbackInterface([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]))

# job = qmm.simulate(config, rr_spec, simulation_config)
qm = qmm.open_qm(config)
job = qm.execute(play_pulses)

