# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:16:20 2021
play pulses and check on oscilloscope 
@author: lfl
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from configuration import *
t_min = 4
t_max = 250 # Maximum pulse duration (in clock cycles, 1 clock cycle = 4 ns)
dt = 1  # timestep
t_arr = np.arange(t_min, t_max +dt/2, dt)# Number of timesteps
N_max = 1
with program() as timeRabiProg:
    n = declare(int)
    t = declare(int)  # Sweeping parameter over the set of durations
    Nrep = declare(int)  # Number of repetitions of the experiment
    Nrep_stream = declare_stream()
    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        save(Nrep,Nrep_stream)
        with for_(t, t_min, t <= t_max, t + dt):  # Sweep from 0 to 100 *4 ns the pulse duration
            with for_(n, 0, n<1000, n+1):
                play("gauss", "qubit", duration=t)

    with stream_processing():
        Nrep_stream.save('n')

qmm = QuantumMachinesManager()  # Reach OPX's IP address
qm = qmm.open_qm(config)

job = qm.execute(timeRabiProg)
res_handles = job.result_handles
n_handle = res_handles.get("n")

# job = qm.simulate(timeRabiProg, SimulationConfig(1000))
# samples = job.get_simulated_samples()
# samples.con1.plot()
