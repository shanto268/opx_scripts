# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from configuration import *

t_min = 10
t_max = 300  # Maximum pulse duration (in clock cycles, 1 clock cycle =4 ns)
dt = 1  # timestep
t_arr = np.arange(t_min, t_max + dt/2, dt)# Number of timesteps
N_max = 2000

qmm = QuantumMachinesManager()  # Reach OPX's IP address

with program() as timeRabiProg:  # Time Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    t = declare(int)  # Sweeping parameter over the set of durations
    Nrep = declare(int)  # Number of repetitions of the experiment
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()
    t_stream = declare_stream()
    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        with for_(t, t_min, t <= t_max, t + dt):  # Sweep from 0 to 100 *4 ns the pulse duration
            play("gauss", "qubit", duration=t)
            align("qubit", "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_minus_sin", "out2", I),
                    dual_demod.full("integW_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_stream)
            save(Q, Q_stream)
            wait(50000,"qubit")

    with stream_processing():
        I_stream.buffer(len(t_arr)).average().save("I")
        Q_stream.buffer(len(t_arr)).average().save("Q")

qm = qmm.open_qm(config)
job = qm.execute(timeRabiProg)
res_handles = job.result_handles
# res_handles.wait_for_all_values()
a = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I_handle.wait_for_values(1)
Q_handle.wait_for_values(1)
while(I_handle.is_processing()):
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    mag = np.sqrt(I**2 + Q**2)
    plt.figure(a)
    plt.clf()
    plt.plot(t_arr*4,mag)
    # plt.title('qubit spectroscopy analysis')
    # plt.xlabel("freq (GHz)")
    # plt.ylabel("Amplitude")
    plt.pause(0.1)
    
    # plt.plot(freqs+qLO, I)
    # plt.plot(freqs+qLO, Q)
    