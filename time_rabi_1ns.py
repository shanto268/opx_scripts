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
from configuration_NIST_Q2 import *
from analysis_functions import convert_V_to_dBm
import plot_functions as pf

t_min = 1
t_max = 400 # Maximum pulse duration (in clock cycles, 1 clock cycle = 4 ns)
dt = 2  # timestep
t_arr = np.arange(t_min, t_max +dt/2, dt)# Number of timesteps
N_max = 2000
detun = 0
theta = 0
ampl = 0.2

with program() as timeRabiProg:  # Time Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    t = declare(int)  # Sweeping parameter over the set of durations
    Nrep = declare(int)  # Number of repetitions of the experiment
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()
    t_stream = declare_stream()
    Nrep_stream = declare_stream()
    update_frequency('qubit', (qbFreq-qb_LO)+detun)
    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        save(Nrep,Nrep_stream)
        with for_(t, t_min, t <= t_max, t + dt):  # Sweep from 0 to 100 *4 ns the pulse duration
            play("gauss"*amp(ampl), "qubit", duration=t)
            align("qubit", "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_stream)
            save(Q, Q_stream)            
            wait(50000,"qubit")

    with stream_processing():
        I_stream.buffer(len(t_arr)).average().save("I")
        Q_stream.buffer(len(t_arr)).average().save("Q")
        Nrep_stream.save('n')
'''---------------------------------------------------------------------'''
qmm = QuantumMachinesManager()  # Reach OPX's IP address
qm = qmm.open_qm(config)
job = qm.execute(timeRabiProg)
res_handles = job.result_handles
res_handles.wait_for_all_values()
# a = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
# n_handle = res_handles.get("n")
# I_handle.wait_for_values(1)
# Q_handle.wait_for_values(1)
# n_handle.wait_for_values(1)
# plt.close('all')
# while(I_handle.is_processing()):
#     I = I_handle.fetch_all()
#     Q = Q_handle.fetch_all()
#     n = n_handle.fetch_all()
#     mag = np.sqrt(I**2 + Q**2)
#     phase = np.unwrap(np.angle(I/Q))
#     plt.figure(a)
#     plt.clf()
#     # plt.plot(t_arr*4, 1e3*mag)
#     #plt.plot(t_arr*4, phase)
#     plt.plot(t_arr*4,1e3*I)
#     # plt.title('qubit spectroscopy analysis')
#     plt.xlabel("time (ns)")
#     #plt.ylabel("Amplitude (mV)")
#     plt.title('n = %d' %(n))
#     plt.pause(0.1)
    
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
# plt.close('all')

pf.pulse_plot(sequence="t_rabi", t_vector = 4*t_arr, y_vector=I,dt=4*dt,qubitDriveFreq=qb_LO + qb_IF +detun,amp_q=gauss_amp)    
# pf.pulse_plot(sequence="t_rabi", t_vector = 4*t_arr, y_vector=Q,dt=dt,qubitDriveFreq=qb_LO + qb_IF,amp_q=gauss_amp)
# plt.title('Qubit Drive Amplitude %.1f mV\nReadout Amplitude %.1f mV'%(1e3*gauss_amp,1e3*amp_r))
#     # plt.plot(freqs+qLO, I)
    # plt.plot(freqs+qLO, Q)
    