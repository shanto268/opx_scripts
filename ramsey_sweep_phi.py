# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:52:03 2021

@author: lfl
"""
"""
T2.py: Single qubit T2 decay time
Author: Steven Frankel, Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

import matplotlib.pyplot as plt
from configuration_NIST_Q2 import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
from scipy.optimize import curve_fit
import time
import plot_functions as pf

QMm = QuantumMachinesManager()


# Create a quantum machine based on the configuration.



# measurement parameters
#######################
phi_min = 0
phi_max = 1
num_of_points = 101
dphi = (phi_max-phi_min)/ (num_of_points-1)
tempdphi =(dphi)*(2**16) #Need to make sure the phi steps are in 2^16 precision
dphi = np.round(tempdphi)/(2**16)
phi_arr = np.arange(phi_min, phi_max, dphi)# Number of timesteps
N_max = 1000
detun = 0
t_delay = 4 # in ns
with program() as ramsey_sweep_phi:
    I = declare(fixed)# QUA variables declaration
    Q = declare(fixed)
    Nrep = declare(int)
    phi = declare(float,value=0)
    I_stream = declare_stream()
    Q_stream = declare_stream()
    Nrep_stream = declare_stream()
    update_frequency('qubit', (qbFreq-qb_LO)+detun)
    # T2
    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        save(Nrep,Nrep_stream)    
        with for_(phi, phi_min, phi <= phi_max, phi + dphi):
            play("pi_half", "qubit")
            wait(t_delay, "qubit")
            frame_rotation_2pi(phi, 'qubit')
            play("pi_half", "qubit")
            reset_frame('qubit')
            align("qubit","rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_stream)
            save(Q, Q_stream)
            wait(50000,"qubit")
            # assign(phi,phi+0.1)

    with stream_processing():
        I_stream.buffer(len(phi_arr)).average().save("I")
        Q_stream.buffer(len(phi_arr)).average().save("Q")
        Nrep_stream.save('n')


qmm = QuantumMachinesManager()  # Reach OPX's IP address
start = time.time()
qm = qmm.open_qm(config)
job = qm.execute(ramsey_sweep_phi)

res_handles = job.result_handles
res_handles.wait_for_all_values()
# a = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
# n_handle = res_handles.get('n')
# I_handle.wait_for_values(1)
# Q_handle.wait_for_values(1)
# n_handle.wait_for_values(1)
# while(I_handle.is_processing()):
#     I = I_handle.fetch_all()
#     Q = Q_handle.fetch_all()
#     n = n_handle.fetch_all()
#     mag = np.sqrt(I**2 + Q**2)
#     plt.figure(a)
#     plt.clf()
#     plt.plot(t_arr*4, 1e3*I)
#     # plt.title('qubit spectroscopy analysis')
#     plt.xlabel("time (ns)")
#     plt.ylabel("Amplitude (mV)")
#     plt.title('n = %d'%(n))
#     plt.pause(0.1)

I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
end = time.time()
print(end-start)    
# plt.title('Qubit IF %.2f MHz'%((qbFreq-qb_LO+detun)*1e-6))
pf.pulse_plot(sequence="ramsey", t_vector = phi_arr, y_vector=I,dt=dphi,qubitDriveFreq=qb_LO + qb_IF +detun,amp_q=pi_amp)
