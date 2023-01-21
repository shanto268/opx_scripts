from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_NIST_Q2 import *

import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

a_min = 0.01
a_max = 0.8
da = 0.01
amps = np.arange(a_min, a_max + da/2, da)
n_avg = 5000
# detun = 2e6

with program() as power_rabi:

    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    a = declare(fixed)
    # update_frequency('qubit', (5.2895e9-5.23e9)+detun)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(a, a_min, a < a_max + da/2, a + da):
            play("gauss"*amp(a), "qubit",duration=pi_len)  
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4) 
           # play("gauss"*amp(a), "qubit",duration=pi_len/4)  
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4) 
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4)   
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4) 
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4)  
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4) 
          #  play("gauss"*amp(a), "qubit",duration=pi_len/4)                                          
            align("qubit", "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            wait(50000, "qubit")

    with stream_processing():
        I_st.buffer(len(amps)).average().save('I')
        Q_st.buffer(len(amps)).average().save('Q')

######################################
# Open Communication with the Server #
######################################
qmm = QuantumMachinesManager()  # Reach OPX's IP address

####################
# Simulate Program #
####################
qm = qmm.open_qm(config)

job = qm.execute(power_rabi)
res_handles = job.result_handles
res_handles.wait_for_all_values()
# a = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
# I_handle.wait_for_values(1)
# Q_handle.wait_for_values(1)
# while(I_handle.is_processing()):
#     I = I_handle.fetch_all()
#     Q = Q_handle.fetch_all()
#     # mag = np.sqrt(I**2 + Q**2)
#     plt.figure(a)
#     plt.plot(amps, 1e3*I)
#     # plt.plot(amps, I)
#     # plt.plot(amps, Q)
#     # plt.title('qubit spectroscopy analysis')
#     plt.xlabel("Drive Amplitude a")
#     plt.ylabel("Amplitude (mV)")
#     plt.pause(0.1)
#     plt.clf()

I = I_handle.fetch_all()
Q = Q_handle.fetch_all()    
plt.figure()
plt.plot(amps,I)
# plt.figure()
# plt.plot(amps,Q)
# plt.figure()
# plt.plot(amps,mag)