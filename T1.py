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

t_min = 50
t_max = 20000  # Maximum pulse duration (in clock cycles, 1 clock cycle =4 ns)
dt = 100  # timestep
t_arr = np.arange(t_min, t_max + dt/2, dt) # Number of timesteps
N_max = 1000

with program() as T1:

    n = declare(int)
    I = declare(fixed)
    res = declare(bool)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    t = declare(int) # added by dmh
    Nrep_stream = declare_stream()
    with for_(n, 0, n < N_max, n + 1):
        save(n,Nrep_stream)
        with for_(t, t_min, t <= t_max, t + dt):
            wait(25000, "qubit")
            # with while_(res):
                # play("pi", "qubit")
                # align("qubit", "rr")
                # discriminator.measure_state("readout", "out1", "out2", res)
                
            # play("pi", "qubit", condition=res)
            
            play("pi", "qubit")
            align("qubit", "rr")
            wait(t, 'rr')
            # discriminator.measure_state("readout", "out1", "out2", res, I=I, Q=Q)
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(t_arr)).average().save("I")
        Q_st.buffer(len(t_arr)).average().save("Q")
        Nrep_stream.save('n')

######################################
# Open Communication with the Server #
######################################

####################
# Simulate Program #
####################
# simulation_config = SimulationConfig(
#                     duration=5000,
#                     simulation_interface=LoopbackInterface([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]))

# job = qmm.simulate(config, IQ_blobs, simulation_config)
# job = qm.execute(T1)
# res_handles = job.result_handles
# res_handles.wait_for_all_values()
# I = res_handles.get("I").fetch_all()['value']
# Q = res_handles.get("Q").fetch_all()['value']

# # plt.figure()
# # plt.title('IQ_blobs sequence')
# # job.get_simulated_samples().con1.plot()
# # plt.xlabel("time [ns]")
# # plt.ylabel("DAC [V]")

# # print(len(I))
# # print(len(Q))
# plt.figure()
# plt.hist2d(I, Q)

qmm = QuantumMachinesManager()  # Reach OPX's IP address
qm = qmm.open_qm(config)
job = qm.execute(T1)
res_handles = job.result_handles
# res_handles.wait_for_all_values()
a = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
n_handle = res_handles.get("n")
I_handle.wait_for_values(1)
Q_handle.wait_for_values(1)
n_handle.wait_for_values(1)
plt.close('all')
while(I_handle.is_processing()):
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    n = n_handle.fetch_all()
    mag = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.angle(I/Q))
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
    
