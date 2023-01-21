from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_NIST_Q2 import config,rr_LO
import matplotlib.pyplot as plt
import numpy as np
from analysis_functions import *

###################
# The QUA program #
###################

f_min = -35e6
f_max = -25e6
df = 0.05e6
freqs = np.arange(f_min, f_max + df/2, df)
n_avg = 1000
readLO = rr_LO

with program() as rr_spec:

    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    f = declare(int)

    # I_test = declare(fixed)
    # I_test_st = declare_stream()
    # Q_test = declare(fixed)
    # Q_test_st = declare_stream()    

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):
            update_frequency("rr", f)
            wait(25000, "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            # wait(25000, "rr")
            # measure("readout", "rr", None,
            #         demod.full("integW_minus_sin",I_test,"out2"),
            #         demod.full("integW_cos",Q_test,"out2"))
            # save(I_test,I_test_st)
            # save(Q_test,Q_test_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')
        # I_test_st.buffer(len(freqs)).average().save("I_test")
        # Q_test_st.buffer(len(freqs)).average().save("Q_test")

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
job = qm.execute(rr_spec)
res_handles = job.result_handles
res_handles.wait_for_all_values()
I2 = res_handles.get("I").fetch_all()
Q2 = res_handles.get("Q").fetch_all()

# I_test2 = res_handles.get("I_test").fetch_all()
# Q_test2 = res_handles.get("Q_test").fetch_all()
# plt.figure()
# plt.title('resonator spectroscopy sequence')
# job.get_simulated_samples().con1.plot()
# plt.xlabel("time [ns]")
# plt.ylabel("DAC [V]")

# plt.figure()
# plt.title('resonator spectroscopy analysis')
# plt.plot(I, Q, '.')
# plt.axis('equal')
# plt.xlabel("I")
# plt.ylabel("Q")

#plt.figure()
#plt.plot(freqs+readLO,convert_mV_to_dBm(np.sqrt(I**2 + Q**2)))
#plt.xlabel('freq (GHz)')
#plt.ylabel('Mag (V)')
#plt.title('Readout Pulse Amplitude = 10mV')

#plt.figure()
#plt.plot(freqs+readLO,(np.sqrt(I**2 + Q**2)))
#plt.ylim((-50,-5))
#plt.title("dual demod")
#plt.xlabel('freq (GHz)')
#plt.ylabel('Log Mag (mV)')
#plt.title('Readout Pulse Amplitude = 10mV')

#plt.figure()
#plt.plot(freqs+readLO,(np.sqrt(I_test**2 + Q_test**2)))
#plt.ylim((-50,-5))
#plt.title("single demod test, out 1")
#plt.xlabel('freq (GHz)')
#plt.ylabel('Log Mag (mV)')
#plt.title('Readout Pulse Amplitude = 10mV')

# plt.plot(freqs+readLO,I)
# plt.plot(freqs+readLO,Q)
# plt.figure()
# plt.plot(freqs+readLO,np.unwrap(np.angle(I+1j*Q)))
# plt.xlabel('freq (GHz)')
# plt.ylabel('Phase (rad)')