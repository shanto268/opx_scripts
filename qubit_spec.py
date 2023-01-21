from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_NIST_Q2 import *
import matplotlib.pyplot as plt
import numpy as np
from analysis_functions import convert_V_to_dBm
#from VISAdrivers import LO845m as LO
import plot_functions as pf
###################
# The QUA program #
###################

f_min = 40e6
f_max = 60e6
df = 0.05e6
freq = np.arange(f_min, f_max + df/2, df)
qLO = qb_LO
n_avg = 1000
saturation_dur = 12500 #in clock cycles
wait_period = 50000 # in clock cycles 

with program() as qubit_spec:

    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    f = declare(int)
    
    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):
            update_frequency("qubit", f)
            play("const","qubit",duration=saturation_dur)
            align("qubit","rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                    dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            wait(wait_period, "rr")

    with stream_processing():
        I_st.buffer(len(freq)).average().save('I')
        Q_st.buffer(len(freq)).average().save('Q')

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
job = qm.execute(qubit_spec)
res_handles = job.result_handles
# plt.close('all')
res_handles.wait_for_all_values()
#a = plt.figure()
#b = plt.figure()
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
I_handle.wait_for_values(1)
Q_handle.wait_for_values(1)
#while(I_handle.is_processing()):
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
    # mag = np.sqrt(I**2 + Q**2)
    # plt.figure(a)
    # plt.clf()
    # plt.plot(freq+qLO, 1e3*mag)
    # # plt.plot(freqs+qLO,mag)
    # plt.title('qubit spectroscopy analysis')
    # plt.xlabel("freq (GHz)")
    # plt.ylabel("Amplitude (V)")
    # plt.pause(0.1)
    
    #phase = np.unwrap(np.angle(I+1j*Q))
    #plt.figure(b)
    #plt.clf()
    # plt.plot(freqs+qLO, 1e3*Q)
    # plt.ylabel("Amplitude (mV)")
    #plt.title('qubit spectroscopy analysis')
    #plt.plot(freq+qb_LO,phase)
    #plt.xlabel("freq (GHz)")
    #plt.ylabel("Phase (rad)")
    #plt.pause(0.1)
    
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

#plt.close('all')

# plt.figure()
# plt.plot(freqs+qLO,1e3*I)
# plt.title('qubit spectroscopy analysis')
# plt.xlabel("freq (GHz)")
# plt.ylabel("Power (dBm)")
mag = np.sqrt(I**2 + Q**2)
plt.figure()
plt.plot(freq+qLO,mag)
plt.title('qubit spectroscopy analysis')
plt.xlabel("freq (GHz)")
plt.ylabel("mag (V)")

angle = np.unwrap(np.angle(I+1j*Q))
plt.figure()
plt.plot(freq+qLO,angle)
plt.title('qubit spectroscopy analysis')
plt.xlabel("freq (GHz)")
plt.ylabel("Phase (rad)")

#pf.spec_plot(freq=freq+qb_LO,I=I, Q=Q,res_type='r', readout_power=-56,qubit_drive_amp=amp_q)
