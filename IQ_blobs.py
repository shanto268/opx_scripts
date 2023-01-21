from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import config
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

with program() as IQ_blobs:

    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    I_st_exc = declare_stream()
    Q_st_exc = declare_stream()
    with for_(n, 0, n < 2000, n + 1):
            wait(50000, "qubit")
            align("qubit", "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_minus_sin", "out2", I),
                    dual_demod.full("integW_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            align('qubit','rr')
            wait(50000, "qubit")
            play("pi"*amp(1.02), "qubit")
            align("qubit", "rr")
            measure("readout", "rr", None,
                    dual_demod.full("integW_cos", "out1", "integW_minus_sin", "out2", I),
                    dual_demod.full("integW_sin", "out1", "integW_cos", "out2", Q))
            save(I, I_st_exc)
            save(Q, Q_st_exc)
            
    with stream_processing():
        I_st.save_all('I')
        Q_st.save_all('Q')
        I_st_exc.save_all('I_exc')
        Q_st_exc.save_all('Q_exc')

######################################
# Open Communication with the Server #
######################################
qmm = QuantumMachinesManager()

####################
# Simulate Program #
####################
# simulation_config = SimulationConfig(
#                     duration=5000,
#                     simulation_interface=LoopbackInterface([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]))

# job = qmm.simulate(config, IQ_blobs, simulation_config)
qm = qmm.open_qm(config)
job = qm.execute(IQ_blobs)
res_handles = job.result_handles
res_handles.wait_for_all_values()
I = res_handles.get("I").fetch_all()['value']
Q = res_handles.get("Q").fetch_all()['value']
I_exc = res_handles.get("I_exc").fetch_all()['value']
Q_exc = res_handles.get("Q_exc").fetch_all()['value']

# plt.figure()
# plt.title('IQ_blobs sequence')
# job.get_simulated_samples().con1.plot()
# plt.xlabel("time [ns]")
# plt.ylabel("DAC [V]")

# print(len(I))
# print(len(Q))
plt.figure()
plt.plot(I,Q,'.',alpha=0.5)
plt.axis('equal')
plt.plot(I_exc,Q_exc,'.',alpha=0.5)

# plt.hist2d(I, Q)
