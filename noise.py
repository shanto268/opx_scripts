from configuration2 import config
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt

qmm = QuantumMachinesManager()

qm = qmm.open_qm(config)
tau_0 = 50
tau_max = 1250
pi_len = 16

with program() as noise:
    tau = declare(int)
    
    with infinite_loop_():
        with for_(tau, tau_0, tau < tau_max, tau+200):
            play("pi2","qubit")
            align()
            play("tel_noise", "noise_1st", truncate=tau)
            wait(pi_len-1,"noise_2nd","noise_2nd_cancel")
            wait(tau,"qubit")
            reset_frame("noise_2nd")
            reset_frame("noise_2nd_cancel")
            play("tel_noise","noise_2nd",truncate=2*tau)
            play("tel_noise"*amp(-1),"noise_2nd_cancel",truncate=tau)
            play("pi","qubit")
            # wait(6, "noise_2nd", "noise_2nd_cancel")
            # play("CW", "noise_2nd", truncate=20)
            # play("-CW", "noise_2nd_cancel", truncate=10)
            wait(tau,'qubit')
            play("pi2","qubit")
            wait(50)

job = qmm.simulate(config, noise, simulate=SimulationConfig(5000))
samples = job.get_simulated_samples()
# samples.con1.plot(analog_ports=['5','6','7','8'])
# samples.con1.plot(analog_ports=['3','5','7'])
samples.con1.plot()
plt.figure()
plt.plot(config['waveforms']['telegraph_noise']['samples'])

# job = qm.execute(noise)
