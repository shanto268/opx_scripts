from configuration import *
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt


def FeedbackAmplitude(I, Q):
    assign(A, I + Q + r.rand_fixed() - s)


def FeedbackFrequency(I, Q):
    assign(freq, Cast.mul_int_by_fixed(5e6, I + Q + r.rand_fixed() + s))


with program() as analog_feedback:
    I = declare(fixed)
    Q = declare(fixed)
    r = Random()
    s = declare(fixed, value=0.5)
    A = declare(fixed)
    freq = declare(int)
    n = declare(int)

    play("Drive", "qubit")
    play("pre-readout", "rr")
    with for_(n, 0, n < 100, n + 1):
        measure("Readout", "rr", None, demod.full("cos", I), demod.full("sin", Q))
        FeedbackFrequency(I, Q)
        FeedbackAmplitude(I, Q)
        play("Drive" * amp(A), "qubit")
        update_frequency("qubit", freq, keep_phase=True)
        FeedbackFrequency(I, Q)
        FeedbackAmplitude(I, Q)
        play("pre-readout" * amp(A), "rr")
        update_frequency("rr", freq, keep_phase=True)
        amp()
    wait(100)

qmm = QuantumMachinesManager()
qmm.close_all_quantum_machines()
qm = qmm.open_qm(config)
job = qm.simulate(analog_feedback, SimulationConfig(800))

samples = job.get_simulated_samples()
# samples.con1.plot()
res = job.result_handles
# I = res.I.fetch_all()

s1 = samples.con1.analog['3']
s2 = samples.con1.analog['4']
s3 = samples.con1.analog['6']

[f, (ax1, ax2)] = plt.subplots(nrows=2, ncols=1)
ax1.plot(s1)
ax1.plot(s2)
ax2.plot(s3)
ax2.set_xlabel('Time [ns]')
