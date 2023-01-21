from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qm import LoopbackInterface
from qm.qua import *
from examples._config import config
import matplotlib.pyplot as plt

qmm = QuantumMachinesManager()

##
# ## Example 1 - No inputs

with program() as prog1:
        a = declare(fixed)
        b = declare(fixed, value=0.2)
        c = declare(fixed, value=0.4)

        play("measurement", "qe1")
        play("marker", "qeDig")

        with for_(a, 0.2, a < 0.9, a + 0.1):
                play("measurement" * amp(a), "qe1")

        assign(c, c + b*1.5)
        save(c, "c")

# simulate program
job = qmm.simulate(config, prog1, SimulationConfig(
        duration=1000,                  # duration of simulation in units of 4ns
        include_analog_waveforms=True,    # include analog waveform names
        include_digital_waveforms=True    # include digital waveform names
))

# get DAC and digital samples
samples = job.get_simulated_samples()

# plot all ports:
samples.con1.plot()

# another way, plot analog output 1 and digital output 9:
# plt.figure()
# plt.plot(samples.con1.analog["1"])
# plt.plot(samples.con1.digital["9"])
# plt.legend(("analog 1", "digital 9"))
# plt.xlabel("Time [ns]")
# plt.ylabel("Signal [V]")

# print waveform names and times
print(job.simulated_analog_waveforms())
print(job.simulated_digital_waveforms())

# get results
res = job.result_handles
c = res.variable_results.c.values
print(c)

##
# ## Example 2 - Loopback inputs

with program() as prog2:
        I = declare(fixed)
        Q = declare(fixed)
        f = declare(int)
        a = declare(fixed)

        with for_(a, 0.1, a < 0.4, a + 0.1):
                with for_(f, 50e6, f < 92e6, f + 1e6):
                        update_frequency("qe1", f)
                        measure("measurement"*amp(a), "qe1", "adc", demod.full('integW1', I), demod.full('integW2', Q))
                        save(I, "I")
                        save(Q, "Q")


# simulate program
job = qmm.simulate(config, prog2, SimulationConfig(
        duration=40000,
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])    # loopback from output 1 to input 1
))


# get results
res = job.get_results()
adc = res.raw_results.input1.values
ts = res.raw_results.input1.ts_nsec
I = res.variable_results.I.values
Q = res.variable_results.Q.values

plt.figure()
plt.plot(ts, adc)
plt.xlabel("Time [ns]")
plt.ylabel("ADC")

plt.figure()
plt.plot(I, Q, '.')
plt.xlabel("I")
plt.xlabel("Q")

##
# ## Example 3:

with program() as prog3:
        d = declare(int)

        with for_(d, 10, d <= 100, d + 10):
                play("measurement", "qe1")
                play("measurement", "qe2", duration=d)
                wait(50, "qe1")


# simulate program
job = qm.simulate(prog3, SimulationConfig(
        duration=1700,                  # duration of simulation in units of 4ns
))


# get DAC and digital samples
samples = job.get_simulated_samples()

# plot analog ports 1 and 3:
samples.con1.plot(analog_ports={'1', '3'}, digital_ports={})

# another way:
# plt.figure()
# plt.plot(samples.con1.analog["1"], "-")
# plt.plot(samples.con1.analog["3"], "--")
# plt.legend(("analog 1", "analog 3"))
# plt.xlabel("Time [ns]")
# plt.ylabel("Signal [V]")