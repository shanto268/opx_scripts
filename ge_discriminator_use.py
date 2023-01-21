# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# this file works with version 0.7.411 & gateway configuration of a single controller #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator import TwoStateDiscriminator
from configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 2), ("con1", 2, "con1", 1)], latency=230, noisePower=0.07 ** 2
    )
)

rr_qe1 = 'rr1a'
rr_qe2 = 'rr2a'
# training + testing to get fidelity:
qmm = QuantumMachinesManager()
discriminator1 = TwoStateDiscriminator(qmm, config, True, rr_qe1, f'ge_disc_params_{rr_qe1}.npz', lsb=True)
discriminator2 = TwoStateDiscriminator(qmm, config, True, rr_qe2, f'ge_disc_params_{rr_qe2}.npz', lsb=True)

wait_time = 10
N = 100

with program() as test_program:
    n = declare(int)
    res = declare(bool)
    res2 = declare(bool)
    I = declare(fixed)
    I2 = declare(fixed)
    Q = declare(fixed)
    Q2 = declare(fixed)

    res_st = declare_stream()
    res_st2 = declare_stream()
    I_st = declare_stream()
    I_st2 = declare_stream()
    Q_st = declare_stream()
    Q_st2 = declare_stream()

    with for_(n, 0, n < N, n + 1):
        wait(wait_time, rr_qe1, rr_qe2)
        align(rr_qe1, rr_qe2)
        discriminator1.measure_state("readout_pulse_g", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)
        discriminator2.measure_state("readout_pulse_g", "out1", "out2", res2, I=I2, Q=Q2)
        save(res2, res_st2)
        save(I2, I_st2)
        save(Q2, Q_st2)

        wait(wait_time, rr_qe1, rr_qe2)
        align(rr_qe1, rr_qe2)
        discriminator1.measure_state("readout_pulse_e", "out1", "out2", res, I=I, Q=Q)
        save(res, res_st)
        save(I, I_st)
        save(Q, Q_st)
        discriminator2.measure_state("readout_pulse_e", "out1", "out2", res2, I=I2, Q=Q2)
        save(res2, res_st2)
        save(I2, I_st2)
        save(Q2, Q_st2)

        seq0 = [0, 1] * N

    with stream_processing():
        res_st.save_all('res')
        I_st.save_all('I')
        Q_st.save_all('Q')
        res_st2.save_all('res2')
        I_st2.save_all('I2')
        Q_st2.save_all('Q2')

    for i in range(16):
        play(f'short_ramsey{i}', 'qubit')
        measure('readout', 'rr', None, ...)
        save(I, I_st)
        save(Q, Q_st)

job = qmm.simulate(config, test_program, simulate=simulation_config)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']
I = result_handles.get('I').fetch_all()['value']
Q = result_handles.get('Q').fetch_all()['value']

res2 = result_handles.get('res2').fetch_all()['value']
I2 = result_handles.get('I2').fetch_all()['value']
Q2 = result_handles.get('Q2').fetch_all()['value']

plt.figure()
plt.hist(I[np.array(seq0) == 0], 50)
plt.hist(I[np.array(seq0) == 1], 50)
plt.plot([discriminator1.get_threshold()] * 2, [0, 60], 'g')
plt.show()

plt.figure()
plt.hist(I[np.array(seq0) == 0], 50)
plt.hist(I[np.array(seq0) == 1], 50)
plt.plot([discriminator2.get_threshold()] * 2, [0, 60], 'g')
plt.show()

theta = np.linspace(0, 2 * np.pi, 100)

plt.figure()
plt.plot(I, Q, '.')
for i in range(discriminator1.num_of_states):
    a = discriminator1.sigma[i] * np.cos(theta) + discriminator1.mu[i][0]
    b = discriminator1.sigma[i] * np.sin(theta) + discriminator1.mu[i][1]
    plt.plot([discriminator1.mu[i][0]], [discriminator1.mu[i][1]], 'o')
    plt.plot(a, b)
plt.axis('equal')

plt.figure()
plt.plot(I2, Q2, '.')
for i in range(discriminator2.num_of_states):
    a = discriminator2.sigma[i] * np.cos(theta) + discriminator2.mu[i][0]
    b = discriminator2.sigma[i] * np.sin(theta) + discriminator2.mu[i][1]
    plt.plot([discriminator2.mu[i][0]], [discriminator2.mu[i][1]], 'o')
    plt.plot(a, b)
plt.axis('equal')

p_s = np.zeros(shape=(2, 2))
for i in range(2):
    res_i = res[np.array(seq0) == i]
    p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

labels = ['g', 'e']
plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt='g', cmap='Blues')

ax.set_xlabel('Predicted labels')
ax.set_ylabel('Prepared labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

plt.show()

p_s2 = np.zeros(shape=(2, 2))
for i in range(2):
    res_i = res2[np.array(seq0) == i]
    p_s2[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

labels = ['g', 'e']
plt.figure()
ax2 = plt.subplot()
sns.heatmap(p_s2, annot=True, ax=ax2, fmt='g', cmap='Blues')

ax2.set_xlabel('Predicted labels')
ax2.set_ylabel('Prepared labels')
ax2.set_title('Confusion Matrix')
ax2.xaxis.set_ticklabels(labels)
ax2.yaxis.set_ticklabels(labels)

plt.show()
