from configuration2 import *
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt

qmm = QuantumMachinesManager()

qm = qmm.open_qm(config)
tau_0 = 100
tau_max = 500
dt = 100
pi_len = 16

# with program() as noise:

#     tau = declare(int)
#     tau_2nd = declare(int)
#     tau_st = declare_stream()


#     with for_(tau, tau_0, tau < tau_max, tau + dt):

#         # Do math calculations before playing pulse so that
#         # is not at the same time two-real time calculations
#         # at the same time
#         assign(tau_2nd, 2*tau)

#         play("pi2", "qubit")
#         align()

#         # play noise in the first half of the echo
#         reset_phase('noise_1st')
#         play("tele_noise", "noise_1st", truncate=tau)

#         # play pi pulse
#         wait(tau, 'qubit')
#         play('pi', 'qubit')

#         # play noise in the second part of the echo
#         wait(pi_len, 'noise_2nd', 'noise_2nd_cancel')
#         reset_phase('noise_2nd')
#         reset_phase('noise_2nd_cancel')
#         play('tele_noise', 'noise_2nd', truncate=tau_2nd)
#         play('tele_noise'*amp(-1.0), 'noise_2nd_cancel', truncate=tau)

#         # play pi/2 pulse
#         wait(tau, 'qubit')
#         play('pi2', 'qubit')

#         save(tau, tau_st)
#         wait(20)

#     with stream_processing():
#         tau_st.save_all('tau')

with program() as noise:

    tau = declare(int)
    tau_st = declare_stream()
    tau_2nd = declare(int)
    tau_2nd_cancel = declare(int)

    # frame rotation calculation before for_ loop starts
    frame_rotation_2pi(qubit_IF*pi_len*4/(2*np.pi), 'noise_2nd', 'noise_2nd_cancel')

    with for_(tau, tau_0, tau < tau_max, tau + dt):

        # Do math calculations before playing pulse so that
        # is not at the same time two-real time calculations
        # at the same time
        assign(tau_2nd, 2*tau + pi_len)
        assign(tau_2nd_cancel, tau + pi_len)

        # play pi/2 pulse
        play("pi2", "qubit")
        align()

        # play noise in the first half of the echo
        play("tele_noise", "noise_1st", truncate=tau)

        # play pi pulse
        wait(tau, 'qubit')
        play('pi', 'qubit')

        # play noise in the second part of the echo
        # if I call '-tele_noise' it will generate another wave
        play('tele_noise', 'noise_2nd', truncate=tau_2nd)
        play('tele_noise'*amp(-1), 'noise_2nd_cancel', truncate=tau_2nd_cancel)

        # play pi/2 pulse
        wait(tau, 'qubit')
        play('pi2', 'qubit')

        save(tau, tau_st)
        # wait(20)

    with stream_processing():
        tau_st.save_all('tau')




job = qmm.simulate(config, noise, simulate=SimulationConfig(duration=10000, include_analog_waveforms=True))
sw = job.simulated_analog_waveforms()
samples = job.get_simulated_samples()
t = np.arange(0, 5000*4, 1)
# ref = 0.2 * np.cos(2*np.pi*int(33.37e6)*t*1e-9+1.1+np.pi)
samples.con1.plot(analog_ports=['1'])
res_handles = job.result_handles
res_handles.wait_for_all_values()
tau = res_handles.get('tau').fetch_all()['value']
tau = tau*4

# assert noise_1st starts after pi2
counter_1 = 0
counter_2 = 0
for j in range(len(tau)):

    aux1 = sw['elements']['noise_1st'][counter_2]['timestamp']
    aux2 = sw['elements']['qubit'][counter_1]['timestamp'] + sw['elements']['qubit'][counter_1]['duration']

    if (sw['elements']['noise_1st'][counter_2]['duration'] != tau[j]):
        print('The duration of noise_1st is not tau')

    if (aux1 != aux2):
        print('For tau=', tau[j],' noise_1st does not start immediately after pi2')

    counter_1 += 6 # there are 3 waveforms in the qubit element, pi2,pi,pi2, each requiring 2 wfm points--one for I and one for Q
    counter_2 += 2 # these values have to be adapted to the experiment

# assert noise_1st and noise_2nd and noise_2nd_cancel start at the same time
counter = 0
for j in range(len(tau)):

    aux1 = sw['elements']['noise_1st'][counter]['timestamp'] + sw['elements']['qubit'][0]['duration']
    aux2 = sw['elements']['noise_2nd'][counter]['timestamp']
    aux3 = sw['elements']['noise_2nd_cancel'][counter]['timestamp']

    if (aux1 != aux2):
        print('For tau=', tau[j],' noise_2nd and noise_1st do not start at the same time')

    if (aux1 != aux3):
        print('For tau=', tau[j],' noise_2nd_cancel and noise_1st do not start at the same time')

    if (aux2 != aux3):
        print('For tau=', tau[j],' noise_2nd_cancel and noise_2nd do not start at the same time')

    counter += 2

# assert that pi starts after noise_1st
counter_1 = 2
counter_2 = 0
for j in range(len(tau)):

    aux1 = sw['elements']['noise_1st'][counter_2]['timestamp'] + sw['elements']['noise_1st'][counter_2]['duration']
    aux2 = sw['elements']['qubit'][counter_1]['timestamp']

    if (aux1 != aux2):
        print('For tau=', tau[j],' pi does not start immediately after noise_1st')

    counter_1 += 6
    counter_2 += 2

# assert that noise_2nd after cancelation starts after pi
counter_1 = 2
counter_2 = 0
for j in range(len(tau)):

    aux1 = sw['elements']['noise_2nd'][counter_2]['timestamp'] + sw['elements']['noise_2nd_cancel'][counter_2]['duration']
    aux2 = sw['elements']['qubit'][counter_1]['timestamp'] + sw['elements']['qubit'][counter_1]['duration']

    if (aux1 != aux2):
        print('For tau=', tau[j],' noise_2nd after cancelation starts after pi')

    counter_1 += 6
    counter_2 += 2

# assert that last pi2 starts after noise_2nd after cancelation
counter_1 = 4
counter_2 = 0
for j in range(len(tau)):

    aux1 = sw['elements']['noise_2nd'][counter_2]['timestamp'] + sw['elements']['noise_2nd'][counter_2]['duration']
    aux2 = sw['elements']['qubit'][counter_1]['timestamp']

    if (aux1 != aux2):
        print('For tau=', tau[j],' noise_2nd after cancelation starts after pi')

    counter_1 += 6
    counter_2 += 2

# plt.plot(ref)
# samples.con1.plot(analog_ports=['3','5','7'])
# samples.con1.plot()
# plt.plot(config['waveforms']['telegraph_noise']['samples'])
