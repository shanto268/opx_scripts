import time

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

with program() as IQ_blobs:

    n = declare(int)

    with for_(n, 0, n < num_wf, n+1):
        with switch_(n):
            for i in range(num_wf):
                i_el = i // number_noise_wf_per_el
                i_op = i - i_el * number_noise_wf_per_el
                with case_(i):
                    play(f'noise_op{i_op}', f'noise_el{i_el}')


######################################
# Open Communication with the Server #
######################################
qmm = QuantumMachinesManager()

start = time.time()
qm = qmm.open_qm(config)
open = time.time()
prog_id = qm.compile(IQ_blobs)
compile = time.time()
pending_job = qm.queue.add_compiled(prog_id, overrides={
    'waveforms': {
        **{f'arb_wf{i}': [0.1] * len(arb_wf_list[0]) for i in range(number_noise_wf_per_el*number_noise_el)},
    }
})
new_job = pending_job.wait_for_execution()
run = time.time()

print(open - start)
print(compile - open)
print(run - compile)
