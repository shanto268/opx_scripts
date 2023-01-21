import time

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_simple import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

with program() as IQ_blobs:
    play('noise_op', 'noise_el')


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
        'arb_wf': [0.1] * len(arb_wf),
    }
})
replace_wfm = time.time()
new_job = pending_job.wait_for_execution()
run = time.time()

print("Opening configuration file took %.1f ms"%((open-start)*1e3))
print("Compiling took %.1f ms"%((compile - open)*1e3))
print("Replacing waveform took %.1f ms"%((replace_wfm - compile)*1e3))
