from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

config = {
    "version": 1,

    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.00},
            },
        },
    },

    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 50e6,
            "operations": {
                "saturation": "saturation_pulse",
            },
        },
    },

    "pulses": {
        "saturation_pulse": {
            "operation": "control",
            # the maximum pulse length for this version is 67 ms (2^24 cycles)
            # the pulse length follows the following formula:
            # length = (# samples of waveform)*(1e9/sampling_rate)
            # if you input a length that is not derived from the formula above you will get an error
            "length": 40000,  # in ns
            "waveforms": {"single": "saturation_wf"},
        },
    },

    "waveforms": {
        # Here you can define the # samples of waveform by "samples"
        # also the sampling rate by "sampling_rate"
        "saturation_wf": {"type": "arbitrary", "samples": [0.3]*20, "sampling_rate": 0.5e6},
    },
}


QMm = QuantumMachinesManager()


with program() as prog:
    play("saturation", "qe1")

for i in range(500):
    QM1 = QMm.open_qm(config)
# job = QM1.simulate(prog, SimulationConfig(int(2000//4)))  # in clock cycles, 4 ns
    job = QM1.execute(prog)

# samples = job.get_simulated_samples()
# samples.con1.plot()
# samples.con2.plot()
