import numpy as np

######################
# AUXILIARY FUNCTIONS:
######################

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


################
# CONFIGURATION:
################

long_readout_len = 3600
readout_len = 200

qubit_IF = 80e6
rr_IF = 20e6
rr2_IF = 22e6
qubit_LO = 6.345e9
rr_LO = 4.755e9



config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit I
                2: {"offset": 0.0},  # qubit Q
                3: {"offset": 0.0},  # RR I
                4: {"offset": 0.0},  # RR Q
                5: {"offset": 0.0},  # Flux Bias
                6: {"offset": 0.0},  # Drive Channel
                7: {"offset": 0.0},  # RR2
                8: {"offset": 0.0},  # RR2
            },
            "digital_outputs": {},
            "analog_inputs": {1: {"offset": 0.2}},
        }
    },
    "elements": {
        "Qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "CW": "CW",
                "saturation": "saturation_pulse",
                "Gaussian": "gaussian_pulse",
                "PiPulse": "pi_pulse",
                "pi2": "pi2_pulse",
                "minus_pi2": "minus_pi2_pulse",
            },
        },
        'qubit': {
            'singleInput': {
                'port': ('con1', 6),
            },
            'intermediate_frequency': 10e6,
            'operations': {
                'Drive': 'Drive'
            },
            'hold_offset': {'duration': 10},
        },
        "FluxBiasLine": {
            "singleInput": {
                "port": ("con1", 5)
            },
            'intermediate_frequency': 0e6,
            'operations': {
                'FluxBias': "FluxBias",
            },
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "CW": "CW",
                "long_readout": "long_readout_pulse",
                "pre-readout": "readout_pulse",
                "Readout": "empty_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
            'hold_offset': {'duration': 10},
        },
    },
    "pulses": {
        "FluxBias": {
            "operation": "control",
            "length": 100,
            "waveforms": {"single": "const_wf"},
        },
        "CW": {
            "operation": "control",
            "length": 600,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "Drive": {
            "operation": "control",
            "length": 100,
            "waveforms": {"single": "const_wf"},
        },
        "saturation_pulse": {
            "operation": "control",
            "length": 20000,  # several T1s
            "waveforms": {"I": "saturation_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "pi2_pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "pi2_wf", "Q": "zero_wf"},
        },
        "minus_pi2_pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "minus_pi2_wf", "Q": "zero_wf"},
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,
            "waveforms": {"I": "long_readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "long_integW1": "long_integW1",
                "long_integW2": "long_integW2",
            },
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
        },
        "empty_readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
            "integration_weights": {
                "cos": "cos",
                "sin": "sin",
                "optW1": "optW1",
                "optW2": "optW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.1},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "saturation_wf": {"type": "constant", "sample": 0.211},
        "gauss_wf": {"type": "arbitrary", "samples": gauss(0.4, 0.0, 20.0, 100)},
        "pi_wf": {"type": "arbitrary", "samples": gauss(0.3, 0.0, 20.0, 100)},
        "pi2_wf": {"type": "arbitrary", "samples": gauss(0.15, 0.0, 20.0, 100)},
        "minus_pi2_wf": {"type": "arbitrary", "samples": gauss(-0.15, 0.0, 20.0, 100)},
        "long_readout_wf": {"type": "constant", "sample": 0.32},
        "readout_wf": {"type": "constant", "sample": 0.2},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "long_integW1": {
            "cosine": [1.0] * int(long_readout_len / 4),
            "sine": [0.0] * int(long_readout_len / 4),
        },
        "long_integW2": {
            "cosine": [0.0] * int(long_readout_len / 4),
            "sine": [1.0] * int(long_readout_len / 4),
        },
        "cos": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "sin": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
        "optW1": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "optW2": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
        "mixer_RR2": [
            {
                "intermediate_frequency": rr2_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
    },
}