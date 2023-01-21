from scipy.signal.windows import gaussian
import numpy as np


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]


qb_LO = 5e9
qb_IF = 50e6
rr_LO = 6e9
rr_IF = 50e6

# Creating 3 noise elements each with 6 operations (waveforms):
number_noise_el = 3
number_noise_wf_per_el = 6  # Assuming ~10k length, can play with this number
num_wf = number_noise_wf_per_el*number_noise_el

# Creating just a single list of arb wf (for simplicity), but 18 times
arb_wf_list = [[0.2]*10000 for i in range(num_wf)]

rr_pulse_len_in_clk = 50

pi_len = 20
pi_amp = 0.3
pi_half_amp = 0.15

config = {

    "version": 1,

    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit I
                2: {"offset": 0.0},  # qubit Q
                3: {"offset": 0.0},  # rr I
                4: {"offset": 0.0},  # rr Q
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0},  # rr I
                2: {"offset": 0.0}  # rr Q
            },
        },
    },

    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qb_LO,
                "mixer": "mixer_q1",
            },
            "intermediate_frequency": qb_IF,
            "digitalInputs": {},
            "operations": {
                "const": "const_pulse_IQ",
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse1",
                "pi_half": "pi_half_pulse1",
            },
        },
        **{f"noise_el{j}": {  # This creates 3 noise elements, tripling the amount of waveforms you could add
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qb_LO,
                "mixer": "mixer_q1",
            },
            "intermediate_frequency": qb_IF,
            "digitalInputs": {},
            "operations": {  # Next line create the different operations, addressing the different wfs
                **{f"noise_op{i}": f"noise_pulse{number_noise_wf_per_el*j+i}" for i in range(number_noise_wf_per_el)},
            },
        } for j in range(number_noise_el)},
        "rr": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": rr_LO,
                "mixer": "mixer_rl1",
            },
            "intermediate_frequency": rr_IF,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 188,
            "smearing": 0,
            "operations": {
                "const": "const_pulse_IQ",
                "readout": "ro_pulse1",
            },
        },
    },
    "pulses": {
        **{f"noise_pulse{i}": {
            "operation": "control",
            "length": len(arb_wf_list[0]),
            "waveforms": {
                "I": f"arb_wf{i}",
                "Q": "zero_wf",
            },
        } for i in range(num_wf)},
        "const_pulse_IQ": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "pi_pulse1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "pi_wf_i1",
                "Q": "pi_wf_q1",
            },
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "gaussian_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "pi_half_wf_i1",
                "Q": "pi_half_wf_q1",
            },
        },
        "ro_pulse1": {
            "operation": "measurement",
            "length": rr_pulse_len_in_clk * 4,
            "waveforms": {"I": "ro_wf1", "Q": "zero_wf"},
            "integration_weights": {
                "integW_cos": "integW1_cos",
                "integW_sin": "integW1_sin",
                "integW_minus_sin": "integW1_minus_sin"
            },
            "digital_marker": "ON",
        },
    },

    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": 0.1},
        "gaussian_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_len, pi_len/5)]},
        "ro_wf1": {"type": "constant", "sample": 0.1},
        "pi_wf_i1": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_len, pi_len/5)]},
        "pi_wf_q1": {"type": "constant", "sample": 0.0},
        "pi_half_wf_i1": {"type": "arbitrary", "samples": [float(arg) for arg in pi_half_amp * gaussian(pi_len, pi_len/5)]},
        "pi_half_wf_q1": {"type": "constant", "sample": 0.0},
        **{f"arb_wf{i}": {
            "type": "arbitrary",
            'is_overridable': True,  # This is what allows overriding the waveforms
            "samples": arb_wf_list[i]
        } for i in range(num_wf)}
    },

    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]}
    },

    "integration_weights": {
        "integW1_cos": {
            "cosine": [1.0] * rr_pulse_len_in_clk,
            "sine": [0.0] * rr_pulse_len_in_clk,
        },
        "integW1_sin": {
            "cosine": [0.0] * rr_pulse_len_in_clk,
            "sine": [1.0] * rr_pulse_len_in_clk,
        },
        "integW1_minus_sin": {
            "cosine": [0.0] * rr_pulse_len_in_clk,
            "sine": [-1.0] * rr_pulse_len_in_clk,
        },
        "integW2_cos": {
            "cosine": [1.0] * rr_pulse_len_in_clk,
            "sine": [0.0] * rr_pulse_len_in_clk,
        },
        "integW2_sin": {
            "cosine": [0.0] * rr_pulse_len_in_clk,
            "sine": [1.0] * rr_pulse_len_in_clk,
        },
        "integW2_minus_sin": {
            "cosine": [0.0] * rr_pulse_len_in_clk,
            "sine": [-1.0] * rr_pulse_len_in_clk,
        }
    },

    "mixers": {
        "mixer_q1": [{"intermediate_frequency": qb_IF, "lo_frequency": qb_LO, "correction": IQ_imbalance(0, 0)}],
        "mixer_rl1": [{"intermediate_frequency": rr_IF, "lo_frequency": rr_LO, "correction": IQ_imbalance(0, 0)}],
    }
}
