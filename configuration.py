from scipy.signal.windows import gaussian
import numpy as np
# from VISAdrivers import LO845m as LO

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

def delayed_gauss(amp, length, sigma):
    gauss_arg = np.linspace(-sigma, sigma, length)
    delay = 16 - length - 4
    if delay < 0:
        return amp * np.exp(-(gauss_arg ** 2) / 2)

    return np.r_[np.zeros(delay), amp * np.exp(-(gauss_arg ** 2) / 2), np.zeros(4)]


amp_q = 0.3
amp_r = 0.45 # amplitude of const and readout wfms in V, max is 0.5, but we shouldn't use the full range

# resFreq = 6.87745e9
# resFreq = int(6.9213e9)
# resFreq = 7.0186e9
# resFreq = 7.06e9
rr_LO = int(7.1e9)
# rr_LO = resFreq
# rr_IF = resFreq - rr_LO
rr_IF = 30e6

qbFreq = int(5.28e9)
# qbFreq = 5.58e9
# qb_LO = 3.834e9
qb_LO = int(5.23e9)
# qb_IF = qbFreq-qb_LO
qb_IF = 50e6
rr_pulse_len_in_clk = 350

phi = 0.45 # the angle to rotate the IQ plane such that most of the information lies on a single quadrature

pi_half_len = 24 # needs to be a multiple of 4
pi_len = 2 * pi_half_len
pi_amp = 0.45

gauss_len = 20
gauss_amp = 0.45
gauss_half_amp = gauss_amp / 2

gauss_wf_4ns = delayed_gauss(gauss_amp, 4, 2)

config = {

    "version": 1,

    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": -0.008},  # qubit I
                2: {"offset": 0.0115},  # qubit Q
                3: {"offset": -0.008},  # rr I
                4: {"offset": 0.001},  # rr Q
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0,"gain_db": 3},  # rr I
                2: {"offset": 0,"gain_db": 3}  # rr Q
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
                "gauss_4ns": "gaussian_4ns_pulse",
                "pi": "pi_pulse1",
                "pi_half": "pi_half_pulse1",
                "arb_op": "arb_pulse",
            },
        },
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
            "time_of_flight": 24, #without smearing # should be multiple of 4 (at least 24)
            "smearing": 0, #adds 40ns of data from each side of raw  adc trace to account for ramp up and down of readout pulse
            "operations": {
                "const": "const_pulse_IQ",
                "readout": "ro_pulse1",
            },
        },
    },
    
    "pulses": {
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
            "length": gauss_len,
            "waveforms": {
                "I": "gaussian_wf",
                "Q": "zero_wf",
            },
        },
        "gaussian_4ns_pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "I": "gaussian_4ns_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse1": {
            "operation": "control",
            "length": pi_half_len,
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
        "arb_pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "arb_wfm",
                "Q": "zero_wf",
            },
        },
    },

    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": amp_q},
        "gaussian_wf": {"type": "arbitrary", "samples": [float(arg) for arg in gauss_amp * gaussian(gauss_len, gauss_len/5)]},
        "gaussian_4ns_wf": {"type": "arbitrary", "samples": gauss_wf_4ns},
        "ro_wf1": {"type": "constant", "sample": amp_r},
        "pi_wf_i1": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_len, pi_len/5)]},
        "pi_wf_q1": {"type": "constant", "sample": 0.0},
        "pi_half_wf_i1": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_half_len, pi_half_len/5)]},
        "pi_half_wf_q1": {"type": "constant", "sample": 0.0},
        "arb_wfm": {"type": "arbitrary", "samples": [0.2]*10+[0.3]*10+[0.25]*20},
    },

    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]}
    },

    "integration_weights": {
        "integW1_cos": {
            "cosine": [np.cos(phi)] * rr_pulse_len_in_clk,
            "sine": [-np.sin(phi)] * rr_pulse_len_in_clk,
        },
        "integW1_sin": {
            "cosine": [np.sin(phi)] * rr_pulse_len_in_clk,
            "sine": [np.cos(phi)] * rr_pulse_len_in_clk,
        },
        "integW1_minus_sin": {
            "cosine": [-np.sin(phi)] * rr_pulse_len_in_clk,
            "sine": [-np.cos(phi)] * rr_pulse_len_in_clk,
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
        "mixer_q1": [{"intermediate_frequency": qb_IF, "lo_frequency": qb_LO, "correction": IQ_imbalance(-0.002,-0.021)}],
        "mixer_rl1": [{"intermediate_frequency": rr_IF, "lo_frequency": rr_LO, "correction": IQ_imbalance(0.0,-0.08)}],
    }
}
