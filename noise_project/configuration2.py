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
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

def gen_tel_noise(numPoints,tau,dt):
    signal = np.ones(numPoints)*(-1)**np.random.randint(0,2)
    for i in range(1,numPoints-1):
        if np.random.rand() < 1/(2*tau/dt)*np.exp(-1/(2*tau/dt)):
            signal[i+1] = - signal[i]
        else:
            signal[i+1] = signal[i]
    return signal


################
# CONFIGURATION:
################

long_redout_len = 3600
readout_len = 2000

qubit_IF = int(33.37e6)
rr_IF = 50e6

qubit_LO = 6.345e9
rr_LO = 4.755e9

# parameters for random telegraph noise
N_points = 10000
tau = 500
dt = 1

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  # qubit I
                2: {'offset': 0.0},  # qubit Q
                3: {'offset': 0.0},  # RR I
                4: {'offset': 0.0},  # RR Q
                5: {'offset': 0.0},
                6: {'offset': 0.0},
                7: {'offset': -0.0},
                8: {'offset': -0.0},
                # 9: {'offset': 0.0},
                # 10: {'offset': 0.0},
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {"offset": 0.0}
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            }
        },

        'noise_1st': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'tele_noise': 'tele_noise',
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            }
        },

        'noise_2nd': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'tele_noise': 'tele_noise',
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            }
        },

        'noise_2nd_cancel': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'tele_noise': 'tele_noise',
                'CW': 'CW',
                '-CW': '-CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            }
        },

        'rr': {
            'mixInputs': {
                'I': ('con1', 7),
                'Q': ('con1', 8),
                'lo_frequency': rr_LO,
                'mixer': 'mixer_RR'
            },
            'intermediate_frequency': rr_IF,
            'operations': {
                'CW': 'CW',
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                "out2": ("con1", 2)
            },
            'time_of_flight': 28,
            'smearing': 0,
        },
    },

    "pulses": {

        "CW": {
            'operation': 'control',
            'length': 64,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            }
        },

        "-CW": {
            'operation': 'control',
            'length': 60000,
            'waveforms': {
                'I': '-const_wf',
                'Q': 'zero_wf'
            }
        },

        "tele_noise": {
            'operation': 'control',
            'length': 10000,
            'waveforms': {
                'I': 'telegraph_noise',
                'Q': 'zero_wf'
            }
        },

        "saturation_pulse": {
            'operation': 'control',
            'length': 20000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            }
        },

        "gaussian_pulse": {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },

        'pi_pulse': {
            'operation': 'control',
            'length': 64,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            }
        },

        'pi2_pulse': {
            'operation': 'control',
            'length': 64,
            'waveforms': {
                'I': 'pi2_wf',
                'Q': 'zero_wf'
            }
        },

        'minus_pi2_pulse': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'I': 'minus_pi2_wf',
                'Q': 'zero_wf'
            }
        },

        'long_readout_pulse': {
            'operation': 'measurement',
            'length': long_redout_len,
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW1': 'long_integW1',
                'long_integW2': 'long_integW2',
            },
            'digital_marker': 'ON'
        },

        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                '-integW2': '-integW2',
                'optW1': 'optW1',
                'optW2': 'optW2'
            },
            'digital_marker': 'ON'
        },

    },

    'waveforms': {

        'const_wf': {
            'type': 'constant',
            'sample': 0.4
        },

        '-const_wf': {
            'type': 'constant',
            'sample': -0.4
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'saturation_wf': {
            'type': 'constant',
            'sample': 0.211
        },

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.4, 0.0, 6.0, 60)
        },

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3, 0.0, 6.0, 64)
        },

        'pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.15, 0.0, 6.0, 64)
        },

        'minus_pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(-0.15, 0.0, 6.0, 60)
        },

        'long_readout_wf': {
            'type': 'constant',
            'sample': 0.32
        },

        'readout_wf': {
            'type': 'constant',
            'sample': 0.34
        },

        'telegraph_noise': {
            'type': 'arbitrary',
            'samples': list(0.1*gen_tel_noise(N_points,tau,dt))
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'long_integW1': {
            'cosine': [1.0] * int(long_redout_len / 4),
            'sine': [0.0] * int(long_redout_len / 4)
        },

        'long_integW2': {
            'cosine': [0.0] * int(long_redout_len / 4),
            'sine': [1.0] * int(long_redout_len / 4)
        },

        'integW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
        },

        '-integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
        },

        'optW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4)
        },
    },

    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(0.0, 0.0)}
        ],
        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': IQ_imbalance(0.0, 0.0)}
        ],
    }

}

