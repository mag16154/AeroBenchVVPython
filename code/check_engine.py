'''
Stanley Bak

Engine controller specification checking
'''

import numpy as np
from numpy import deg2rad

from RunF16Sim import RunF16Sim
from PassFailAutomaton import AirspeedPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import FixedSpeedAutopilot
from controlledF16 import controlledF16
import random

from plot import plot2d


def generateTrajs(self):

    # Initial Conditions ###

    setpoint = 2220
    p_gain = 0.01
    power = 0  # Power
    alt = 20000  # Initial Attitude
    # Vt = 1000  # Initial Speed
    Vt = random.randint(1000, 1020)
    phi = 0  # (pi/2)*0.5           # Roll angle from wings level (rad)
    theta = 0  # (-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = 0  # -pi/4                # Yaw angle from North (rad)

    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()
    llc = LowLevelController(ctrlLimits)

    ap = FixedSpeedAutopilot(setpoint, p_gain, llc.xequil, llc.uequil, flightLimits, ctrlLimits)

    pass_fail = AirspeedPFA(60, setpoint, 5)

    # Default alpha & beta
    alpha_deg = round(random.uniform(2.0, 2.15), 4)
    alpha = deg2rad(alpha_deg) # Trim Angle of Attack (rad)
    beta = round(random.uniform(-0.06, 0.06), 4)              # Side slip angle (rad)

    print(Vt, alpha_deg, alpha, beta)
    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 100 # simulation time

    def der_func(t, y):
        'derivative function'

        der = controlledF16(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0]  # speed
        rv[12] = der[12]  # power lag term

        return rv

    passed, times, states, modes, ps_list, Nz_list, u_list = \
        RunF16Sim(initialState, tMax, der_func, f16_plant, ap, llc, pass_fail, sim_step=0.1)

    print("Simulation Conditions Passed: {}".format(passed))
    print(len(states))
    if passed is True:
        return states
    else:
        None
    # plot
    # filename = None # engine_e.png
    # plot2d(filename, times, [(states, [(0, 'Vt'), (12, 'Pow')]), (u_list, [(0, 'Throttle')])])
