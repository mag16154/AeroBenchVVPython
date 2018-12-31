import sys
# sys.path.append('../NNConfiguration_setup/')
import numpy as np
from NNConfiguration import NNConfiguration
from itertools import combinations
from check_engine import generateTrajs
import random

# Input (x,v',x')
# Output v
# x values are taken at consecutive times but x' is picked at a random time


class nnOnStateSpace(NNConfiguration):

    def __init__(self, samples=10, dimensions=2):

        NNConfiguration.__init__(self)
        self.samples = samples
        self.trajectories = []
        self.dimensions = dimensions
        self.steps = 0

    def generateTrajectories(self):

        ns = 0
        while ns < self.samples+1:
            traj = generateTrajs(self)
            if traj is not None:
                self.trajectories.append(traj)
                ns += 1
        print("Gathered {} sample trajectories".format(ns))

    def createNN(self, jump_size=1):

        traj_combs = []

        for jump in range(1, jump_size+1):
            start_idx = (jump-1)*12
            end_idx = jump*12
            traj_indices = list(range(start_idx, end_idx))
            traj_combs += list(combinations(traj_indices, 2))
        print(traj_combs)

        print(jump_size)
        input = []
        output = []
        self.steps = len(self.trajectories[traj_combs[0][0]])-1
        # print(self.steps)
        for traj_pair in traj_combs:
            t_pair = list(traj_pair)
            traj_1 = self.trajectories[t_pair[0]]
            traj_2 = self.trajectories[t_pair[1]]
            for step in range(0, self.steps, jump_size):
                x_vp_xp_pair = list(traj_1[step])
                v_val = traj_2[step] - traj_1[step]
                t_idx = random.randint(0, self.steps-step)
                t_val = step+t_idx
                # print(t_val)
                vprime_val = traj_2[t_val] - traj_1[t_val]
                x_vp_xp_pair = x_vp_xp_pair + list(vprime_val)
                x_vp_xp_pair = x_vp_xp_pair + list(traj_1[t_val])
                input.append(x_vp_xp_pair)
                output.append(v_val)
        print(len(input))
        self.input = np.asarray(input)
        self.output = np.asarray(output)
