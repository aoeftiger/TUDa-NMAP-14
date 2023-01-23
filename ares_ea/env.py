import cheetah
import gym
import numpy as np
from gym import spaces

"""The Gym Environment for ARES-EA Focusing and Positioning Task

This is a simplified version from the actual ARESEA environment, containing only the essential functionalities required to build a Gym environment for this task.
Some details are removed for the sake of simplicity (weighting, general interface for interaction with real machine etc.) 

c.f. J. Kaiser's paper on an RL application using this environment: https://proceedings.mlr.press/v162/kaiser22a.html
"""


class ARESEA(gym.Env):
    def __init__(self, action_mode="direct_unidirectional_quads") -> None:
        self.action_mode = action_mode

        if self.action_mode == "direct_unidirectional_quads":
            # physical bounds for magnet strengths, default case for simplicity
            self.action_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )
        elif self.action_mode == "direct":
            # for the case when the quad power supply can switch polarity
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

    @property
    def magnets(self):
        return np.array(
            [
                self.simulation.Q1.k1,
                self.simulation.Q2.k1,
                self.simulation.CV.angle,
                self.simulation.Q3.k1,
                self.simulation.CH.angle,
            ]
        )

    @magnets.setter
    def magnets(self, values):

        self.simulation.Q1.k1 = values[0]
        self.simulation.Q2.k1 = values[1]
        self.simulation.CV.angle = values[2]
        self.simulation.Q3.k1 = values[3]
        self.simulation.CH.angle = values[4]

    def build_aresea_lattice(self):
        # Initialize the ARES-EA lattice
        Q1 = cheetah.Quadrupole(length=0.1, name="Q1")
        Q2 = cheetah.Quadrupole(length=0.1, name="Q1")
        Q3 = cheetah.Quadrupole(length=0.1, name="Q1")
        CV = cheetah.VerticalCorrector(length=0.1, name="CV")
        CH = cheetah.VerticalCorrector(length=0.1, name="CH")
        lattice_cell = [Q1, Q2, CV, Q3, CH]
        self.simulation = cheetah.Segment(cell=lattice_cell)

    def step(self, action):
        pass
