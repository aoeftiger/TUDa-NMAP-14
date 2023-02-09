from io import BytesIO
from typing import Optional

import cheetah
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from matplotlib.patches import Ellipse

"""The Gym Environment for ARES-EA Focusing and Positioning Task

This is a simplified version from the actual ARESEA environment, containing only the essential functionalities required to build a Gym environment for this task.
Some details are removed for the sake of simplicity (weighting, misalignments, general interface for interaction with real machine etc.) 

c.f. J. Kaiser's paper on an RL application using this environment: https://proceedings.mlr.press/v162/kaiser22a.html
"""


class ARESEA(gym.Env):
    def __init__(
        self,
        action_mode="direct_unidirectional_quads",
        incoming_offset=np.array([0, 0]),
        magnet_init_mode=None,
        magnet_init_values=None,
        target_beam_mode="random",
        target_beam_values=None,
        threshold=5e-6,
        reward_mode="logl1",
    ) -> None:
        self.action_mode = action_mode
        self.incoming_offset = incoming_offset
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.threshold = threshold

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

        self.target_beam_space = spaces.Box(
            low=np.array([-1e-3, 0, -1e-3, 0], dtype=np.float32),
            high=np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float32),
        )

        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=0,
            mu_x=self.incoming_offset[0],
            mu_xp=0,
            mu_y=self.incoming_offset[1],
            mu_yp=0,
            sigma_x=3e-4,
            sigma_xp=1e-5,
            sigma_y=3e-4,
            sigma_yp=1e-5,
            sigma_s=0,
            sigma_p=0,
        )

        self.build_aresea_lattice()

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
        Q1 = cheetah.Quadrupole(length=0.122, name="Q1")
        Q2 = cheetah.Quadrupole(length=0.122, name="Q2")
        Q3 = cheetah.Quadrupole(length=0.122, name="Q3")
        CV = cheetah.VerticalCorrector(length=0.02, name="CV")
        CH = cheetah.HorizontalCorrector(length=0.02, name="CH")
        D1 = cheetah.Drift(length=0.1)
        D2 = cheetah.Drift(length=0.1)
        D3 = cheetah.Drift(length=0.204)
        D4 = cheetah.Drift(length=0.179)
        D5 = cheetah.Drift(length=0.45)
        Screen = cheetah.Screen(
            resolution=(2448, 2040),
            pixel_size=(3.3198e-6, 2.4469e-6),
            binning=4,
            name="Screen",
        )
        Screen.is_active = True
        lattice_cell = [Q1, D1, Q2, D2, CV, D3, Q3, D4, CH, D5, Screen]
        self.simulation = cheetah.Segment(cell=lattice_cell, device=torch.device("cpu"))

    def step(self, action):
        # Set magnet values
        self.magnets = action
        # Run the simulation / Update the accelerator
        self.update_accelerator()
        # Observe the beam
        self.out_beam = self.get_beam_parameters()
        # Build Observation
        observation = {
            "magnets": self.magnets,
            "target_beam": self.target_beam,
            "current_beam": self.out_beam,
        }
        # Calculate reward
        reward = self.calculate_reward()
        # Check if termination is achieved
        done = np.max(np.abs(self.out_beam - self.target_beam)) < self.threshold
        # Supply with additional info
        info = {}
        self.history.append(observation)
        return observation, reward, done, info

    def reset(self):
        # Set New Target
        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.target_beam_space.sample()
        else:
            self.target_beam = np.zeros(4)

        # Reset Magnets
        if self.magnet_init_mode == "random" or self.magnet_init_mode is None:
            pass
        elif self.magnet_init_mode == "constant":
            self.magnets = self.magnet_init_values
        else:
            raise ValueError(f"magnet_init_mode {self.magnet_init_mode} not valid")

        self.update_accelerator()
        self.out_beam = self.get_beam_parameters()

        observation = {
            "magnets": self.magnets,
            "target_beam": self.target_beam,
            "current_beam": self.out_beam,
        }
        info = {}
        self.history = [observation.copy()]
        return observation, info

    def calculate_reward(self):
        if self.reward_mode == "logl1":
            reward = -1 * np.log(np.mean(np.abs(self.out_beam - self.target_beam)))
        else:
            print("Reward mode not implemented yet")
            reward = 0
        return reward

    def update_accelerator(self):
        self.simulation(self.incoming)

    def get_beam_parameters(self):
        return np.array(
            [
                self.simulation.Screen.read_beam.mu_x,
                self.simulation.Screen.read_beam.sigma_x,
                self.simulation.Screen.read_beam.mu_y,
                self.simulation.Screen.read_beam.sigma_y,
            ]
        )

    def render(self, mode="human") -> Optional[np.ndarray]:
        """Renders the env image"""
        if not hasattr(self, "fig"):
            self.fig, self.axs = plt.subplots(
                1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 2]}
            )
        for ax in self.axs:
            ax.clear()

        # Beam Image Plot
        binning = np.array(self.simulation.Screen.binning)
        pixel_size = np.array(self.simulation.Screen.pixel_size)
        resolution = np.array(self.simulation.Screen.resolution)

        # Plot action
        plot_quadrupole_history(self.axs[0], self.history)
        plot_steerer_history(self.axs[1], self.history)
        # Plot current beam image
        tb = self.target_beam
        pos_x = tb[0] * 1e3
        pos_y = tb[2] * 1e3
        diameter_x = tb[1] * 2e3
        diameter_y = tb[3] * 2e3
        tb_ellipse = Ellipse(
            (pos_x, pos_y),
            width=diameter_x,
            height=diameter_y,
            edgecolor="green",
            lw=2,
            facecolor="none",
        )
        plot_beam_image(self.axs[2], self.get_beam_image(), resolution, pixel_size)
        self.axs[2].add_patch(tb_ellipse)

        self.fig.tight_layout(pad=0.5)

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            image_from_plot = np.frombuffer(
                self.fig.canvas.tostring_rgb(), dtype=np.uint8
            )
            image_from_plot = image_from_plot.reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,)
            )

            return image_from_plot

    def get_beam_image(self) -> np.ndarray:
        # Beam image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits)
        return self.simulation.Screen.reading / 1e9 * 2**12


# Plotting
def plot_quadrupole_history(ax, history):

    areamqzm1 = [obs["magnets"][0] for obs in history]
    areamqzm2 = [obs["magnets"][1] for obs in history]
    areamqzm3 = [obs["magnets"][3] for obs in history]

    start = 0
    steps = np.arange(start, len(history))

    ax.set_title("Quadrupoles")
    ax.set_xlim([start, len(history) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="Q1")
    ax.plot(steps, areamqzm2, label="Q2")
    ax.plot(steps, areamqzm3, label="Q3")
    ax.legend()


def plot_steerer_history(ax, history):
    areamcvm1 = np.array([obs["magnets"][2] for obs in history])
    areamchm2 = np.array([obs["magnets"][4] for obs in history])

    start = 0
    steps = np.arange(start, len(history))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([start, len(history) + 1])
    ax.plot(steps, areamcvm1 * 1e3, label="CV")
    ax.plot(steps, areamchm2 * 1e3, label="CH")
    ax.legend()


def render_env(env, fig, ax=None):
    data = env.render(mode="rgb_array")
    fig.set_size_inches(15, 6, forward=False)
    if ax is None:
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
    ax.imshow(data)


def plot_beam_image(ax, img, screen_resolution, pixel_size, title="Beam Image"):
    screen_size = screen_resolution * pixel_size

    ax.set_title(title)
    ax.set_xlabel("(mm)")
    ax.set_ylabel("(mm)")
    ax.imshow(
        img,
        vmin=0,
        aspect="equal",
        interpolation="none",
        extent=(
            -screen_size[0] / 2 * 1e3,
            screen_size[0] / 2 * 1e3,
            -screen_size[1] / 2 * 1e3,
            screen_size[1] / 2 * 1e3,
        ),
    )
