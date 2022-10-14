"""
Author: Tosatto Samuele
Personal Email: samuele.tosatto@gmail.com
Institutional Email: samuele.tosatto@uibk.ac.at
Institution: Universitaet Innsbruck
Date: 14 October 2022
"""

from robot_link import RobotLink, RobotLinkController
from visualize import VisualizeDynamicalSystem2D
import matplotlib.pyplot as plt
import jax.numpy as jnp

theta_d = eval(input("Insert Desired Theta: "))
K_0 = eval(input("Insert Gravitational Compensation: "))
K_1 = eval(input("Insert Proportional Term: "))
K_2 = eval(input("Insert Derivative Term: "))

robot_link = RobotLink()
controller = RobotLinkController(robot_link, K_0, K_1, K_2, theta_d)
robot_link_controller = robot_link.get_dynamics(controller)

visualizer = VisualizeDynamicalSystem2D(robot_link_controller)


visualizer.show_trajectories(-2 * jnp.pi, 2 * jnp.pi, -8, 8, time=10.,
                            equilibrium=jnp.array([theta_d, 0]))
plt.show()

visualizer.show_trajectories(theta_d - 0.05, theta_d + 0.05, -0.05, +0.05, time=10., equilibrium=jnp.array([theta_d, 0]))
plt.show()

_, ax_phase = plt.subplots(1, 1)
ax_phase.set_title("Linearized Dynamics")

visualizer.visualize_eigenvectors(ax_phase, jnp.array([theta_d, 0.]), theta_d - 0.05, theta_d + 0.05, -0.05, +0.05)
plt.show()