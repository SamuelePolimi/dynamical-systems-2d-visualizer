"""
Author: Tosatto Samuele
Personal Email: samuele.tosatto@gmail.com
Institutional Email: samuele.tosatto@uibk.ac.at
Institution: Universitaet Innsbruck
Date: 14 October 2022
"""

import jax.numpy as jnp
from dynamic_systems import DynamicSystems2D


class GenericRobotLinkController:

    def get_torque(self, theta, omega) -> jnp.ndarray:
        pass


class RobotLink:
    def __init__(self, m=4/3, g=9.81, l=3/2):
        """

        :param m: mass of the arm
        :param g: gravitational acceleration
        :param l: length of the arm
        """
        self.m = m
        self.g = g
        self.l = l

    def get_dynamics(self, controller: GenericRobotLinkController):
        theta_dot = lambda theta, omega: self.f_1(omega)
        omega_dot = lambda theta, omega: self.f_2(theta, omega, controller)
        return DynamicSystems2D(theta_dot, omega_dot, name_x_0=r"$\theta$", name_x_1=r"$\omega$")

    def f_1(self, omega):
        theta_dot = omega
        return theta_dot

    def f_2(self, theta, omega, controller: GenericRobotLinkController):
        omega_dot = 3 * controller.get_torque(theta, omega) / (self.m * self.l**2) \
                    + 3/2 * self.g * jnp.sin(theta) / self.l
        return omega_dot


class RobotLinkController(GenericRobotLinkController):

    def __init__(self, robot_link: RobotLink, K_0, K_1, K_2, theta_d):
        """

        :param robot_link:
        :param K_0: Gravitational compensation (only for the desired theta)
        :param K_1: Proportional term
        :param K_2: Derivative term
        :param theta_d:
        """
        self.robot_link = robot_link
        self.K_0 = K_0
        self.K_1 = K_1
        self.K_2 = K_2
        self.theta_d = theta_d

    def get_torque(self, theta, omega):
        delta_theta = theta - self.theta_d
        grav_comp = - self.K_0 * 0.5 * self.robot_link.m * self.robot_link.l * self.robot_link.g * jnp.sin(self.theta_d)
        # this term is not real gravity compensation -- it compensate the gravity only for a desired angle
        # how can we change this term to make it working correctly?
        # Suggested exercise: rewrite grav_comp to compensate gravity.
        # Try to predict: how that will change the global dynamics?
        # How do we need to change K_1 and K_2 to obtain stable controller (possibly not spiral?)
        proportional_comp = - self.K_1 * delta_theta
        derivative_comp = - self.K_2 * omega
        torque = grav_comp + proportional_comp + derivative_comp
        return torque



