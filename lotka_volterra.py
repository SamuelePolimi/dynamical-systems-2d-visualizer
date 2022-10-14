"""
Author: Tosatto Samuele
Personal Email: samuele.tosatto@gmail.com
Institutional Email: samuele.tosatto@uibk.ac.at
Institution: Universitaet Innsbruck
Date: 14 October 2022
"""
from dynamic_systems import DynamicSystems2D


class LotkaVolterra:

    def __init__(self, alpha, beta, delta, gamma):
        """
        LotkaVolterra dynamical model
        :param alpha: the number of preys increases proportionally to the current number of prays
        :param beta: the current number of preys decreases proportionally to the current number of predators
        :param delta: the current number of predator increases proportionally to the current number of preys
        :param gamma: the current number of predators decreases if the number of predators is much larger than the one
        of the preys
        """
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def get_dynamics(self):
        return DynamicSystems2D(self.f_1, self.f_2, name_x_0=r"$\x", name_x_1=r"$\y")

    def f_1(self, x, y):
        return self._alpha * x - self._beta * x * y

    def f_2(self, x, y):
        return self._delta * x * y - self._gamma * y