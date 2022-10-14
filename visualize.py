"""
Author: Tosatto Samuele
Personal Email: samuele.tosatto@gmail.com
Institutional Email: samuele.tosatto@uibk.ac.at
Institution: Universitaet Innsbruck
Date: 14 October 2022
"""

import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import imageio
from scipy.integrate import odeint
import numpy as np
from matplotlib import rc
rc('text', usetex=True)

from dynamic_systems import DynamicSystems2D


class VisualizeDynamicalSystem2D:

    def __init__(self, dynamical_system: DynamicSystems2D):
        self._d_s = dynamical_system

    def visualize_phase(self, ax, x_min, x_max, y_min, y_max, x_res=20, y_res=20):
        x_space = np.linspace(x_min, x_max, x_res)
        y_space = np.linspace(y_min, y_max, y_res)
        X, Y = np.meshgrid(x_space, y_space)
        U = self._d_s.f_1(X, Y)
        V = self._d_s.f_2(X, Y)

        ax.quiver(X, Y, U, V, angles="xy", scale_units="xy")
        self.phase_axis_labels(ax)

    def show_trajectories(self, x_min, x_max, y_min, y_max, x_res=20, y_res=20, time=3.,
                       iterations=100, title="system", equilibrium=None):
        fig, ax_phase = plt.subplots(1, 1)
        self.visualize_phase(ax_phase, x_min, x_max, y_min, y_max, x_res, y_res)

        init_points = np.random.uniform(np.array([x_min, y_min]), np.array([x_max, y_max]), size=(10, 2))

        solutions = []
        ode_system = lambda t, x: self._d_s.f(x)
        steps = np.linspace(0, time, iterations)
        for init_point in init_points:
            solution = odeint(ode_system, init_point, steps, tfirst=True)
            solutions.append(solution)
            ax_phase.plot(solution[:, 0], solution[:, 1], color="b")
            if equilibrium is not None:
                ax_phase.plot(equilibrium[0], equilibrium[1], "ro")

        ax_phase.set_xlim(x_min, x_max)
        ax_phase.set_ylim(y_min, y_max)

        fig, ax_time = plt.subplots(2, 1)
        ax_time[1].set_ylabel(self._d_s.name_x_1)
        ax_time[1].set_xlabel("$t$")
        ax_time[0].set_ylabel(self._d_s.name_x_0)

        for solution in solutions:
            ax_time[0].plot(steps, solution[:, 0], color="b")
            ax_time[1].plot(steps, solution[:, 1], color="b")

    def gif_trajectories(self, x_min, x_max, y_min, y_max, x_res=20, y_res=20, time=3.,
                       iterations=100, filename="system", title="", save_gif=False, equilibrium=None):

        fig, ax_phase = plt.subplots(1, 1)

        init_points = np.random.uniform(np.array([x_min, y_min]), np.array([x_max, y_max]), size=(10, 2))

        solutions = []
        ode_system = lambda t, x: self._d_s.f(x)
        steps = np.linspace(0, time, iterations)

        filenames = []
        for init_point in init_points:
            solution = odeint(ode_system, init_point, steps, tfirst=True)
            solutions.append(solution)
        for i in range(0, len(steps) - 1, 1):
            plt.cla()
            ax_phase.set_title(title)
            ax_phase.set_xlim(x_min, x_max)
            ax_phase.set_ylim(y_min, y_max)
            self.visualize_phase(ax_phase, x_min, x_max, y_min, y_max, x_res, y_res)
            for solution in solutions:
                ax_phase.plot(solution[:i, 0], solution[:i, 1], color="b")
                ax_phase.plot(solution[i-1, 0], solution[i-1, 1], 'bo')
                if equilibrium is not None:
                    ax_phase.plot(equilibrium[0], equilibrium[1], "ro")
            filenames.append("gif/%s_temp_%s.png" % (filename, i))
            plt.savefig(filenames[-1])

        # build gif
        with imageio.get_writer('gif/%s_phase.gif' % filename, mode='I') as writer:
            for name in filenames:
                image = imageio.imread(name)
                writer.append_data(image)

        fig, ax_time = plt.subplots(2, 1)

        ax_time[0].set_title(title)
        ax_time[1].set_ylabel(self._d_s.name_x_1)
        ax_time[1].set_xlabel("$t$")
        ax_time[0].set_ylabel(self._d_s.name_x_0)

        solutions_a = np.array(solutions)
        ax_time[0].set_xlim(0, time)
        ax_time[1].set_xlim(0, time)
        ax_time[0].set_ylim(np.min(solutions_a[..., 0]), np.max(solutions_a[..., 0]))
        ax_time[1].set_ylim(np.min(solutions_a[..., 1]), np.max(solutions_a[..., 1]))

        filenames = []
        for i in range(0, len(steps) - 1, 1):
            for solution in solutions:

                if equilibrium is not None:
                    ax_time[0].plot([0, steps[-1]], [equilibrium[0], equilibrium[0]], color="r")
                    ax_time[1].plot([0, steps[-1]], [equilibrium[1], equilibrium[1]], color="r")
                ax_time[0].plot(steps[0:i+1], solution[0:i+1, 0], color="b")
                ax_time[1].plot(steps[0:i+1], solution[0:i+1, 1], color="b")
            filenames.append("gif/%s_temp_%s.png" % (filename, i))
            plt.savefig(filenames[-1])

        # build gif
        with imageio.get_writer('gif/%s_trajectories.gif' % filename, mode='I') as writer:
            for name in filenames:
                image = imageio.imread(name)
                writer.append_data(image)

        # Remove files
        for filename in set(filenames):
            os.remove(filename)

    def linearize(self, x_eq):
        f_grad = jax.jacfwd(self._d_s.f)

        A = f_grad(x_eq)

        e_values, e_vectors = np.linalg.eig(A)

        trace = jnp.trace(A)
        det = jnp.linalg.det(A)
        delta = trace**2 - 4 * det

        type_lin = ""
        if det == 0.:
            if trace > 0.: type_lin = "Line of stable fixed points"
            if trace > 0.: type_lin = "Line of unstable fixed points"
            if trace == 0.: type_lin = "Uniform motion"
        elif det < 0.: type_lin = "Saddle equilibrium"
        else:
            if trace > 0:
                if delta > 0.:
                    type_lin = "Unstable spiral"
                elif delta < 0.:
                    type_lin = "Simple source (unstable)"
                else:
                    type_lin = "Degenerate source (unstable)"
            elif trace < 0:
                if delta > 0.:
                    type_lin = "Stable spiral"
                elif delta < 0.:
                    type_lin = "Simple sink (stable)"
                else:
                    type_lin = "Degenerate sink (stable)"
            else:
                type_lin = "Neutrally stable orbits"


        print("""
        Linear system analysis: %s
        
        Eigenvalue1: %s
        Eigenvector1: %s
        
        Eigenvalue2: %s
        Eigenvector2: %s      
        
        Trace: %f
        Determinant: %f
        Delta: %f
        """ % (type_lin, e_values[0], e_vectors[0], e_values[1], e_vectors[1], trace, det, delta))

        linearized_eq = lambda x: self._d_s.f(x_eq) + A @ (x - x_eq)
        f_1 = lambda x, y: linearized_eq(jnp.array([x, y]))[0]
        f_2 = lambda x, y: linearized_eq(jnp.array([x, y]))[1]

        return DynamicSystems2D(np.vectorize(f_1), np.vectorize(f_2), name_x_0=self._d_s.name_x_0,
                                name_x_1=self._d_s.name_x_1), e_values, e_vectors.T

    def visualize_eigenvectors(self, ax, x_eq, x_min, x_max, y_min, y_max, x_res=20, y_res=20):
        d_s_l, eigenvalues, eigenvectors = self.linearize(x_eq)
        visualizer = VisualizeDynamicalSystem2D(d_s_l)
        visualizer.visualize_phase(ax, x_min, x_max, y_min, y_max, x_res, y_res)
        for i in range(2):
            if jnp.imag(eigenvalues[i]) == 0.:
                print("Eigenvalue %d is real, therefore we can draw it" % i)
                if eigenvalues[i] > 0.:
                    label = "Unstable eigenvector"
                    color = "orange"
                else:
                    label = "Stable eigenvector"
                    color = "green"

                def f(x):
                    norm_eigen = eigenvectors[i]/eigenvectors[i, 0]
                    return x_eq[1] + norm_eigen[1] * (x - x_eq[0])

                e_y_min = f(x_min)
                e_y_max = f(x_max)
                ax.plot([x_min, x_max], [e_y_min, e_y_max], label=label, color=color)
            ax.plot(x_eq[0], x_eq[1], 'ro')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()

    def phase_axis_labels(self, ax):
        ax.set_xlabel(self._d_s.name_x_0)
        ax.set_ylabel(self._d_s.name_x_1)




