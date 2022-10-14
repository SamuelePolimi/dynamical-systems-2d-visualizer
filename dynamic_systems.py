import jax.numpy as jnp


class DynamicSystems2D:

    def __init__(self, f_1, f_2, name_x_0="", name_x_1=""):
        self.f_1 = f_1
        self.f_2 = f_2
        self.f = lambda x: jnp.array([f_1(x[0], x[1]), f_2(x[0], x[1])])
        self.name_x_0 = name_x_0
        self.name_x_1 = name_x_1
