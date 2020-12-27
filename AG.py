import numpy as np


class Individual:
    """
    Class describing individual in genetic algorithm
    """

    def __init__(self, N, min_value, max_value):
        """

        :param N: Length of excitation
        :param min_value: minimum excitation value
        :param max_value: maximum excitation value
        """
        self.N = N
        self.excitation = min_value + np.random.rand(N)*(max_value - min_value)

    def calc_performance_index(self, x0):
        """
        Calculates state in time based on initial value and excitation.
        Later calculates performance index with formula: J = x^2 + u^2
        :param x0: initial state value
        :return: performance index
        """

        x_array = np.zeros(self.N+1)
        x_array[0] = x0
        # Calculating state
        for i in range(1, self.N+1):
            # print(self.excitation[i-1])
            x_array[i] = self.excitation[i-1]

        #  performance index - excitation^2 + state^2
        return np.sum(np.square(self.excitation)) + np.sum(x_array)

    def calc_state(self, x0):
        """
        Calculates state in time based on initial value and excitation.
        :param x0: initial state value
        :return: state in time
        """
        x_array = np.zeros(self.N + 1)
        x_array[0] = x0
        for i in range(1, self.N + 1):
            x_array[i] = self.excitation[i - 1]

        return x_array


class GA:
    def __init__(self, N, min_exc_value, max_exc_value, max_iterations, no_indviduals):
        self.N = N
        self.max_iterations = max_iterations
        self.individuals_list = [Individual(self.N, min_exc_value, max_exc_value) for i in range(no_indviduals)]




ga = GA(N=100, min_exc_value=-200, max_exc_value=200, max_iterations=10000, no_indviduals=100)

