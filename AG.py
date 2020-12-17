import numpy as np


class Individual:
    """
    Class describing individual in genetic algorithm
    """

    def __init__(self, N, min_value, max_value, initialize=True):
        """

        :param N: Length of excitation
        :param min_value: minimum excitation value
        :param max_value: maximum excitation value
        """
        stand_deviation_max_value = 3.0  # Do zmiany
        self.N = N
        if initialize:
            self.excitation = min_value + np.random.rand(N)*(max_value - min_value)
            self.stand_deviation = (np.random.rand(N)) * (stand_deviation_max_value)
        else:
            self.excitation = np.zeros(N)
            self.stand_deviation = np.zeros(N)

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
        for k in range(1, self.N+1):
            x_array[k] = self.excitation[k-1]

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
        for k in range(1, self.N + 1):
            x_array[k] = self.excitation[k - 1]

        return x_array

    def mutate(self, tau, yps):
        # tau = 0.2 # Zalezne od kroku algorytmu < 1
        # yps = 0.3

        stand_dev = 1.0
        for k, u_k, in enumerate(self.excitation):
            # self.excitation[i] = u_k + np.random.normal(scale=stand_dev)
            self.excitation[k] = u_k + np.random.normal(scale=self.stand_deviation[k])


class GA:
    def __init__(self, N, min_exc_value, max_exc_value, max_iterations, no_indviduals, lambda_ga):
        self.N = N
        self.min_exc_value = min_exc_value
        self.max_exc_value = max_exc_value

        self.max_iterations = max_iterations
        self.individuals_list = [Individual(self.N, self.min_exc_value, self.max_exc_value, True) for i in range(no_indviduals)]

        self.lambda_ga = lambda_ga + lambda_ga % 2  # Zapewnienie parzystosci lambda
        self.children_list = [Individual(self.N, self.min_exc_value, self.max_exc_value, False) for i in range(lambda_ga)]

    def run_algorithm(self):
        # Glowna petla algorytmu
        for i in range(self.max_iterations):
            #
            self.mutate_all()
            self.crossover()

    def mutate_all(self):
        for ind in self.individuals_list:
            ind.mutate(tau=0.2, yps=0.3)

    def crossover(self):
        """
        Choosing lambda individuals and doing crossover
        :return: None
        """
        #  Choosing parents indices in random way
        indices_to_cross_1 = np.random.randint(low=0, high=self.N, size=int(self.lambda_ga/2))
        indices_to_cross_2 = np.random.randint(low=0, high=self.N, size=int(self.lambda_ga/2))

        #  Iterating over parents - in evey iteration creating 2 children
        for i, (i1, i2) in enumerate(zip(indices_to_cross_1, indices_to_cross_2)):
            # self.children_list[i] = Individual(self.N, self.min_exc_value, self.max_exc_value, False)
            # self.children_list[i+int(self.lambda_ga/2)] = Individual(self.N, self.min_exc_value, self.max_exc_value, False)
            self.cross_2(i, i1, i2)


    def cross_2(self, i, i1, i2):
        """
        Crossover between to individuals
        :param i: child individual id
        :param i1: first parent id
        :param i2: second parent id
        :return: None
        """
        a_exc = np.random.rand(self.N)
        a_std_dev = np.random.rand(self.N)

        # Arithmetic crossover for excitation and standard deviation
        self.children_list[i].excitation = a_exc * self.individuals_list[i1].excitation + \
                                           (1-a_exc) * self.individuals_list[i2].excitation
        self.children_list[i].stand_deviation = a_std_dev * self.individuals_list[i1].stand_deviation + \
                                                (1 - a_std_dev) * self.individuals_list[i2].stand_deviation




        pass









ga = GA(N=100, min_exc_value=-200, max_exc_value=200, max_iterations=100, no_indviduals=100, lambda_ga=20)

ga.run_algorithm()




