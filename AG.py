import numpy as np
from matplotlib import pyplot as plt
import copy
import os


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
        stand_deviation_max_value = 5.0  # Do zmiany
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
            x_array[k] = x_array[k-1] + self.excitation[k-1]

        #  performance index -> excitation^2 + state^2
        return np.sum(np.square(self.excitation)) + np.sum(np.square(x_array))

    def calc_state(self, x0):
        """
        Calculates state in time based on initial value and excitation.
        :param x0: initial state value
        :return: state in time
        """
        x_array = np.zeros(self.N + 1)
        x_array[0] = x0
        for k in range(1, self.N + 1):
            x_array[k] = x_array[k-1] + self.excitation[k - 1]

        return x_array

    def mutate(self, tau, yps):
        """
        Mutation for individual:
        First mutation of standard deviation, based on previous standard deviation values and parameters
        which depend on current algorithm step (tau, yps).
        Later mutation of excitation based on previous value of excitation and current value of standard deviation.
        :param tau: parameter, based on step of algorithm. Decreases in time
        :param yps: parameter, based on step of algorithm. Decreases in time
        :return: None
        """
        # Mutation of standard deviation
        self.stand_deviation = self.stand_deviation * np.exp(
            tau * np.random.normal(size=self.N) + yps * np.random.normal(size=self.N))

        # Mutation of excitation
        self.excitation = self.excitation + np.random.normal(scale=self.stand_deviation)
        self.excitation[self.excitation > 200] = 200
        self.excitation[self.excitation < -200] = -200


class GA:

    def __init__(self, N, min_exc_value, max_exc_value, x0, max_iterations, no_indviduals, lambda_ga):
        self.N = N
        self.min_exc_value = min_exc_value
        self.max_exc_value = max_exc_value
        self.no_individuals = no_indviduals

        self.average_performance_list = []
        self.best_performance_list = []
        self.average_standard_dev_list = []

        self.max_iterations = max_iterations
        self.individuals_list = [Individual(self.N, self.min_exc_value, self.max_exc_value, True) for i in range(no_indviduals)]

        self.lambda_ga = lambda_ga + lambda_ga % 2  # Zapewnienie parzystosci lambda
        self.children_list = [Individual(self.N, self.min_exc_value, self.max_exc_value, False) for i in range(lambda_ga)]
        self.x0 = x0

    def run_algorithm(self):
        """
        Main algorithm loop
        :return: None
        """
        for alg_it in range(self.max_iterations):

            self.crossover()
            self.mutate_all(alg_it)
            self.find_best_individuals()

    def mutate_all(self, alg_it):
        """
        Mutate all individuals
        :param alg_it: iteration of algorithm
        :return: None
        """
        # Parameters indicating size of mutation
        tau = 1/(np.sqrt(2*np.sqrt(alg_it+1)))
        yps = 1/(np.sqrt(2*alg_it+1))

        for i, ind in enumerate(self.children_list):
            ind.mutate(tau=tau, yps=yps)

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
            self.cross_2(i, i1, i2)

    def cross_2(self, i, i1, i2):
        """
        Crossover between to individuals
        :param i: child individual id
        :param i1: first parent id
        :param i2: second parent id
        :return: None
        """
        # Coefficients for crossover
        a_exc = np.random.rand(self.N)
        a_std_dev = np.random.rand(self.N)

        # Arithmetic crossover for excitation and standard deviation
        # First child
        self.children_list[i].excitation = a_exc * self.individuals_list[i1].excitation + \
                                           (1-a_exc) * self.individuals_list[i2].excitation
        self.children_list[i].stand_deviation = a_std_dev * self.individuals_list[i1].stand_deviation + \
                                                (1 - a_std_dev) * self.individuals_list[i2].stand_deviation

        # Second child
        self.children_list[i+int(self.lambda_ga/2)].excitation = (1-a_exc) * self.individuals_list[i1].excitation + \
                                           a_exc * self.individuals_list[i2].excitation
        self.children_list[i+int(self.lambda_ga/2)].stand_deviation = (1-a_std_dev) * self.individuals_list[i1].stand_deviation + \
                                                a_std_dev * self.individuals_list[i2].stand_deviation

    def find_best_individuals(self):
        """
        Finds best individuals in individuals_list and children_list.
        Also calculates average performance of individual_list, and best best performance
        :return:
        """
        individuals_performance = np.asarray([individual.calc_performance_index(x0=self.x0) for individual in self.individuals_list])
        worst_performance = np.max(individuals_performance)

        for i, child in enumerate(self.children_list):
            child_performance = child.calc_performance_index(x0=self.x0)
            if child_performance < worst_performance:
                individuals_performance = np.append(individuals_performance, child_performance)
                self.individuals_list.append(copy.deepcopy(child))

        indexed_list = zip(individuals_performance, self.individuals_list)

        self.individuals_list = [x for _, x in sorted(indexed_list, key=lambda pair: pair[0])]
        self.individuals_list = self.individuals_list[: self.no_individuals]

        self.average_performance_list.append(np.average(individuals_performance))
        self.best_performance_list.append(np.min(individuals_performance))

        average_standard_deviation = 0.0
        for i, ind in enumerate(self.individuals_list):
            average_standard_deviation += np.average(ind.stand_deviation)
        self.average_standard_dev_list.append(average_standard_deviation/self.no_individuals)

    def save_to_files(self, dir):
        """
        Save data to files
        :return:
        """
        try:
            os.mkdir(dir)
        except OSError:
            print("Creation of the directory %s failed" % dir)

        file = open("{}/Others.txt".format(dir), "w")
        file.write("{}".format(self.N))
        file.write("\n{}".format(self.x0))
        file.write("\n{}".format(self.max_iterations))
        file.close()

        np.savetxt("{}/Performance.txt".format(dir), [self.best_performance_list, self.average_performance_list])
        np.savetxt("{}/Average_standard_dev.txt".format(dir), np.asarray(self.average_standard_dev_list))

        excitation = []
        calc_state = []
        for i in range(self.no_individuals):
            excitation.append(self.individuals_list[i].excitation)
            calc_state.append(self.individuals_list[i].calc_state(50))
        np.savetxt("{}/Excitation.txt".format(dir), np.asarray(excitation))
        np.savetxt("{}/Calc_state.txt".format(dir), np.asarray(calc_state))


N_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]
x0 = 50
max_iterations = 500
no_indviduals = 200
lambda_ga = 1400

for N in N_list:
    ga = GA(N=N, min_exc_value=-200, max_exc_value=200, x0=x0, max_iterations=max_iterations, no_indviduals=no_indviduals, lambda_ga=lambda_ga)
    ga.run_algorithm()
    ga.save_to_files(f"data_{N}")


