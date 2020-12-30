import numpy as np
from matplotlib import pyplot as plt
import math

class Reader:
    def __init__(self):
        pass

    def read_files(self, dir):
        """
        Read data from files
        :return:
        """
        self.best_performance_list = []
        self.average_performance_list = []

        file = open("{}/Others.txt".format(dir), "r")
        self.N = int(file.readline())
        self.x0 = int(file.readline())
        self.max_iterations = int(file.readline())
        file.close()

        self.average_standard_dev_list = np.loadtxt("{}/Average_standard_dev.txt".format(dir))
        self.individuals_preformance = np.loadtxt("{}/Individuals_performance.txt".format(dir))

        performance = np.loadtxt("{}/Performance.txt".format(dir))
        self.best_performance_list, self.average_performance_list = performance[0], performance[1]

        self.excitation = np.loadtxt("{}/Excitation.txt".format(dir))
        self.calc_state = np.loadtxt("{}/Calc_state.txt".format(dir))

    def draw_results(self, best_results_no):

        # Performance through time
        plt.plot(range(self.max_iterations), self.average_performance_list)
        plt.plot(range(self.max_iterations), self.best_performance_list)
        plt.legend(["average", "best"])
        plt.xlabel("algorithm iteration")
        plt.title("Performance through time x0 = {}, N = {}".format(self.x0, self.N))
        plt.show()

        # Average standard deviation through time
        plt.plot(range(self.max_iterations), self.average_standard_dev_list)
        plt.xlabel("algorithm iteration")
        plt.title("Average standard deviation through time x0 = {}, N = {}".format(self.x0, self.N))
        plt.show()

        for j in range(best_results_no):
            fig, axs = plt.subplots(2, 3)
            for i in range(6):
                row = 0
                col = i
                if i > 2:
                    col = i - 3
                    row = 1
                axs[row, col].plot(range(self.N), self.excitation[i + j*6])
                axs[row, col].plot(range(self.N+1), self.calc_state[i + j*6])
                axs[row, col].set_title("{}. result = {}".format(i + j*6, self.individuals_preformance[i + j*6]))

            for ax in axs.flat:
                ax.set(xlabel='n', ylabel='x/u')

            for ax in axs.flat:
                ax.label_outer()

            fig.legend(["excitation", "state"])
            fig.suptitle("Best algorithm results x0 = {}, N = {}".format(self.x0, self.N))
            plt.show()


RD = Reader()
RD.read_files("data")
RD.draw_results(1) # zestawy



