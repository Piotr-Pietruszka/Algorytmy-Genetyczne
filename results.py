import numpy as np
from matplotlib import pyplot as plt


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

        performance = np.loadtxt("{}/Performance.txt".format(dir))
        self.best_performance_list, self.average_performance_list = performance[0], performance[1]

        self.excitation = np.loadtxt("{}/Excitation.txt".format(dir))
        self.calc_state = np.loadtxt("{}/Calc_state.txt".format(dir))

    def draw_results(self, best_results_no):

        # Performance through time
        plt.plot(range(self.max_iterations), self.average_performance_list)
        plt.plot(range(self.max_iterations), self.best_performance_list)
        plt.legend(["average", "best"])
        plt.show()

        # Average standard deviation through time
        plt.plot(range(self.max_iterations), self.average_standard_dev_list)
        plt.show()

        for i in range(best_results_no):
            # Best excitation and simulation at the end
            plt.plot(range(self.N), self.excitation[i])
            plt.plot(range(self.N+1), self.calc_state[i])
            plt.legend(["excitation", "state"])
            plt.show()


RD = Reader()
RD.read_files("data")
RD.draw_results(5)



