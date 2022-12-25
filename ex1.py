from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import random


def get_genotypes(n, convert_to_genotype=False):
    """

    :param n: length of genotypes
    :param convert_to_genotype:
    :return: list of all binary vectors of size n. if
    convert_to_genotype=True the vectors will be wrapped by a Genotype object
    """

    if convert_to_genotype:
        return [Genotype(np.array(list(i))) for i in product([0, 1], repeat=n)]
    else:
        return [np.array(list(i)) for i in product([0, 1], repeat=n)]


def auto_corr(X : np.array, max_delta):
    """
    Calculates the auto-correlation of a vector X with itself given max delta
    :param X:
    :param max_delta:
    :return: auto-correlation vector
    """

    mean = X.mean()
    auto_corr0 = np.power(X - mean, 2).sum()
    corr = []
    for delta in range(-max_delta, 0):
        corr.append(np.dot((X[:delta] - mean).T, (X[-delta:] - mean) /
                    auto_corr0))
    corr.append(1)
    for delta in range(1, max_delta + 1):
        corr.append(np.dot((X[: -delta]-mean).T, (X[delta:] - mean) /
                                                  auto_corr0))

    return np.asarray(corr)


class Genotype:
    """
    Wrapper for a genotype
    """

    def __init__(self, gen_as_arr):
        """
        init Genotype attributes according to given binary vector
        :param gen_as_arr: Binary vector
        """

        self.fitness = dict()       # fitness for every k
        self.gen_as_arr = gen_as_arr
        self.gen_as_str = str(gen_as_arr)

    def __eq__(self, other):
        if not isinstance(other, Genotype):
            return False
        return self.gen_as_str == other.gen_as_str

    def __hash__(self):
        return hash(self.gen_as_str)

    def __len__(self):
        return len(self.gen_as_arr)


class FLS:
    """
    Fitness Landscape model
    """

    def __init__(self, n, ks, is_uncorrelated=False):
        """

        :param n: length of genotype
        :param ks: correlation coeffients
        :param is_uncorrelated: is model Uncorrelated (or NK)
        """
        self.genotype_len = n
        self.ks = ks
        self.dict_of_tup_to_fitness_dicts = dict()
        self.genotypes = dict()

        if ks:
            self.generate_fitness_to_tup_dict()

        self.calc_fitness(is_uncorrelated)
        self.title_name = "Uncorrelated" if is_uncorrelated else "NK"
        self.is_uncorrelated = is_uncorrelated

    def generate_fitness_to_tup_dict(self):
        """
        For every k in self.ks the function creates a map which links all
        binary vectors of size k to a certain uniform fitness
        :return:
        """
        for k in self.ks:
            k_tuples = get_genotypes(k)
            current_dict = dict()
            for tup in k_tuples:
                current_dict[str(tup)] = np.random.uniform()
            self.dict_of_tup_to_fitness_dicts[k] = current_dict

    def calc_single_gen_fitness(self, gen : Genotype, is_uncorrelated=False):
        """
        This function calculates a fitness of the given genotype.
        If model is uncorrelated, the fitness is sampled from uniform
        distrubution, else it's computed as some of f_i.
        :param gen:
        :param is_uncorrelated:
        :return:
        """

        if is_uncorrelated:
            gen.fitness[0] = np.random.uniform()
            return

        for k in self.ks:
            assert k <= len(gen)
            cyclic_gen = np.concatenate((gen.gen_as_arr, gen.gen_as_arr), axis=None)
            gen.fitness[k] = 0
            for i in range(len(gen)):
                tup_i = cyclic_gen[i:i + k] # tuple of length k

                # Get tuple's fitness and sum it up
                f_i = self.dict_of_tup_to_fitness_dicts[k][str(tup_i)]
                gen.fitness[k] += f_i

    def calc_fitness(self, is_uncorrelated=False):
        """
        This function calculates the fitness of all genotype
        :param is_uncorrelated:
        :return:
        """

        genotypes = get_genotypes(self.genotype_len,  convert_to_genotype=True)
        for gen in genotypes:
            self.genotypes[gen.gen_as_str] = gen
            self.calc_single_gen_fitness(self.genotypes[gen.gen_as_str],
                                         is_uncorrelated)

    def is_maximum(self, gen: Genotype, k):
        """
        This function determines if a certain genotype is a local maximum
        according the its "k" fitness. (for every k, a certain genotype has
        a different fitness)
        :param gen:
        :param k:
        :return:
        """

        # Go through neighbors and check if one of them have larger fitness,
        # if yes - it not a local maximum. Else - it is
        neighbors = self.get_nearest_neighbors(gen)
        for neighbor in neighbors:
            if neighbor.fitness[k] > gen.fitness[k]:
                return False
        return True

    def get_nearest_neighbors(self, gen : Genotype):
        """
        Return list of all the neighbors of the genotype gen
        A genotype nearest neighbors are genotype who differ from it by a
        single entry
        :param gen:
        :return:
        """

        neighbors = []
        for i in range(len(gen)):
            if gen.gen_as_arr[i]:
                mask = np.ones(len(gen))
                mask[i] = 0
                neighbor_arr = np.logical_and(gen.gen_as_arr, mask)
            else:
                mask = np.zeros(len(gen))
                mask[i] = 1
                neighbor_arr = np.logical_or(gen.gen_as_arr, mask)

            neighbors.append(self.genotypes[str(neighbor_arr.astype(int))])

        return neighbors

    def get_random_neighbor(self, gen : Genotype):
        """
        Return a random neighbor of a gen. (For random walk in LFS)
        :param gen:
        :return:
        """

        neighbors = self.get_nearest_neighbors(gen)
        random_idx = np.random.randint(0, len(neighbors))
        return neighbors[random_idx]

    def get_highest_fitness_neighbor(self, gen : Genotype, k):
        """
        Return the neighbor with highest "k" fitness.
        If this neighbor's fitness is not larger than gen fitness - return None
        :param gen:
        :param k:
        :return:
        """

        neighbors = self.get_nearest_neighbors(gen)
        highest_fitness_neighbor = max(neighbors, key= lambda gen:
        gen.fitness[k])
        if highest_fitness_neighbor.fitness[k] < gen.fitness[k]:
            return None
        else:
            return highest_fitness_neighbor

    def count_maxima_for_k(self, k):
        """
        Given k, counts the number of local maxima in the model
        :param k:
        :return:
        """
        count = 0
        for gen in self.genotypes.values():
            if self.is_maximum(gen, k):
                count += 1
        return count

    def non_decreasing_single_trajectory(self, gen : Genotype, k):
        """
        This function simulates a non-decreasing trajectory in LFS that
        starts in genotype "gen" given a certain k and return the length
         of this trajectory.
        :param gen:
        :param k:
        :return:
        """

        this_gen = gen
        neighbor = self.get_highest_fitness_neighbor(this_gen, k)
        trajectory_len = 0
        while neighbor:
            trajectory_len += 1
            this_gen = neighbor
            neighbor = self.get_highest_fitness_neighbor(this_gen, k)

        return trajectory_len

    def non_decreasing_trajectory_for_k(self, k):
        """
        This function simulates a non-decreasing trajectories. Each of them
        starts at a one of the genotypes in the model. Eventually it
        returns a vector of trajectories lengths and the a vector of
        unique values that specifies observed length in the model
        :param k:
        :return:
        """
        trajectory_lens = []
        for gen in self.genotypes.values():
            trajectory_lens.append(self.non_decreasing_single_trajectory(
                gen, k))

        trajectory_lens = np.asarray(trajectory_lens)
        bins = np.unique(np.asarray(trajectory_lens))

        return trajectory_lens, bins

    def calc_spatial_correlation_for_k(self, k, max_delta):
        """
        This function simulates a random walk around the LSF model for a
        certain k. This walk is of length 2^N.
        At every step the fitness of the genotype is saved, and eventually
        creates a vector. The function return it's auto-correlation value
        :param k:
        :param max_delta:
        :return:
        """

        random_gen = random.choice(list(self.genotypes.values()))
        this_gen = random_gen
        fitness_vals = [this_gen.fitness[k]]
        for i in range(len(self.genotypes.keys())):
            this_gen = self.get_random_neighbor(this_gen)
            fitness_vals.append(this_gen.fitness[k])

        fitness_vals = np.asarray(fitness_vals)
        auto_correlation = auto_corr(fitness_vals, max_delta)
        return auto_correlation

    def calc_spatial_correlation(self):
        """
        This function calculates and plots the auto correlation values of a
        random walk around the model (NK / uncorrelated)
        :return:
        """

        plt.figure()
        max_delta = 10
        x_ticks = np.arange(-max_delta, max_delta + 1)
        if self.is_uncorrelated:
            auto_cor_vec = self.calc_spatial_correlation_for_k(0, max_delta)
            plt.plot(x_ticks, auto_cor_vec)

        else:
            for k in self.ks:
                auto_cor_vec = self.calc_spatial_correlation_for_k(k, max_delta)
                plt.plot(x_ticks, auto_cor_vec, label=k)
            plt.legend(title='k')

        plt.title(f"Auto correlation in {self.title_name} model. N = "
                  f"{self.genotype_len}")
        plt.xlabel("delta")
        plt.ylabel("Auto-correlation")
        plt.show()

    def maxima_count(self):
        """
        This function calculates and plots the number of local maxima in the
        model (NK / uncorrelated)
        :return:
        """

        plt.figure()
        x_ticks = np.asarray(self.ks)
        maxima_count_arr = []
        if self.is_uncorrelated:
            print(f"Number of mamixa in uncorrelated model (N = {self.genotype_len}): "
                  f"{self.count_maxima_for_k(0)}")
            return
        for k in self.ks:
            maxima_count_arr.append(self.count_maxima_for_k(k))

        plt.plot(x_ticks, np.asarray(maxima_count_arr))
        plt.title(f"Maxima count in {self.title_name} model. N = {self.genotype_len}")
        plt.xticks(x_ticks)
        plt.xlabel("K")
        plt.ylabel("Maxima count")
        plt.show()

    def non_decreasing_trajectory(self):
        """
        This function calculates and plots the distribution of a lengths of
        non-decreasing trajectories in the model (NK / uncorrelated)
        :return:
        """

        length_arr = []
        unified_bins = np.array([])
        if self.is_uncorrelated:
            lengths, bins = self.non_decreasing_trajectory_for_k(0)
            length_arr = [lengths]
            unified_bins = bins

        else:
            for k in self.ks:
                lengths, bins = self.non_decreasing_trajectory_for_k(k)
                length_arr.append(lengths)
                unified_bins = np.union1d(unified_bins, bins)

        plt.hist(length_arr, unified_bins, label=self.ks)
        if not self.is_uncorrelated:
            plt.legend(title='k')
        plt.title(f"Non decreasing trajectories lengths distribution in "
                  f"{self.title_name} model.\n N = {self.genotype_len}")
        plt.xlabel("trajectory length")
        plt.ylabel("number of genotypes")
        plt.xticks(unified_bins)
        plt.show()

    def run_manager(self):
        """
        Runner function to create graphs and prints according to exercise
        requirements
        :return:
        """

        self.calc_spatial_correlation()
        self.maxima_count()
        self.non_decreasing_trajectory()


if __name__ == '__main__':

    NK_fls = FLS(14, [1, 4, 10])
    uncorrelated_fls = FLS(10, None, is_uncorrelated=True)
    NK_fls.run_manager()
    uncorrelated_fls.run_manager()


