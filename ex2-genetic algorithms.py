import itertools
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

C = 10


TOURNAMENT = "Tournament"
ROULETTE_WHEEL = "Roulette Wheel"
FITNESS_BY_LENGTH = "Adaptive Evolution"
EQUAL_FITNESS = "Neutral Evolution"
VAR_S_TOURNAMENT = "s"
VAR_MUTATION_P = "Mutation Rate"
VAR_POPULATION_SIZE = "Population size"


class TrajectoryDesc:

    def __init__(self, trajectory_as_array, fitness=None):
        self.as_array = list(trajectory_as_array)
        self.fitness = fitness
        self.fitness_type = FITNESS_BY_LENGTH if fitness is None else EQUAL_FITNESS
        self.len = self.get_len()

    def __eq__(self, other):
        return str(self.as_array) == str(other.as_array)

    def __hash__(self):
        return hash(str(self.as_array))

    def get_fitness(self):
        return (C - self.get_len()) if self.fitness_type == FITNESS_BY_LENGTH \
            else self.fitness

    def get_len(self):
        self.len = TRAJECTORIES_POPULATION.get_len(self)
        return self.len

    def to_string(self):
        return f"{str(self.as_array)}. length = {round(self.get_len(), 3)}"


class TrajectoriesPopulation:

    def __init__(self, city_num=10):
        self.trajectories = set()
        self.city_coordinates = []
        self.init_cities(city_num)
        self.init_trajectories(city_num)

    def init_cities(self, city_num):
        self.city_coordinates = np.random.uniform(size=(city_num, 2))

    def init_trajectories(self, city_num):
        cities_array = np.arange(city_num)
        self.trajectories = set(itertools.permutations(cities_array))

    def get_trajectories(self, population_size, set_equal_fitness):
        selected_trajectories = random.sample(self.trajectories,
                                              population_size)
        fitness = 1 if set_equal_fitness else None

        return [TrajectoryDesc(deepcopy(trajectory), fitness) for trajectory in
                selected_trajectories]

    def get_len(self, trajectory: TrajectoryDesc):
        cities_arr = trajectory.as_array
        length = 0
        for i in range(len(cities_arr) - 1):
            a_coordinates = self.city_coordinates[cities_arr[i]]
            b_coordinates = self.city_coordinates[cities_arr[i + 1]]
            length += distance.euclidean(a_coordinates,
                                         b_coordinates)

        return length


def roulette_wheel_selection(trajectory_descs: list,
                             population_size):

    fitness_array = [traj_desc.get_fitness() for traj_desc in
        trajectory_descs]
    sum_of_fitness = sum(fitness_array)
    weights = [fitness / sum_of_fitness for fitness in fitness_array]

    return deepcopy(random.choices(trajectory_descs, k=population_size,
                            weights=weights))


def creation_mutation(trajectory_desc: TrajectoryDesc, p):

    array = trajectory_desc.as_array
    for idx in range(len(array)):
        if random.random() < p:
            swap_idx = random.randint(0, len(array) - 1)

            val1 = array[idx]
            val2 = array[swap_idx]

            array[idx] = val2
            array[swap_idx] = val1

    trajectory_desc.as_array = array
    return deepcopy(trajectory_desc)


def create_mutations(population, p):

    return [creation_mutation(traj_desc, p) for traj_desc in population]


def tournament_selection(trajectory_descs : list, s,
                         population_size):
    assert s <= len(trajectory_descs)

    population_after_selection = []
    for _ in range(population_size):
        chromosomes = random.sample(trajectory_descs, k=s)
        max_chromosome = max(chromosomes, key=lambda x: x.get_fitness())
        population_after_selection.append(deepcopy(max_chromosome))

    # population after selection is the indices of the solutions
    return population_after_selection


def apply_elitism_func(population, N_e):
    assert N_e < len(population)
    if N_e == 0:
        return np.asarray([])
    return sorted(population, key=lambda x:x.get_len())[:N_e]


def apply_selection(population, selection_type, N_e, s=0):

    if selection_type == ROULETTE_WHEEL:
        return roulette_wheel_selection(population, len(population) - N_e)
    else:
        return tournament_selection(population, s, len(population) - N_e)


def single_model_runner(population_size, selection_type, fitness_type,
                        mutation_p, N_e, s=0):
    num_iters = 100
    population = TRAJECTORIES_POPULATION.get_trajectories(population_size,
                                                          fitness_type == EQUAL_FITNESS)

    min_traj_length = []
    mean_traj_length = []
    unique_variants_num = []
    traj_strings = []

    for _ in range(num_iters):
        traj_lenghts = np.array([trajectory_desc.get_len() for
                                 trajectory_desc in population])

        min_traj_length.append(traj_lenghts.min())
        mean_traj_length.append(traj_lenghts.mean())
        unique_variants_num.append(len(set(population)))

        elita = apply_elitism_func(population.copy(), N_e)
        rest = apply_selection(population.copy(), selection_type, N_e, s)

        rest_after_mutations = list(create_mutations(rest.copy(), mutation_p))
        population = rest_after_mutations + list(elita).copy()
        traj_strings.append([trajectory_desc.to_string() for
                                 trajectory_desc in population])

    txt = [r'$\mathbf{MODEL\:PARAMETERS}$', "\n"
           r"$\mathit{Population\:size} :$" + str(population_size),
           r"$\mathit{Mutation\:rate} :$" + str(mutation_p),
           r"$\mathit{Elitism\:parameter}: N_e=$" + str(N_e),
           r"$\mathit{Fitness\:type} :$" + fitness_type,
           r"$\mathit{Selection\:type} :$" + selection_type,
           r"$\mathit{Tournament\:parameter}: s=$" + str(s)]

    return unpack_results(
        [min_traj_length, mean_traj_length,
         unique_variants_num, txt])


def plot_results_manager(population_size, selection_type, fitness_type,
                         mutation_rate, N_e, s, var_range=None,
                         var_name=None):
    overall_data = []
    txt = ""
    if var_range is None:
        overall_data, txt = single_model_runner(population_size,
                                   selection_type=selection_type,
                                   fitness_type=fitness_type,
                                   mutation_p=mutation_rate, N_e=N_e, s=s)

    if var_name == VAR_S_TOURNAMENT:

        for var in var_range:
            data, txt = single_model_runner(population_size,
                                            selection_type=selection_type,
                                            fitness_type=fitness_type,
                                            mutation_p=mutation_rate, N_e=N_e,
                                            s=var)
            overall_data.append(data)

    elif var_name == VAR_MUTATION_P:
        for var in var_range:
            data, txt = single_model_runner(population_size,
                                            selection_type=selection_type,
                                            fitness_type=fitness_type,
                                            mutation_p=var, N_e=N_e, s=s)
            overall_data.append(data)

    else:  # var_name == POPULATION_SIZE:
        for var in var_range:
            data, txt = single_model_runner(var,
                                            selection_type=selection_type,
                                            fitness_type=fitness_type,
                                            mutation_p=mutation_rate, N_e=N_e, s=s)
            overall_data.append(data)

    data = rearrange_data(overall_data)
    var_range_as_arg = var_range if var_range else [0]
    plot_results(data, var_range_as_arg, var_name, txt)


def unpack_results(tuple4way):

    data = [tuple4way[0], tuple4way[1], tuple4way[2]]
    txt = tuple4way[3]
    return data, txt


def rearrange_data(data):

    if len(data) == 1:
        return [data]
    else:
        new_data = []
        for i in range(len(data[0])):
            axis_data = []
            for j in range(len(data)):
                axis_data.append(data[j][i])
            new_data.append(axis_data)

        return new_data


def refactor_txt(var_name, txt : list):

    if var_name == VAR_S_TOURNAMENT:
        del txt[6]
    elif var_name == VAR_MUTATION_P:
        del txt[2]
    elif var_name == VAR_POPULATION_SIZE:
        del txt[1]

    return '\n'.join(txt)


def plot_results(data, variable_ranges, var_name, txt):

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    combine_plots(axes, data, variable_ranges)

    # set titles
    y_labels = ["Trajectory length (Euclidean distance)",
                "Trajectory length (Euclidean distance)",
                "Unique variants number"]
    titles = ["Minimum trajectory length",
              "Mean trajectory length",
              "Unique variants number"]

    for idx, ax in enumerate(axes):
        ax.set_xlabel("iter #")
        ax.set_ylabel(y_labels[idx])
        ax.set_title(titles[idx])

    x_place = -5
    y_place = max([max(inner_data) for inner_data in data[0]]) * 1.1
    axes[0].text(x_place, y_place, refactor_txt(var_name, txt), bbox=props,
                 verticalalignment='bottom')
    plt.suptitle("Trajectory length and unique variants number by iter#")

    if len(variable_ranges) > 1:
        plt.legend(title=var_name)

    plt.subplots_adjust()

    plt.show()


def combine_plots(axes, data, variable_range):

    for idx, ax in enumerate(axes):
        axis_data = data[idx]
        for j in range(len(variable_range)):
            to_plot = axis_data[j]
            ax.plot(to_plot, label=variable_range[j])


def quest_1():
    # Question 1 (What is neutral ?)
    population_sizes = [30, 100, 300]
    plot_results_manager(0, selection_type=ROULETTE_WHEEL,
                         fitness_type=EQUAL_FITNESS,
                         mutation_rate=0, N_e=0, s=0,
                         var_range=population_sizes,
                         var_name=VAR_POPULATION_SIZE)


def quest_2():
    # Question 2
    population_sizes = [30, 100, 300]
    plot_results_manager(0, selection_type=ROULETTE_WHEEL,
                         fitness_type=FITNESS_BY_LENGTH,
                         mutation_rate=0, N_e=0, s=0,
                         var_range=population_sizes,
                         var_name=VAR_POPULATION_SIZE)


def quest_3():
    # Question 3
    s_vals = [2, 3, 4]
    plot_results_manager(100, selection_type=TOURNAMENT,
                         fitness_type=FITNESS_BY_LENGTH,
                         mutation_rate=0.005, N_e=0, s=0,
                         var_range=s_vals, var_name=VAR_S_TOURNAMENT)


def quest_4():
    # Question 4
    mutation_rates = [0.1, 0.05, 0.025]
    plot_results_manager(300, selection_type=ROULETTE_WHEEL,
                         fitness_type=FITNESS_BY_LENGTH,
                         mutation_rate=0, N_e=0, s=0,
                         var_range=mutation_rates, var_name=VAR_MUTATION_P)


def quest_5():
    # Question 5
    mutation_rates = [0.1, 0.05, 0.025]
    plot_results_manager(300, selection_type=ROULETTE_WHEEL,
                         fitness_type=FITNESS_BY_LENGTH,
                         mutation_rate=0, N_e=30, s=0,
                         var_range=mutation_rates, var_name=VAR_MUTATION_P)


def run_manager():
    quest_1()
    quest_2()
    quest_3()
    quest_4()
    quest_5()


TRAJECTORIES_POPULATION = TrajectoriesPopulation()
run_manager()
