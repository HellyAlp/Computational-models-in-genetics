from copy import deepcopy, copy
import random
from Bridge import *
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import os
import tqdm
import threading

TOURNAMENT = "Tournament"
ROULETTE_WHEEL = "Roulette Wheel"
FITNESS_BY_LENGTH = "Adaptive Evolution"
EQUAL_FITNESS = "Neutral Evolution"
VAR_S_TOURNAMENT = "s"
PLOT_OUTPUT_DIR = "BridgeEvolutionPlots\\no_blocks_variance_fix_1.0.0"

VAR_MUTATION_P = "Mutation Rate"
VAR_POPULATION_SIZE = "Population size"
VAR_GENERATION_NUMBER = "Generation number"
VAR_ELITA = "Elita factor"

SIZE_VARIANCE_IN_MUTATION = 3
ANGLE_MAX_VAL = 178
ANGLE_VAR_IN_MUTATION = 10
EDGE_VARIANCE_IN_MUTATION = 3
ANGLE_MIN_VAL = 1
EDGE_MIN_VAL = 1
PARENT_PLOT_PATH = ""


def create_random_building_block():
    tmp_triangle = BuildingBlock()

    # this code remains unused after moving to stage 2 (no variance in blocks)
    # can be commented in for applying variance into the model
    ang = randint(20, 90)
    edge1 = randint(1, MAX_EDGE_LEN_VAL)
    edge2 = randint(1, MAX_EDGE_LEN_VAL)
    ##

    tmp_triangle.generate_triangle(60, 5, 5)
    return BuildingBlockHolder(tmp_triangle)


class BuildingBlockPopulation:

    def __init__(self, population_size):
        self.size = population_size
        self.population = []
        self.init_blocks()

    def init_blocks(self):
        self.population = [create_random_building_block() for _ in range(
            self.size)]

    def get_random_building_block(self):
        return create_random_building_block()


class BridgesPopulation:

    def __init__(self, population_size, building_blocks_population):
        self.size = population_size
        self.population = []
        self.init_bridges(building_blocks_population)

    def init_bridges(self, bb_population):
        for _ in range(self.size):
            bridge = GeneticBridge(bb_population)
            self.population.append(bridge)


def gather_bridges(bridges: List[GeneticBridge]):

    bridges_count = dict()
    bridges_fitness = dict()
    min_fitness = 1000
    for bridge in bridges:
        if bridge not in bridges_count.keys():
            bridges_count[bridge] = 0

            fitness = bridge.get_fitness()
            bridges_fitness[bridge] = fitness

            if fitness < min_fitness:
                min_fitness = fitness

        bridges_count[bridge] += 1

    min_fitness = min(0, min_fitness)
    weights = []
    for bridge in bridges_count.keys():
        actual_fitness = bridges_fitness[bridge] + abs(min_fitness)
        overall_in_population = actual_fitness * bridges_count[bridge]
        weights.append(overall_in_population)

    return weights

def roulette_wheel_selection(bridges: list,
                             population_size):
    fitness_array = np.array([bridge.get_fitness() for bridge in
                     bridges])
    min_val = fitness_array.min()
    if min_val < 0:
        fitness_array += np.abs(min_val)

    weights = list(fitness_array / fitness_array.sum())

    return deepcopy(random.choices(bridges, k=population_size,
                                   weights=weights)), gather_bridges(bridges)


# this function remained unused in when moving to stage 2 simulations
def mutate_triangle(triangle_holder: BuildingBlockHolder):
    triangle = triangle_holder.triangle

    angle_a = triangle.angels[0]
    edge_b = triangle.edges_length[1]
    edge_c = triangle.edges_length[2]

    angle_a_mutated = angle_a + randint(-ANGLE_VAR_IN_MUTATION,
                                        ANGLE_VAR_IN_MUTATION)
    edge_b_mutated = edge_b + randint(-EDGE_VARIANCE_IN_MUTATION,
                                      EDGE_VARIANCE_IN_MUTATION)
    edge_c_mutated = edge_c + randint(-EDGE_VARIANCE_IN_MUTATION,
                                      EDGE_VARIANCE_IN_MUTATION)

    angle_a_mutated = min(max(angle_a_mutated, ANGLE_MIN_VAL), ANGLE_MAX_VAL)
    edge_b_mutated = min(max(edge_b_mutated, MAX_EDGE_LEN_VAL), EDGE_MIN_VAL)
    edge_c_mutated = min(max(edge_c_mutated, MAX_EDGE_LEN_VAL), EDGE_MIN_VAL)

    mutated_triangle = BuildingBlock()
    mutated_triangle.generate_triangle(angle_a_mutated, edge_b_mutated,
                                       edge_c_mutated)

    return BuildingBlockHolder(mutated_triangle)

# this function remained unused in when moving to stage 2 simulations
def blocks_mutation(genetic_bridge: GeneticBridge, p):
    mutated_blocks = []
    for trig_holder in genetic_bridge.blocks:
        if random.random() < p:
            mutated_blocks.append(mutate_triangle(trig_holder))
        else:
            mutated_blocks.append(trig_holder)

    genetic_bridge.blocks = mutated_blocks
    return deepcopy(genetic_bridge)


def pairs_mutation(genetic_bridge: GeneticBridge, p):

    if random.random() < p:
        mutated_pairs = genetic_bridge.edges_pairs.copy()
        changed_indices = [-1] * len(genetic_bridge.edges_pairs)

        idx = randint(0, len(genetic_bridge.edges_pairs) - 1)

        pair = genetic_bridge.edges_pairs[idx]
        which_edge = random.randint(0, 1)
        other_edge = int(not bool(which_edge))  # opposite position

        new_edge = random.randint(0, 2)
        constant_edge = pair[other_edge]

        new_pair = [0] * 2
        new_pair[other_edge] = constant_edge
        new_pair[which_edge] = new_edge

        pair = tuple(new_pair)
        changed_indices[idx] = which_edge
        mutated_pairs[idx] = pair
        genetic_bridge.set_new_pairs(mutated_pairs, changed_indices)

    return genetic_bridge


def bridge_mutation(genetic_bridge: GeneticBridge, p):
    # size_mutation
    if random.random() < p:
        size = genetic_bridge.size + randint(-SIZE_VARIANCE_IN_MUTATION,
                                             SIZE_VARIANCE_IN_MUTATION)

        size = max(1, size)     # size can't be smaller than 1
        genetic_bridge.change_bridge_size(size)

    # uncomment this for applying blocks mutations
    # genetic_bridge = blocks_mutation(genetic_bridge, p)
    return pairs_mutation(genetic_bridge, p)


def create_mutations(population, p):
    return [bridge_mutation(bridge, p) for bridge in population]


def tournament_selection(bridges: list, s,
                         population_size):
    assert s <= len(bridges)

    population_after_selection = []
    for _ in range(population_size):
        chromosomes = random.sample(bridges, k=s)
        max_chromosome = max(chromosomes, key=lambda x: x.get_fitness())
        population_after_selection.append(deepcopy(max_chromosome))

    # population after selection is the indices of the solutions
    return population_after_selection


def apply_elitism_func(population, N_e):
    assert N_e <= len(population)
    if N_e == 0:
        return np.asarray([])
    return sorted(population, key=lambda x: x.get_dist_from_target())[:N_e]


def apply_selection(population, selection_type, N_e, s=0):
    if selection_type == ROULETTE_WHEEL:
        return roulette_wheel_selection(population, len(population) - N_e)
    else:
        return tournament_selection(population, s, len(population) - N_e)


def single_model_runner(generation_num, population_size, selection_type
                        , mutation_p, N_e, output_path, s=0):

    BB_population = BuildingBlockPopulation(population_size)
    bridges = BridgesPopulation(population_size, BB_population).population

    pie_path = path.join(output_path, "distribution")
    if not path.exists(pie_path):
        os.makedirs(path.join(output_path, "distribution"))

    min_dist_from_target = []
    mean_dist_from_target = []
    unique_variants_num = []

    txt = [r'$\mathbf{MODEL\:PARAMETERS}$', "\n"
           r"$\mathit{Population\:size} :$" + str(population_size),
           r"$\mathit{Mutation\:rate} :$" + str(mutation_p),
           r"$\mathit{Elitism\:parameter}: N_e=$" + str(N_e),
           r"$\mathit{Selection\:type} :$" + selection_type,
           r"$\mathit{Tournament\:parameter}: s=$" + str(s)]

    # verbose prints
    print("#" * 100)
    print("Started evolution simulation with the following params :" +
          "\nPopulation size :" + str(population_size) +
          "\nMutation rate :" + str(mutation_p) +
          "\nElitism parameter: N_e= " + str(N_e) +
          "\nSelection type :" + selection_type +
          "\nTournament parameter: s=" + str(s) + "\n")

    prev_bridge = bridges[0]
    for idx in tqdm.tqdm(range(generation_num)):
        # gather stats

        dists = np.array([bridge.get_dist_from_target() for bridge in bridges])

        distance_from_target = dists.min()
        min_dist_from_target.append(dists.min())
        mean_dist_from_target.append(dists.mean())
        unique_variants_num.append(len(set(bridges)))

        idx_of_max_fitness = np.argmin(dists)
        best_bridge = bridges[idx_of_max_fitness]

        if hash(best_bridge) != hash(prev_bridge):
            best_bridge.plot_with_target(path=f'{output_path}\\after_{idx}_gens',
                                         title=f"distance from target: {distance_from_target}")

            prev_bridge = deepcopy(best_bridge)

        # run evolution
        bridges_before_this_generation = bridges.copy()
        try:
            elita = apply_elitism_func(bridges.copy(), N_e)
            rest, weights = apply_selection(bridges.copy(), selection_type, N_e, s)

            rest_after_mutations = list(create_mutations(rest.copy(),
                                                          mutation_p))
            bridges = rest_after_mutations + list(elita).copy()

            # plotting
            plt.figure()
            labels = np.arange(len(weights)) + 1
            plt.pie(np.array(weights), labels=labels)
            plt.savefig(f'{output_path}\\distribution\\after_{idx}_gens')
            plt.close()

        except Exception as e:
            print(f"Exception occurred during generation {idx}."
                  f" Skipping this generation. Error info : {e}")
            bridges = bridges_before_this_generation

    print("Evolution finished")
    print("#" * 100)

    return unpack_results(
        [min_dist_from_target, mean_dist_from_target,
         unique_variants_num, txt])


def refactor_txt(var_name, txt: list):
    if var_name == VAR_S_TOURNAMENT:
        del txt[5]
    elif var_name == VAR_MUTATION_P:
        del txt[2]
    elif var_name == VAR_POPULATION_SIZE:
        del txt[1]

    return '\n'.join(txt)


def unpack_results(tuple4way):
    data = [tuple4way[0], tuple4way[1], tuple4way[2]]
    txt = tuple4way[3]
    return data, txt


def generate_path(generation_num, population_size, selection_type,
                  mutation_rate, N_e, var_range=None,
                  var_name=None):

    generation_val = f"num_gens_{generation_num}"
    population_val = f"pop_size_{population_size}"
    mutation_rate_val = f"mutation_rate_{str(mutation_rate).replace('.', '')}"
    N_e_val = f"elita_num_{N_e}"
    selection_val = f"selection_type_{selection_type}"

    GEN_IDX = 0
    POP_IDX = 1
    MUTATION_IDX = 2
    ELITA_IDX = 3

    dir_name = [generation_val, population_val,
                mutation_rate_val, N_e_val, selection_val]

    if var_name == VAR_MUTATION_P:
        dir_name.pop(MUTATION_IDX)
    if var_name == VAR_POPULATION_SIZE:
        dir_name.pop(POP_IDX)
    if var_name == VAR_GENERATION_NUMBER:
        dir_name.pop(GEN_IDX)
    if var_name == VAR_ELITA:
        dir_name.pop(ELITA_IDX)

    dir_name_str = '__'.join(dir_name)
    parent_dir_name = path.join(PLOT_OUTPUT_DIR, dir_name_str)
    if var_name is None:
        if not path.exists(parent_dir_name):
            os.makedirs(parent_dir_name)
    else:
        for var in var_range:
            inner_dir = path.join(parent_dir_name, str(var))
            if not path.exists(inner_dir):
                os.makedirs(inner_dir)

    return parent_dir_name


def plot_results_manager(generation_num, population_size, selection_type,
                         mutation_rate, N_e, s, var_range=None,
                         var_name=None):
    overall_data = []
    txt = ""
    global PARENT_PLOT_PATH
    PARENT_PLOT_PATH = generate_path(generation_num, population_size,
                                     selection_type,
                                     mutation_rate, N_e, var_range, var_name)

    if var_range is None:
        output_path = PARENT_PLOT_PATH
        overall_data, txt = single_model_runner(generation_num,
                                                population_size,
                                                selection_type=selection_type,
                                                mutation_p=mutation_rate,
                                                N_e=N_e, s=s,
                                                output_path=output_path)

    elif var_name == VAR_S_TOURNAMENT:

        for var in var_range:
            output_path = path.join(PARENT_PLOT_PATH, str(var))
            data, txt = single_model_runner(generation_num, population_size,
                                            selection_type=selection_type,
                                            mutation_p=mutation_rate, N_e=N_e,
                                            s=var,
                                            output_path=output_path)
            overall_data.append(data)

    elif var_name == VAR_MUTATION_P:
        for var in var_range:
            output_path = path.join(PARENT_PLOT_PATH, str(var))
            data, txt = single_model_runner(generation_num, population_size,
                                            selection_type=selection_type,
                                            mutation_p=var, N_e=N_e, s=s,
                                            output_path=output_path)
            overall_data.append(data)

    elif var_name == VAR_GENERATION_NUMBER:
        for var in var_range:
            output_path = path.join(PARENT_PLOT_PATH, str(var))
            data, txt = single_model_runner(var, population_size,
                                            selection_type=selection_type,
                                            mutation_p=mutation_rate, N_e=N_e, s=s,
                                            output_path=output_path)
            overall_data.append(data)

    else:  # var_name == POPULATION_SIZE:
        for var in var_range:
            output_path = path.join(PARENT_PLOT_PATH, str(var))
            data, txt = single_model_runner(generation_num, var,
                                            selection_type=selection_type,
                                            mutation_p=mutation_rate, N_e=N_e,
                                            s=s,
                                            output_path=output_path)
            overall_data.append(data)

    data = rearrange_data(overall_data)
    var_range_as_arg = var_range if var_range else [0]
    plot_results(data, var_range_as_arg, var_name, txt)


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


def plot_results(data, variable_ranges, var_name, txt):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    fig, axes = plt.subplots(1, len(variable_ranges), figsize=(18, 6))

    combine_plots(axes, data, variable_ranges)

    # set titles
    y_labels = ["Distance from target (Euclidean distance)",
                "Distance from target (Euclidean distance)",
                "Unique variants number"]
    titles = ["Minimum distance from target",
              "Mean distance from target",
              "Unique variants number"]

    for idx, ax in enumerate(axes):
        ax.set_xlabel("generation #")
        ax.set_ylabel(y_labels[idx])
        ax.set_title(titles[idx])

    x_place = -5
    y_place = max([max(inner_data) for inner_data in data[0]]) * 1.1
    axes[0].text(x_place, y_place, refactor_txt(var_name, txt), bbox=props,
                 verticalalignment='bottom')
    plt.suptitle("Distance from target and unique variants number by "
                 "generation#")

    if len(variable_ranges) > 1:
        plt.legend(title=var_name)

    plt.subplots_adjust()

    plot_path = path.join(PARENT_PLOT_PATH, "summary")
    plt.savefig(plot_path)
    # plt.show()


def combine_plots(axes, data, variable_range):
    for idx, ax in enumerate(axes):
        axis_data = data[idx]
        for j in range(len(variable_range)):
            to_plot = axis_data[j]
            ax.plot(to_plot, label=variable_range[j])


### examples for simulations definitions ###

def sim_1():
    population_sizes = [30, 100, 300]
    plot_results_manager(generation_num=100,
                         population_size=0,  # Would be filled using
                                             # var_range param
                         selection_type=ROULETTE_WHEEL,
                         mutation_rate=0,
                         N_e=0,
                         s=0,
                         var_range=population_sizes,
                         var_name=VAR_POPULATION_SIZE)


def sim_2():
    s_vals = [2, 3, 4]
    plot_results_manager(generation_num=100,
                         population_size=100,
                         selection_type=TOURNAMENT,
                         mutation_rate=0.005,
                         N_e=0,
                         s=0,  # Would be filled using
                               # var_range param
                         var_range=s_vals, var_name=VAR_S_TOURNAMENT)


def sim_3():
    mutation_rates = [0.1, 0.2, 0.5]
    plot_results_manager(generation_num=100,
                         population_size=300,
                         selection_type=ROULETTE_WHEEL,
                         mutation_rate=0,   # Would be filled using
                                            # var_range param
                         N_e=0,
                         s=0,
                         var_range=mutation_rates, var_name=VAR_MUTATION_P)


def sim_4():
    mutation_rates = [0.1, 0.2, 0.5]
    plot_results_manager(generation_num=100,
                         population_size=300,
                         selection_type=ROULETTE_WHEEL,
                         mutation_rate=0,   # Would be filled using
                                            # var_range param
                         N_e=30,
                         s=0,
                         var_range=mutation_rates, var_name=VAR_MUTATION_P)


def sim_5():
    generations_number = [200, 500, 1000]
    plot_results_manager(generation_num=0,    # Would be filled using
                                              # var_range param
                         population_size=300,
                         selection_type=ROULETTE_WHEEL,
                         mutation_rate=0.2,
                         N_e=30,
                         s=0,
                         var_range=generations_number,
                         var_name=VAR_GENERATION_NUMBER)


def sim_6():
    mutation_rates = [0.1, 0.2, 0.5]
    plot_results_manager(generation_num=2000,
                         population_size=300,
                         selection_type=ROULETTE_WHEEL,
                         mutation_rate=0,   # Would be filled using
                                              # var_range param
                         N_e=30,
                         s=0,
                         var_range=mutation_rates,
                         var_name=VAR_MUTATION_P)

def sim_7():
    s = [2, 3, 4]
    plot_results_manager(generation_num=500,
                         population_size=300,
                         selection_type=TOURNAMENT,
                         mutation_rate=0.1,
                         N_e=30,
                         s=0,
                         var_range=s,
                         var_name=VAR_S_TOURNAMENT)


def run_manager():

    # The written simulations in functions sim_1(),...,sim_7() are subsets of
    # simulation we have conducted during our projects

    sim_1()
    sim_2()
    sim_3()
    sim_4()
    sim_5()
    sim_6()
    sim_7()
