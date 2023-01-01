from typing import List
import sys
from BuildingBlocksGenetator import BuildingBlockHolder, BuildingBlock, \
    get_other_idx
from random import randint, sample
import numpy as np
import utils
from copy import deepcopy

MAX_NUM_OF_TRIANGLES_INIT = 30
TARGET_X = 100
TARGET_Y = 100
TARGET_POINT = np.array([TARGET_X, TARGET_Y])
MAX_EDGE_LEN_VAL = 10
MAX_TRIANGLE_AREA = MAX_EDGE_LEN_VAL  # assuming 90 degree triangle
# with two edges of length BUILDING_BLOCK_MAX_EDGE (i.e =
# BUILDING_BLOCK_MAX_EDGE * 2 /2 = BUILDING_BLOCK_MAX_EDGE)
NUMBER_OF_BUILDING_BLOCKS_THRESH = 10
DIST_FITNESS_CONSTANT = np.linalg.norm(TARGET_POINT)
BB_NUM_FITNESS_CONSTANT = DIST_FITNESS_CONSTANT / MAX_TRIANGLE_AREA

FIRST_OF_PAIR = 0
SECOND_OF_PAIR = 0


class GeneticBridge:

    def __init__(self, building_block_population):  # random init

        self.population = building_block_population
        self.size = randint(10, MAX_NUM_OF_TRIANGLES_INIT)
        self.building_blocks = []
        self.blocks = []
        self.edges_pairs = List[tuple]
        self.order = list(np.random.permutation(self.size))

        for _ in range(self.size):
            self.building_blocks.append(
                building_block_population.get_random_building_block())

        self.fitness_func = None

        self.order_by_permutation()
        self.order_edges_in_pairs()
        self.generate_coordinates()

    def __eq__(self, other):
        if type(other) != GeneticBridge:
            return False
        else:
            if len(other.edges_pairs) != self.edges_pairs:
                return False
            for idx in range(len(self.edges_pairs)):
                this_pair = self.edges_pairs[idx]
                other_pair = other.edges_pairs[idx]
                if this_pair[0] != other_pair[0] or this_pair[1] != \
                        other_pair[1]:
                    return False

            return True

    def __hash__(self):
        return hash(str(self.edges_pairs))

    def order_edges_in_pairs(self):

        self.edges_pairs = [None] * (self.size - 1)
        for i in range(len(self.edges_pairs)):
            pair = self.get_legal_pairing(i)
            self.update_pair(pair, i, i)

    def order_by_permutation(self):

        self.blocks = [None] * self.size

        for idx, building_blk_idx in enumerate(self.order):
            self.blocks[idx] = self.building_blocks[
                building_blk_idx]

    def get_legal_pairing(self, idx_of_1st_triangle):

        is_legal = False
        pair = tuple()

        while not is_legal:
            edge_idx1 = randint(0, 2)
            edge_idx2 = randint(0, 2)
            pair = (edge_idx1, edge_idx2)
            is_legal = self.is_legal_pairing(pair, idx_of_1st_triangle)

        return pair

    def is_legal_pairing(self, pair, idx_of_1st_triangle):

        ret_val, _ = self.get_conflicted_edge(pair, idx_of_1st_triangle)
        return ret_val == -1

    def get_conflicted_edge(self, pair, idx_of_tri):

        # since there isn't a neighbor on the right
        only_edge_1_relevant = \
            idx_of_tri + 1 >= len(self.blocks)

        edge_of_1st, edge_of_2nd = pair

        is_edge_taken1 = \
            self.blocks[idx_of_tri].is_edge_taken[
                edge_of_1st]

        if only_edge_1_relevant:
            return (edge_of_1st, FIRST_OF_PAIR) if is_edge_taken1 else (-1, -1)

        is_edge_taken2 = self.blocks[idx_of_tri + 1
                                     ].is_edge_taken[
            edge_of_2nd]

        if is_edge_taken1:
            return edge_of_1st, FIRST_OF_PAIR
        if is_edge_taken2:
            return edge_of_2nd, SECOND_OF_PAIR

        return -1, -1

    def update_pair(self, pair, idx_of_first_triangle, idx_of_pair):
        self.edges_pairs[idx_of_pair] = pair

        edge_of_1st, edge_of_2nd = pair

        self.blocks[idx_of_first_triangle].is_edge_taken[
            edge_of_1st] \
            = True

        if idx_of_first_triangle + 1 < len(self.blocks):
            self.blocks[
                idx_of_first_triangle + 1].is_edge_taken[
                edge_of_2nd] = True

    def generate_coordinates(self):

        self.blocks[0].set_pivot()
        prev_triangle = self.blocks[0]
        for i in range(1, len(self.blocks), 1):
            edge1, edge2 = self.edges_pairs[i - 1]
            self.blocks[i].align_to(prev_triangle, edge1,
                                    edge2, i)

            prev_triangle = self.blocks[i]

    def plot(self, indices_to_paint):

        coors_list = [triangle.get_coors() for triangle in
                      self.blocks]
        end_x, end_y = self.get_end_point()
        utils.plot_triangle(coors_list, title=f"whole bridge",
                            pts_x=[end_x], pts_y=[end_y],
                            indices_to_paint=indices_to_paint)

    def plot_with_target(self, path=None, title=""):

        coors_list = [triangle.get_coors() for triangle in
                      self.blocks]
        end_x, end_y = self.get_end_point()
        utils.plot_triangle(coors_list, title=title,
                            pts_x=[end_x, TARGET_X], pts_y=[end_y,
                                                            TARGET_Y],
                            path=path)

    def plot_growth(self, indices_to_add=None):

        coors_list = [triangle.get_coors() for triangle in
                      self.blocks]
        for i in range(1, len(coors_list) + 1):
            utils.plot_triangle(coors_list[:i], title=f"with {i} triangles",
                                # path=f"random_bridge/full_bridge_{i}",
                                indices_to_paint=indices_to_add,
                                pts_x=[TARGET_X], pts_y=[TARGET_Y])

    def get_end_point(self):

        last_triangle: BuildingBlockHolder = self.blocks[-1]
        if len(self.edges_pairs) == 0:
            return np.array([0, 0])

        joint_edge_of_last_triangle = self.edges_pairs[-1][1]

        coors_indices = last_triangle.get_edge_coors_indices(
            joint_edge_of_last_triangle)

        end_coor_idx = get_other_idx(coors_indices[0], coors_indices[1])

        return last_triangle.get_coors()[end_coor_idx]

    def get_dist_from_target(self):
        return np.linalg.norm(TARGET_POINT - self.get_end_point())

    def get_fitness(self):

        dist_from_target = self.get_dist_from_target()
        if dist_from_target < NUMBER_OF_BUILDING_BLOCKS_THRESH:
            protection_offset = (DIST_FITNESS_CONSTANT - dist_from_target) *\
                                1.5
            return protection_offset + BB_NUM_FITNESS_CONSTANT - self.size
        else:
            return DIST_FITNESS_CONSTANT - dist_from_target

    def remove_pair(self, idx):

        edge_of_1st, edge_of_2nd = self.edges_pairs[idx]
        self.blocks[idx].is_edge_taken[edge_of_1st] = \
            False
        self.blocks[idx + 1].is_edge_taken[edge_of_2nd] = \
            False

    def random_new_edge(self, exclude):

        rand_val = randint(0, 2)
        while rand_val == exclude:
            rand_val = randint(0, 2)

        return rand_val

    def replace_edge_in_pair(self, pair, pair_idx, which_one, new_edge):

        new_pair = deepcopy(list(pair))
        edge_to_remove = pair[which_one]

        triangle_idx = pair_idx + which_one
        self.blocks[triangle_idx].is_edge_taken[
            edge_to_remove] = False

        new_pair[which_one] = new_edge
        self.update_pair(tuple(new_pair), pair_idx, pair_idx)

    def handle_pairs_mismatch(self, pair, pair_idx, which_one_changed,
                              which_one_conflicted):

        conflicted_edge = pair[which_one_conflicted]
        replace_edge = self.random_new_edge(conflicted_edge)

        if which_one_changed == -1:  # a pair that needs to change due to
            # triangles changes that came before him
            conflicted_edge = pair[which_one_conflicted]
            replace_edge = self.random_new_edge(conflicted_edge)
            self.replace_edge_in_pair(pair, pair_idx,
                                      FIRST_OF_PAIR, replace_edge)

        else:
            if which_one_changed == FIRST_OF_PAIR:
                # previous pair needs an updated
                other_pair = self.edges_pairs[pair_idx - 1]
                self.replace_edge_in_pair(other_pair, pair_idx - 1,
                                          SECOND_OF_PAIR, replace_edge)

            else:  # the edge changed was for second of pair
                if pair_idx + 1 >= len(self.edges_pairs):
                    # ignore pair change is last block
                    return

                other_pair = self.edges_pairs[pair_idx + 1]
                self.replace_edge_in_pair(other_pair, pair_idx + 1,
                                          FIRST_OF_PAIR, replace_edge)

    def set_new_pair(self, pair, pair_idx, which_one_changed):

        self.remove_pair(pair_idx)
        if not self.is_legal_pairing(pair, pair_idx):
            _, which_one_conflicted = self.get_conflicted_edge(pair, pair_idx)
            self.handle_pairs_mismatch(pair, pair_idx, which_one_changed,
                                       which_one_conflicted)

        self.update_pair(pair, pair_idx, pair_idx)

    def set_new_pairs(self, new_pairs, fixed_indices):

        for idx, pair in enumerate(new_pairs):
            self.set_new_pair(pair, idx, fixed_indices[idx])
        self.generate_coordinates()

    def add_new_building_block(self, building_block: BuildingBlockHolder,
                               position: int):

        # update building block
        self.blocks.insert(position, building_block)

        # update pair information
        if position - 1 >= len(self.edges_pairs):
            pair = self.get_legal_pairing(
                len(self.blocks) - 1)
            self.edges_pairs.append(pair)
            self.update_pair(pair, position - 1, position - 1)
            return

        pair_in_position = self.edges_pairs[position - 1]
        edge1 = randint(0, 2)
        edge2 = self.random_new_edge(exclude=edge1)

        pair1 = tuple([pair_in_position[0], edge1])
        pair2 = tuple([edge2, pair_in_position[1]])
        self.update_pair(pair1, position - 1, position - 1)
        self.edges_pairs.insert(position, pair2)
        self.update_pair(pair2, position, position)

    def change_bridge_size(self, new_size):

        str_to_dbg = 'expansion'
        growth_factor = abs(new_size - self.size)
        if new_size == self.size:  # nothing changed
            return
        elif self.size < new_size:
            idx = self.handle_expansion(new_size)
        elif self.size > new_size:
            idx = self.handle_shrinkage(new_size)
            str_to_dbg = 'shrinkage'


        if len(self.blocks) == len(self.edges_pairs):
            print(f"bad resize : #of blocks :{new_size},"
                  f"             # should have {new_size - 1} pairs."
                  f"             Actual size is {len(self.edges_pairs)}"
                  f" occured during {str_to_dbg} of {growth_factor}")
            exit(1)
        self.generate_coordinates()

        # uncomment this for debug
        # self.plot(idx)

    def handle_shrinkage(self, new_size):

        bb_num_to_remove = self.size - new_size
        indices_to_remove = sorted(sample(range(self.size),
                                          bb_num_to_remove), reverse=True)

        new_bb_list = []
        for idx, bb in enumerate(self.blocks):
            if idx not in indices_to_remove:
                new_bb_list.append(bb)

        self.blocks = new_bb_list

        # remove from tuples list
        for idx in indices_to_remove:

            lhs_pair = None
            rhs_pair = None
            first_half = []

            if idx > 0:
                lhs_pair = self.edges_pairs[idx - 1]

            if idx > 1:
                first_half = self.edges_pairs[:idx - 1]

            if idx < len(self.edges_pairs):
                rhs_pair = self.edges_pairs[idx]

            second_half = self.edges_pairs[idx + 1:]

            joint_pair = []
            if rhs_pair is not None and lhs_pair is not None:
                joint_pair = [tuple([lhs_pair[0], rhs_pair[1]])]

            self.edges_pairs = first_half + joint_pair + second_half

        self.size = new_size
        self.align_taken_edges()

    def align_taken_edges(self):

        for idx, bb in enumerate(self.blocks):

            bb.is_edge_taken = [False] * 3

            if idx < len(self.edges_pairs) - 1:
                pair2_edge = self.edges_pairs[idx][FIRST_OF_PAIR]
                bb.is_edge_taken[pair2_edge] = True

            if idx > 0:
                pair1_edge = self.edges_pairs[idx - 1][SECOND_OF_PAIR]
                bb.is_edge_taken[pair1_edge] = True

    def handle_expansion(self, new_size):

        num_to_add = new_size - self.size
        indices_to_add = sorted(sample(range(new_size), num_to_add))

        for position in indices_to_add:
            bb = self.population.get_random_building_block()
            self.add_new_building_block(bb, position)

        self.size = new_size

        return indices_to_add
