from utils import *

TRIANGLE_FACTOR = 3


def get_other_idx(idx1, idx2):
    indices = [i for i in range(TRIANGLE_FACTOR)]
    indices.remove(idx1)
    indices.remove(idx2)
    return indices[0]


class BuildingBlock:

    def __init__(self):
        self.angels = [0] * TRIANGLE_FACTOR
        self.edges_length = [0] * TRIANGLE_FACTOR

    def generate_triangle(self, angle_a: int, edge_b: int, edge_c: int):
        angle_a_rad = np.deg2rad(angle_a)
        edge_a = np.sqrt((edge_b ** 2) + (edge_c ** 2) - (2 * edge_b * edge_c *
                                                          np.cos(angle_a_rad)))

        angle_b_rad = np.arcsin(edge_b * np.sin(angle_a_rad) / edge_a)
        angle_b_deg = np.rad2deg(angle_b_rad)
        angle_c = 180 - angle_a - angle_b_deg

        self.angels = [angle_a, angle_b_deg, angle_c]
        self.edges_length = [edge_a, edge_b, edge_c]

    def get_edges(self):
        return self.edges_length

    def get_angels(self):
        return self.angels

    def get_fitness(self):

        return self.calc_area()

    def calc_area(self):

        angle_a = self.angels[0]
        edge_b = self.edges_length[1]
        edge_c = self.edges_length[2]

        return (edge_b * edge_c) * np.sin(np.deg2rad(angle_a)) / 2


class BuildingBlockHolder:

    def __init__(self, building_block: BuildingBlock):
        self.triangle = building_block
        self.coors = self.get_polygon()
        self.is_edge_taken = [False] * TRIANGLE_FACTOR
        self.is_aligned = False

    def align_to(self, other_transformer, other_edge_idx, my_edge_idx,
                 debug_text):
        other_v1, other_v2 = self.get_edge_coors_indices(other_edge_idx)
        my_v1, my_v2 = self.get_edge_coors_indices(my_edge_idx)

        self.coors = self.align_to_helper(other_transformer.coors, other_v1,
                                          other_v2, my_v1, my_v2, debug_text)

    def get_edge_coors_indices(self, edge_idx: int):
        assert 0 <= edge_idx < TRIANGLE_FACTOR

        first_coor_idx = edge_idx
        second_coor_idx = (edge_idx + 1) % TRIANGLE_FACTOR

        return first_coor_idx, second_coor_idx

    def get_edge_coor(self, edge_idx:int):

        coor1_idx, coor2_idx = self.get_edge_coors_indices(edge_idx)
        return self.coors[coor1_idx], self.coors[coor2_idx]

    def get_polygon(self, base_x=0, base_y=0, tilt=0, pivot_idx=0, idx=1):
        assert idx != pivot_idx

        other_edge = get_other_idx(pivot_idx, idx)
        x_a, y_a = base_x, base_y
        edge_a2b = self.triangle.edges_length[idx]  # the edge in front of C
        edge_a2c = self.triangle.edges_length[other_edge]  # the edge in front
        # of B

        inner_ang_rad = np.deg2rad(tilt - self.triangle.angels[pivot_idx])
        context_ang_rad = np.deg2rad(tilt)

        x_b = x_a + edge_a2b * np.cos(context_ang_rad)
        y_b = y_a + edge_a2b * np.sin(context_ang_rad)

        x_c = x_a + edge_a2c * np.cos(inner_ang_rad)
        y_c = y_a + edge_a2c * np.sin(inner_ang_rad)

        pts = np.array([[x_a, y_a], [x_b, y_b], [x_c, y_c]])

        return pts

    def set_pivot(self):

        self.coors = self.get_polygon()

    def __get_angle(self, v1, v2):
        # calc tilt
        x1, y1 = v1
        x2, y2 = v2

        delta_x = x2 - x1
        delta_y = y2 - y1

        ang_rad = np.arctan2(delta_y, delta_x)

        return np.rad2deg(ang_rad)

    def __get_aligned_coors(self, other_coors, other_v1_idx, other_v2_idx,
                            my_v1_idx,
                            my_v2_idx):
        ang = self.__get_angle(other_coors[other_v1_idx],
                               other_coors[other_v2_idx])
        x1, y1 = other_coors[other_v1_idx]
        return self.get_polygon(x1, y1, ang, my_v1_idx, my_v2_idx)

    def __test_alignment(self, other_coors, other_v1_idx, other_v2_idx,
                         try1_coors, try2_coors, debug_text):
        other_extra_idx = get_other_idx(other_v1_idx, other_v2_idx)

        try1_extra_coor = try1_coors[TRIANGLE_FACTOR - 1]
        try2_extra_coor = try2_coors[TRIANGLE_FACTOR - 1]

        other_edge_coors = other_coors[other_extra_idx]

        dist_try1 = np.linalg.norm(try1_extra_coor - other_coors[
            other_extra_idx])

        dist_try2 = np.linalg.norm(try2_extra_coor - other_edge_coors)

        path = r'random_bridge/debug_' + str(debug_text)
        plot_triangle([try1_coors, try2_coors, other_coors], title="DEBUG "
                                                                   + str(debug_text),
                      pts_x=[try1_extra_coor[0], try2_extra_coor[0],
                             other_edge_coors[0]],
                      pts_y=[try1_extra_coor[1], try2_extra_coor[1],
                             other_edge_coors[1]],
                      path=path)

        return try1_coors if dist_try1 > dist_try2 else try2_coors

    def align_to_helper(self, other_coors, other_v1_idx, other_v2_idx,
                        my_v1_idx, my_v2_idx, debug_txt):
        try1_coors = self.__get_aligned_coors(other_coors, other_v1_idx,
                                              other_v2_idx, my_v1_idx,
                                              my_v2_idx)

        if not check_for_triangle_intersection(try1_coors, other_coors):
            coors = try1_coors
        else:
            coors = self.__get_aligned_coors(other_coors, other_v2_idx,
                                              other_v1_idx, my_v1_idx,
                                              my_v2_idx)

        third_idx = get_other_idx(my_v1_idx, my_v2_idx)
        return self.reorder_and_return(coors, [my_v1_idx, my_v2_idx, third_idx])

    @staticmethod
    def reorder_and_return(coors, order):

        ordered_coors = [None] * TRIANGLE_FACTOR
        for idx, ordered_idx in enumerate(order):
            ordered_coors[ordered_idx] = coors[idx]

        return ordered_coors

    def get_coors(self):
        return self.coors

    def set_aligned(self):
        self.is_aligned = True

    def get_fitness(self):
        return self.triangle.get_fitness()

#####
# This code block is inspired by stackoverflow
# https://stackoverflow.com/questions/1585459/whats-the-most-efficient-way-to-detect-triangle-triangle-intersections
# It serves the purpose of checking if 2 triangles are intersected / overlap
# This is done to make sure that two consecutive building blocks will be
# connected in the must area-efficient way
###


def check_for_triangle_intersection(t1, t2):

    if is_two_line_intersect(t1[0], t1[1], t2[0], t2[1]): return True
    if is_two_line_intersect(t1[0], t1[1], t2[0], t2[2]): return True
    if is_two_line_intersect(t1[0], t1[1], t2[1], t2[2]): return True
    if is_two_line_intersect(t1[0], t1[2], t2[0], t2[1]): return True
    if is_two_line_intersect(t1[0], t1[2], t2[0], t2[2]): return True
    if is_two_line_intersect(t1[0], t1[2], t2[1], t2[2]): return True
    if is_two_line_intersect(t1[1], t1[2], t2[0], t2[1]): return True
    if is_two_line_intersect(t1[1], t1[2], t2[0], t2[2]): return True
    if is_two_line_intersect(t1[1], t1[2], t2[1], t2[2]): return True

    in_tri = True
    in_tri = in_tri and point_in_triangle2(t1[0], t1[1], t1[2], t2[0])
    in_tri = in_tri and point_in_triangle2(t1[0], t1[1], t1[2], t2[1])
    in_tri = in_tri and point_in_triangle2(t1[0], t1[1], t1[2], t2[2])

    if in_tri:
        return True

    in_tri = True
    in_tri = in_tri and point_in_triangle2(t2[0], t2[1], t2[2], t1[0])
    in_tri = in_tri and point_in_triangle2(t2[0], t2[1], t2[2], t1[1])
    in_tri = in_tri and point_in_triangle2(t2[0], t2[1], t2[2], t1[2])

    if in_tri:
        return True

    return False


def is_two_line_intersect(v1, v2, v3, v4):

    d = (v4[1] - v3[1]) * (v2[0] - v1[0]) - (v4[0] - v3[0]) * (
            v2[1] - v1[1])
    u = (v4[0] - v3[0]) * (v1[1] - v3[1]) - (v4[1] - v3[1]) * (
            v1[0] - v3[0])
    v = (v2[0] - v1[0]) * (v1[1] - v3[1]) - (v2[1] - v1[1]) * (
            v1[0] - v3[0])
    if d < 0:
        u, v, d = -u, -v, -d
    return (0 <= u <= d) and (0 <= v <= d)


def point_in_triangle2(A, B, C, P):

    v0 = [C[0] - A[0], C[1] - A[1]]
    v1 = [B[0] - A[0], B[1] - A[1]]
    v2 = [P[0] - A[0], P[1] - A[1]]
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    u = cross(v2, v0)
    v = cross(v1, v2)
    d = cross(v1, v0)
    if d < 0:
        u, v, d = -u, -v, -d
    return u >= 0 and v >= 0 and (u + v) <= d
