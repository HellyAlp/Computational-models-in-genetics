import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib
MARGIN = 1.3


def plot_triangle(coors_list, ax=None, title="", pts_x=None, pts_y=None,
                  path=None, indices_to_paint=None):
    if pts_y is None:
        pts_y = []
    if pts_x is None:
        pts_x = []
    ax_was_none = ax is None
    if ax is None:
        plt.figure()
        ax = plt.gca()

    bounding_box = np.array(list(find_bounding_box(coors_list, pts_x, pts_y)))
    bounding_box *= MARGIN # add margins to graph
    max_x, min_x, max_y, min_y = tuple(bounding_box)

    for idx, coors in enumerate(coors_list):

        fc = (0, 0, 1, 0.5)
        if indices_to_paint:
            if idx in indices_to_paint:
                fc = (1, 0, 0, 0.5)
        ax.add_patch(Polygon(np.array(coors), edgecolor='black',  fc=fc))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title(title)
    ax.scatter(pts_x, pts_y)

    if path:
        plt.savefig(path)
        plt.close()
        return

    if ax_was_none:
        plt.show()


def find_bounding_box(list_coors, pts_x, pts_y):

    xs = []
    ys = []
    for triangle in list_coors:
        for coor in triangle:
            xs.append(coor[0])
            ys.append(coor[1])

    xs.extend(pts_x)
    ys.extend(pts_y)

    max_x = max(xs)
    min_x = min(xs)
    max_y = max(ys)
    min_y = min(ys)

    return max_x, min_x, max_y, min_y


def find_ratio(N):

    n_sqrt = np.sqrt(N)
    height = int(np.ceil(n_sqrt))
    width = int(np.floor(n_sqrt))

    return height, width


def plot_N_times(list_to_plot, edge_pairs, path=None):

    h, w = find_ratio(len(list_to_plot))
    _, axes = plt.subplots(ncols=w, nrows=h)

    for i in range(h):
        for j in range(w):
            data = list_to_plot[i * w + j]
            pair = edge_pairs[i * w + j]
            edge1, edge2 = pair
            plot_triangle(data, axes[i][j], f"Edges are {edge1} and {edge2}")

    if path:
        plt.savefig(path)
    else:
        plt.show()

