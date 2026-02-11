import numpy as np
import os 
import matplotlib.pyplot as plt 
from shapely import Polygon, Point
import numpy as np
from typing import List
from itertools import product
from numpy.typing import ArrayLike
from python_tsp.exact import solve_tsp_dynamic_programming
from helpers import plot_CDOP

M = 4
T_MAX_LEVELS = 4

# Cluster sizes.
N_LARGE = 128 
N_MEDIUM = 96
N_SMALL = 64 
N_TINY = 32

def generate_polygons(std: float, m: int, l: int) -> List[Polygon]:
    """Generates the convex hull of m polygons, using a set of l polygons"""
    polygons = []
    while len(polygons) < m:
        mean = np.random.uniform(0, 1, size=2)
        cov = np.array([[std**2, 0], [0, std**2]])
        points = np.random.multivariate_normal(mean, cov, size=l)

        convex_hull = Polygon(points).convex_hull
        for poly in polygons:
            if convex_hull.intersects(poly):
                break
        else:
            polygons.append(convex_hull)

    return polygons

def uniform_points_in_polygon(polygon: Polygon, n: int, seed=None):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds

    points = []
    while len(points) < n:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            points.append([x, y])

    return np.array(points)

def save_CDOP(clusters: List[ArrayLike], scores_within_clusters: List[ArrayLike], layout: int):
    """Saves CDOP instance, within a txt file in a similar format as the Chao instances."""
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    distance_matrix = np.zeros((M + 1, M + 1))
    for i in range(M + 1):
        for j in range(M + 1):
            if j == 0:
                distance_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[M + 1])
            else:
                distance_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            
    permutation, tsp_distance = solve_tsp_dynamic_programming(distance_matrix)
    ordering = permutation + [M + 1]
    clusters = [clusters[i] for i in ordering]

    for i, n in product(range(1, T_MAX_LEVELS + 1), [N_TINY, N_SMALL, N_MEDIUM, N_LARGE]):
        with open(os.path.join(os.getcwd(), "resources", "CDOP", f"q{layout}.{n}.{chr(ord('a') + i - 1)}.txt"), "w+") as file:
            content = f"n {n}\nm {M}\ntmax {(1 + 2 * i / T_MAX_LEVELS) * tsp_distance:.2f}\n\n" + "\n\n".join(["\n".join([f"{p[0]:.2f} {p[1]:.2f} {s:.2f}" for p, s in zip(cluster[:n], scores[:n])]) for cluster, scores in zip(clusters, scores_within_clusters)])

            file.write(content)

if __name__ == "__main__":
    # Generate the 1st instances, with a distinct source and sink and distinct clusters.
    first_generation = True
    while first_generation or input("Happy with the layout (y/n)? ") != "y":
        first_generation = False
        polygons = generate_polygons(0.12, M, 32)
        clusters = [np.array([[0, 0.5]])] + [uniform_points_in_polygon(poly, N_LARGE) for poly in polygons] + [np.array([[1, 0.5]])]
        scores_within_clusters = [[0]] + [np.random.uniform(0.2, 1.0, size=N_LARGE) for _ in range(M)] + [[0]]
        plot_CDOP(clusters, scores_within_clusters)
    
    save_CDOP(clusters, scores_within_clusters, 1)

    ## Generate the 2nd instances 
    first_generation = True
    while first_generation or input("Happy with the layout (y/n)? ") != "y":
        first_generation = False
        polygons = generate_polygons(0.12, M, 32)
        clusters = [np.array([[0.5, 0.5]])] + [uniform_points_in_polygon(poly, N_LARGE) for poly in polygons] + [np.array([[0.5, 0.5]])]
        scores_within_clusters = [[0]] + [np.random.uniform(0.2, 1.0, size=N_LARGE) for _ in range(M)] + [[0]]
        plot_CDOP(clusters, scores_within_clusters)
    
    save_CDOP(clusters, scores_within_clusters, 2)

    ## Generate the 3rd instances 
    first_generation = True
    while first_generation or input("Happy with the layout (y/n)? ") != "y":
        first_generation = False
        polygons = generate_polygons(0.12, M, 32)
        clusters = [np.array([[0.0, 0.333]])] + [uniform_points_in_polygon(poly, N_LARGE) for poly in polygons] + [np.array([[0.0, 0.666]])]
        scores_within_clusters = [[0]] + [np.random.uniform(0.2, 1.0, size=N_LARGE) for _ in range(M)] + [[0]]
        plot_CDOP(clusters, scores_within_clusters)
    
    save_CDOP(clusters, scores_within_clusters, 3)

    ## Generate the 4th instances 
    clusters = [np.array([[0.0, 0.5]])] + [np.array([[np.random.uniform((m + 0.1) / M, (m + 0.9) / M), np.random.uniform(0.0, 1.0)] for _ in range(N_LARGE)]) for m in range(M)] + [np.array([[1.0, 0.5]])]
    scores_within_clusters = [[0]] + [np.random.uniform(0.2, 1.0, size=N_LARGE) for _ in range(M)] + [[0]]
    save_CDOP(clusters, scores_within_clusters, 4)


