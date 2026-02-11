import os 
from typing import List, Tuple, Optional, Dict, Set
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

COLORS = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:cyan", "tab:olive", "tab:pink"]

def load_CDOP_instance(instance_id: str) -> Tuple[float, List[ArrayLike], List[ArrayLike]]:
    """Loads the CDOP instance from the resources folder."""
    with open(os.path.join(os.getcwd(), "resources", "CDOP", f"{instance_id}.txt"), "r") as file:
        contents = file.read()
        sections = contents.split("\n\n")
        
        # Load all information from the headers
        header_lines = sections[0].split("\n")
        n = int(header_lines[0].split()[-1])
        m = int(header_lines[1].split()[-1])
        t_max = float(header_lines[2].split()[-1])

        clusters: List[ArrayLike] = []
        scores_within_clusters: List[ArrayLike] = []
        for section in sections[1:]:
            matrix = np.genfromtxt(StringIO(section), delimiter=" ")
            if len(matrix.shape) == 1:
                matrix = np.reshape(matrix, (1, 3))

            clusters.append(matrix[:, :2])
            scores_within_clusters.append(matrix[:, 2])

    return (t_max, clusters, scores_within_clusters)

def plot_CDOP(clusters: List[ArrayLike], scores_within_clusters: List[ArrayLike], instance_id: Optional[str] = None, show: bool = True, sources: Dict[int, Set[int]] = {}, sinks: Dict[int, Set[int]] = {}):
    """Plots the given CDOP instance using matplotlib."""
    plt.style.use("seaborn-v0_8-ticks")
    plt.scatter(clusters[0][:, 0], clusters[0][:, 1], s=40, c="black", marker="s", zorder=4)
    plt.scatter(clusters[-1][:, 0], clusters[-1][:, 1], s=40, c="black", marker="D", zorder=4)
    
    for m, (cluster, scores, color) in enumerate(zip(clusters[1:-1], scores_within_clusters[1:-1], COLORS), 1):
        for i in range(len(cluster)):
            if (i not in sources[m]) and (i not in sinks[m]):
                plt.scatter(cluster[i, 0], cluster[i, 1], s = 20 * (1 + scores[i]), c=color)
            if i in sources[m]:
                plt.scatter(cluster[i, 0], cluster[i, 1], s = 20 * (1 + scores[i]), c=color, marker="s", zorder=4)
            if i in sinks[m]:
                plt.scatter(cluster[i, 0], cluster[i, 1], s = 20 * (1 + scores[i]), c=color, marker="D", zorder=4)

    if instance_id:
        plt.title(f"Instance: {instance_id}")

    if show:
        plt.show()

if __name__ == "__main__":
    instance_id = "q4.96.a"
    t_max, clusters, scores_within_clusters = load_CDOP_instance(instance_id)
    plot_CDOP(clusters, scores_within_clusters, instance_id)
    