import numpy as np
from scipy.spatial import Delaunay
from typing import Dict, List, Tuple
from numpy.typing import ArrayLike
from scipy.spatial import distance_matrix
from helpers import load_CDOP_instance, plot_CDOP

def find_sources_and_sinks(clusters: List[ArrayLike], l_max: int = 8) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Finds the sources and sink candidates within each cluster, using the distance matrix as well as the intra-cluster Delaunay triangulation."""
    sources, sinks = {m: set() for m in range(len(clusters))}, {m: set() for m in range(len(clusters))}

    for m in range(1, len(clusters)):
        # Find source & sink candidates
        intra_cluster_distances = distance_matrix(clusters[m - 1], clusters[m])
        source_candidates = set(np.argsort(np.min(intra_cluster_distances, axis=0))[:l_max])
        sink_candidates = set(np.argsort(np.min(intra_cluster_distances, axis=1))[:l_max])

        # Remove candidates which are not within a Delaunay simplex, with the other cluster.
        triangulation = Delaunay(np.vstack([clusters[m - 1], clusters[m]]))
        N_prev = len(clusters[m - 1])
        for simplex in triangulation.simplices:
            if any([n < N_prev for n in simplex]) and any([n >= N_prev for n in simplex]):
                for n in simplex:
                    if n in sink_candidates:
                        sink_candidates.remove(n)
                        sinks[m - 1].add(n)

                    elif n - N_prev in source_candidates:
                        source_candidates.remove(n - N_prev)
                        sources[m].add(n - N_prev)
    return sources, sinks

def compute_upper_bound_for_maximal_t_max_within_cluster (t_max: float, m: int, clusters: List[ArrayLike], sources: Dict[int, List[int]], sinks: Dict[int, List[int]]) -> float:
    """Allocates the t-max, that we can utilize within the cluster m, since we need to go from source to sink within each cluster and from sink to source between clusters."""
    M = len(clusters) - 2
    decreased_t_max = t_max
    for m_prime in range(1, len(clusters) - 1):
        # We need to be able to go from the source to the sink within m'
        sources_in_cluster_m_prime, sinks_in_cluster_m_prime = [clusters[m_prime][i] for i in sources[m_prime]], [clusters[m_prime][i] for i in sinks[m_prime]]
        if m != m_prime: 
            decreased_t_max -= np.min(distance_matrix(sources_in_cluster_m_prime, sinks_in_cluster_m_prime))
        
        # We need to be able to traverse the distance from cluster m' - 1 to cluster m'
        sinks_in_cluster_before_cluster_m_prime = [clusters[m_prime - 1][i] for i in sinks[m_prime - 1]]
        decreased_t_max -= np.min(distance_matrix(sinks_in_cluster_before_cluster_m_prime, sources_in_cluster_m_prime))

    # We need to be able to reach the sink from cluster M - 1
    sinks_in_cluster_before_actual_sink = [clusters[M][i] for i in sinks[M]]
    decreased_t_max -= np.min(distance_matrix(sinks_in_cluster_before_actual_sink, clusters[-1]))

    return decreased_t_max

if __name__ == "__main__":
    instance_id = "q1.32.a"
    t_max, clusters, scores_within_clusters = load_CDOP_instance(instance_id)
    print(clusters[0][0])
    sources, sinks = find_sources_and_sinks(clusters)
    plot_CDOP(clusters, scores_within_clusters, instance_id, sources = sources, sinks = sinks)
    print(compute_upper_bound_for_maximal_t_max_within_cluster(t_max, 1, clusters, sources, sinks))
