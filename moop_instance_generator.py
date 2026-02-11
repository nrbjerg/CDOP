from helpers import load_CDOP_instance
from decomposition import find_sources_and_sinks, compute_upper_bound_for_maximal_t_max_within_cluster
import os 
import numpy as np

PATH_TO_CDOP_INSTANCES = os.path.join(os.getcwd(), "resources", "CDOP")
PATH_TO_MOOP_INSTANCES = os.path.join(os.getcwd(), "resources", "MOOP")

for file_name in os.listdir(PATH_TO_CDOP_INSTANCES):
    instance_id = ".".join(file_name.split(".")[:-1])
    N = int(instance_id.split(".")[1])
    t_max, clusters, scores_within_clusters = load_CDOP_instance(instance_id)
    sources, sinks = find_sources_and_sinks(clusters, l_max = int(np.floor(np.sqrt(2 * N))))
    for m in range(1, len(clusters) - 1):
        t_max_within_m = compute_upper_bound_for_maximal_t_max_within_cluster(t_max, m, clusters, sources, sinks)
        sources_within_cluster_m = [clusters[m][i] for i in sources[m]]
        scores_of_sources_within_cluster_m = [scores_within_clusters[m][i] for i in sources[m]]
        sinks_within_cluster_m = [clusters[m][i] for i in sinks[m]]
        scores_of_sinks_within_cluster_m = [scores_within_clusters[m][i] for i in sources[m]]
        nodes_within_cluster_m = [clusters[m][i] for i in range(N) if (i not in sinks[m]) and (i not in sources[m])]
        scores_of_nodes_within_cluster_m = [scores_within_clusters[m][i] for i in range(N) if (i not in sinks[m]) and (i not in sources[m])]

        convert_to_triples = lambda points, scores: "\n".join([f"{p[0]} {p[1]} {s}" for p, s in zip(points, scores)])

        contents = f"tmax {t_max_within_m:.2f}\nN {N}\n\n" + "\n\n".join([convert_to_triples(sources_within_cluster_m, scores_of_sources_within_cluster_m), convert_to_triples(nodes_within_cluster_m, scores_of_nodes_within_cluster_m), convert_to_triples(sinks_within_cluster_m, scores_of_sinks_within_cluster_m)])
        with open(os.path.join(PATH_TO_MOOP_INSTANCES, f"{instance_id}.{m}.txt"), "w+") as file:
            file.write(contents)
