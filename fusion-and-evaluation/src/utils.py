from typing import Dict, Tuple

import networkx as nx
import numpy as np

def get_sample_sizes(n_nodes):
    
    multipliers = np.array([10, 30, 100, 300, 1000])
    sizes = (multipliers * n_nodes).astype(int)

    return np.unique(sizes).tolist()
    
    
def get_regime_samples(n_nodes, factors=(10, 100, 1000)):

    sizes = [int(f * n_nodes) for f in factors]

    return sizes
    

def build_oracle_prior(
        dag_true: nx.DiGraph,
        p_correct: float = 0.9,
) -> Dict[Tuple[str, str], float]:

    return {(u, v): p_correct for u, v in dag_true.edges()}