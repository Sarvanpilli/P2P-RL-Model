import torch

def get_grid_graph(n_nodes: int):
    """
    Constructs the physical topology of the Microgrid.
    Returns:
        edge_index (torch.LongTensor): Graph connectivity shape [2, num_edges]
    """
    edges = []
    
    if n_nodes == 4:
        # Simple bus topology: 0-1-2-3
        edges = [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (0, 3), (3, 0) # Ring connection for robustness
        ]
    elif n_nodes == 14:
        # Simplified IEEE 14-bus test feeder topology
        edges = [
            (0, 1), (1, 0), (0, 4), (4, 0),
            (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3), (3, 6), (6, 3), (3, 8), (8, 3),
            (4, 5), (5, 4),
            (5, 10), (10, 5), (5, 11), (11, 5), (5, 12), (12, 5),
            (6, 7), (7, 6), (6, 8), (8, 6),
            (8, 9), (9, 8), (8, 13), (13, 8),
            (9, 10), (10, 9),
            (12, 13), (13, 12)
        ]
    else:
        # Default fallback: Fully connected grid if arbitrary nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append((i, j))
                    
    # Format for PyTorch Geometric [2, E]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    return edge_index
