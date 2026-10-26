"""The architecture of the released weights, and where to find them."""

from typing import Any

# Changing a dimension or a layer count stops the released weights loading; changing the rest
# changes the embedding they compute. Dropout is the exception: the model runs in eval mode.
ZAO_CONFIG: dict[str, Any] = {
    "general": {
        "conformer_embed_dim": 1024,
        "mol_embed_dim": 2048,
        "max_atoms": 100,
        "num_conformers": 10,
        "dropout": 0.1,
    },
    "encoder": {
        "input_dim": 7,
        "hidden_dim": 256,
        "depth": 3,
        "num_nearest_neighbors": 16,
        "norm_coors": True,
        "coor_weights_clamp_value": 2.0,
        "num_queries": 64,
        "substructure_embed_dim": 256,
        "attention_temperature_init": 1.0,
        "edge_dim": 5,
        "mol_graph_num_layers": 3,
        "conformer_pooling": {
            "layers": 3,
            "nhead": 8,
            "dim_feedforward": 512,
            "num_rbf": 16,
            "rbf_max_distance": 20.0,
        },
    },
    "ensemble": {
        "nhead": 8,
        "dim_feedforward_ratio": 2,
        "num_layers": 6,
        "use_boltzmann_aggregation": True,
        "boltzmann_temperature": 1.0,
    },
}


# The Hugging Face Hub repository holding model.safetensors.
DEFAULT_WEIGHTS = "syntheticgestalt/zao"
