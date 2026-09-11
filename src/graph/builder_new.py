import os
import json
import torch
import pandas as pd
from src.utils import read_config
from src.graph.gaugegraphnew import GaugeGraphNew 

def build_baseline_graphs(
    gauge_config="config/graphs/gauge_baseline.yaml", 
    data_dir="database/standard", 
    output_dir="database/graphs/baseline"
):
    """
    Ingests standard data, constructs baseline gauge graphs, 
    and saves the PyG HeteroData objects to disk.
    """
    # first read the config file to get the parameters for graph construction
    config = read_config(gauge_config)
    
    mapping_df = pd.read_csv(
        f"{data_dir}/stations.csv", 
        dtype={"station_id": str}
    )

    mapping_df = mapping_df.rename(columns={"station_id": "id"})
    
    if "use_in_study" in mapping_df.columns:
        mapping_df = mapping_df[mapping_df["use_in_study"] == True]
        
    with open("config/splits/sydney_5fold.json", "r") as f:
        split_info = json.load(f)["folds"]

    long_gauge_df = pd.read_csv(
        f"{data_dir}/gauge_hourly.csv", 
        dtype={"station_id": str}
    )
    # Pivot to wide format as shown in the Component 02 spec
    data_df = long_gauge_df.pivot(
        index="timestamp", 
        columns="station_id", 
        values="rainfall_mm"
    )
        
    for i, fold in enumerate(split_info):
        print(f"Building fold {i}...")
        
        graph_builder = GaugeGraphNew(
            data_df=data_df,
            mapping_df=mapping_df,
            split_info=fold,
            knn=config["k_neighbours"],
            distance=config["distance"],
            edge_normalisation=config["edge_normalisation"],
            include_lpe=config["include_lpe"],
            lpe_dimensions=config["lpe_dimensions"],
            layer_is_directed=config["layer_is_directed"]
        )
        
        fold_dir = f"{output_dir}/fold-{i}"
        os.makedirs(fold_dir, exist_ok=True)
        
        torch.save(graph_builder.get_train_heterodata(), f"{fold_dir}/train.pt")
        torch.save(graph_builder.get_validation_heterodata(), f"{fold_dir}/validation.pt")
        torch.save(graph_builder.get_test_heterodata(), f"{fold_dir}/test.pt")
        
    print(f"✅ All baseline graphs saved to {output_dir}")