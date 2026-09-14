import os
import json
from pathlib import Path
import sys
import numpy as np


import torch
import pandas as pd
from src.utils import read_config
from src.graph.gaugegraphnew import GaugeGraphNew #file is "graph" not "graphs" so changed accordingly 
from src.graph.radargraphnew import RadarGraphNew 

def build_baseline_graphs(
    gauge_config="config/graphs/gauge_baseline.yaml", 
    data_dir="database/standard", 
    baseline_dir="database/graphs/baseline",
    radar_dir="database/graphs/radar_fusion"
    # config/graphs/ is the contract path (AGENTS.md, IMPLEMENTATION_HANDOFF.md, 04 requirements)
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
        # Filter the mapping_df to only include rows where "use_in_study" is True
        # also reset the index after filtering so dataframes are aligned properly for graph construction
        mapping_df = mapping_df[mapping_df["use_in_study"] == True].reset_index(drop=True)
        #remake the order column to be sequential after filtering
        mapping_df["order"] = range(len(mapping_df))
        
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

    # # NEW PADDING LOGIC: Fill missing stations with 0.0
    # missing_stations = set(mapping_df["id"]) - set(data_df.columns)
    # for station in missing_stations:
    #     data_df[station] = 0.0

    # NEW PADDING LOGIC: Fill missing stations with NaN so the mask catches them
    missing_stations = set(mapping_df["id"]) - set(data_df.columns)
    for station in missing_stations:
        data_df[station] = np.nan


    #build the radar graph
    # Radar is optional. Component 02 does not write radar tables yet, and the
    # gauge-only baseline must stay buildable without them
    # (04 contract: "Build gauge-only first and radar second").
    radar_layer = None
    radar_nodes_df = None
    radar_nodes_path = Path(data_dir) / "radar_nodes.csv"
    radar_hourly_path = Path(data_dir) / "radar_hourly.csv"

    if radar_nodes_path.exists() and radar_hourly_path.exists():
        radar_nodes_df = pd.read_csv(radar_nodes_path)
        long_radar_df = pd.read_csv(radar_hourly_path)

        # Pivot and reshape to [952, T, 1]
        wide_radar_df = long_radar_df.pivot(
            index="timestamp", columns="node_id", values="rain_rate"
        )

        # missing = int(wide_radar_df.isna().to_numpy().sum())
        # if missing:
        #     raise ValueError(
        #         f"{radar_hourly_path} has {missing} missing radar values. "
        #         "Filling them with 0.0 would tell the model that no rain fell "
        #         "where nothing was measured (requirements README rule 4; "
        #         "04 rule 5). Carrying gaps needs a validity channel like "
        #         "satellite's [nodes, T, 2], which changes in_channels_dict - "
        #         "agree it with the Component 02 and 06 owners first."
        #     )

        # radar_tensor = torch.tensor(
        #     wide_radar_df.values.T, dtype=torch.float32
        # ).unsqueeze(-1)

        radar_builder = RadarGraphNew(
            data_df=wide_radar_df,
            coords_df=radar_nodes_df,
            knn=8,
            distance_metric="haversine_km"
        )

        radar_layer = radar_builder.heterodata
    else:
        print(f"no radar tables in {data_dir}; building gauge-only graphs")

    #build the satellite graph

    #space to build any other graphs in future implementations
        
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

        #save baseline graphs
        baseline_fold_dir = f"{baseline_dir}/fold-{i}"
        os.makedirs(baseline_fold_dir, exist_ok=True)
        
        torch.save(graph_builder.get_train_heterodata(), f"{baseline_fold_dir}/train.pt")
        torch.save(graph_builder.get_validation_heterodata(), f"{baseline_fold_dir}/validation.pt")
        torch.save(graph_builder.get_test_heterodata(), f"{baseline_fold_dir}/test.pt") 

        #add layer to gauge graph
        if radar_layer is not None:
            graph_builder.add_heterodata(
                heterodata_layer=radar_layer,
                coords=radar_nodes_df,
                layer_name="radar",
                knn=4
            )

            train_fused, val_fused, test_fused = graph_builder.get_fused_heterodata()

            #save radar and gauge fused graphs
            radar_fold_dir = f"{radar_dir}/fold-{i}"
            os.makedirs(radar_fold_dir, exist_ok=True)

            torch.save(train_fused, f"{radar_fold_dir}/train.pt")
            torch.save(val_fused, f"{radar_fold_dir}/validation.pt")
            torch.save(test_fused, f"{radar_fold_dir}/test.pt")

    print(f"Baseline graphs saved to {baseline_dir}")
    if radar_layer is not None:
        print(f"Combined radar graphs saved to {radar_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Component 04: Graph Construction Pipeline")
    
#     # Define the command line flags, keeping your current paths as safe defaults
#     parser.add_argument("--config", type=str, default="config/graphs/gauge_baseline.yaml", help="Path to graph configuration YAML")
#     parser.add_argument("--data-dir", type=str, default="database/standard", help="Directory containing standard input CSVs")
#     parser.add_argument("--baseline-dir", type=str, default="database/graph/baseline", help="Output directory for gauge graphs")
#     parser.add_argument("--radar-dir", type=str, default="database/graph/radar_fusion", help="Output directory for radar graphs")
    
#     args = parser.parse_args()
    
#     print(f"Starting Component 04 Graph Construction using config: {args.config}")
    
#     # Pass the command line arguments into your function
#     build_baseline_graphs(
#         gauge_config=args.config,
#         data_dir=args.data_dir,
#         baseline_dir=args.baseline_dir,
#         radar_dir=args.radar_dir
#     )
    
#     print("Graph construction complete!")

#run by calling: uv run python src/graph/builder_new.py --config config/graphs/gauge_baseline.yaml
    # since gauge_hourly and stations are in tests/graphs instead call:
    # uv run python src/graph/builder_new.py --config config/graphs/gauge_baseline.yaml --data-dir tests/graphs
    # for the time being

if __name__ == "__main__":
    # arg[1] is config, arg[2] is data_dir. Fallback to defaults if missing.
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/graphs/gauge_baseline.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "tests/graphs"

    print(f"Starting Graph Construction using config: {config_path}")
    
    build_baseline_graphs(
        gauge_config=config_path,
        data_dir=data_path
    )
    
    print("Graph construction complete!")

    # run by calling uv run python -m src.graph.builder_new config/graphs/gauge_baseline.yaml tests/data/graphs
    # uv run python -m src.graph.builder_new config/graphs/gauge_baseline.yaml tests/data/graphs
