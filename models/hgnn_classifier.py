import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class HGNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=3):
        super().__init__()
        
        # 1. Temporal Encoder (Recur-then-Convolve)
        # Processes the time-series history for each node independently
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)
        
        # 2. Heterogeneous Message Passing Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # Intra-layer message passing (learning local spatial textures)
                ('radar', 'connects', 'radar'): SAGEConv(-1, hidden_channels),
                ('satellite', 'connects', 'satellite'): SAGEConv(-1, hidden_channels),
                ('gauge', 'connects', 'gauge'): SAGEConv(-1, hidden_channels),
                
                # Cross-layer message passing (funneling remote sensing data into the gauges)
                ('radar', 'connects', 'gauge'): SAGEConv(-1, hidden_channels),
                ('satellite', 'connects', 'gauge'): SAGEConv(-1, hidden_channels)
            }, aggr='mean')
            self.convs.append(conv)
            
        # 3. The Classification Readout Head
        # Maps the latent gauge embeddings to a raw logit. NO Sigmoid here.
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        
        # Step 1: Apply temporal encoding to each node type independently
        for node_type, x in x_dict.items():
            # Expected x shape: [num_nodes, seq_len, features]
            lstm_out, _ = self.lstm(x)
            # Extract the final hidden state of the sequence to represent current temporal context
            h_dict[node_type] = lstm_out[:, -1, :] 
            
        # Step 2: Execute spatial message passing across the full heterogeneous graph
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: torch.relu(h) for key, h in h_dict.items()}
            
        # Step 3: Generate final classification logits exclusively at the target gauge locations
        logits = self.head(h_dict['gauge'])
        
        return logits