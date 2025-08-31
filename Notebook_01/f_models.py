# ==============================================================================
#             REGRESSION MODEL DEFINITIONS AND LOSS FUNCTION
# ==============================================================================
import torch
import torch.nn as nn
#  
# class PredictorNet(nn.Module):
#     """Primary model f(x) for 1D regression."""
#     def __init__(self):
#         super(PredictorNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(1, 32), # Increased size
#             nn.ReLU(),
#             nn.Linear(32, 32), # Increased size
#             nn.ReLU(),
#             nn.Linear(32, 32), # Added layer
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#     def forward(self, x): return self.network(x)
# 
# class ConvictionNet(nn.Module):
#     """Auxiliary model w(x) for conviction score (0-1)."""
#     def __init__(self):
#         super(ConvictionNet, self).__init__()
#         # Kept ConvictionNet size the same for now
#         self.network = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
#     def forward(self, x): return self.network(x)
# 
# 
# # ==============================================================================
#                     NETWORK CREATION FUNCTION
# ==============================================================================
def create_networks(input_dim, predictor_hidden_dims=[32, 32, 32], conviction_hidden_dims=[16, 16]):
    """
    Creates instances of the PredictorNet and ConvictionNet.

    Args:
        input_dim (int): The input dimension for both networks.
        predictor_hidden_dims (list): List of hidden layer sizes for the PredictorNet.
        conviction_hidden_dims (list): List of hidden layer sizes for the ConvictionNet.

    Returns:
        tuple: A tuple containing (predictor_net, conviction_net).
    """
    class PredictorNet(nn.Module):
        """Primary model f(x) for regression."""
        def __init__(self, input_dim, hidden_dims):
            super(PredictorNet, self).__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_dims)-1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[-1], 1)) # Output dimension is 1 for regression

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class ConvictionNet(nn.Module):
        """Auxiliary model w(x) for conviction score (0-1)."""
        def __init__(self, input_dim, hidden_dims):
            super(ConvictionNet, self).__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_dims)-1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[-1], 1)) # Output dimension is 1 for conviction
            layers.append(nn.Sigmoid()) # Ensure output is between 0 and 1

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    predictor_net = PredictorNet(input_dim, predictor_hidden_dims)
    conviction_net = ConvictionNet(input_dim, conviction_hidden_dims)
    return predictor_net, conviction_net