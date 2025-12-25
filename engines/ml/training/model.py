import torch
import torch.nn as nn

class PhysicsAgent(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Output -1 to 1
        )
        
    def forward(self, x):
        return self.net(x)
        
    def get_weights(self):
        """Flatten weights for ES."""
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().numpy()
        
    def set_weights(self, weights):
        """Load weights from flat vector."""
        torch.nn.utils.vector_to_parameters(torch.tensor(weights, dtype=torch.float32), self.parameters())
