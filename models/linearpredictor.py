import torch.nn as nn

class LinearPredictor(nn.Module):
    """
    A simple linear predictor model with one hidden layer.

    Parameters:
    - input_dim (int): Input feature dimension.
    - hidden_dim (int): Hidden layer dimension.
    - output_dim (int): Output layer dimension.
    - activation_fn (nn.Module): Activation function after the hidden layer (default: nn.PReLU).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn=nn.PReLU):
        super(LinearPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.layers(x)
