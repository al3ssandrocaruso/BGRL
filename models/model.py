import copy
import torch
import torch.nn as nn

class BGRL(nn.Module):
    """
    Bootstrapped Graph Latents (BGRL) model.

    Parameters:
    - encoder (nn.Module): The encoder network for learning representations.
    - predictor (nn.Module): The predictor network for generating predictions.

    Methods:
    - forward(x1, x2): Forward pass for the BGRL model.
    - clone_encoder(encoder): Clone the encoder network.
    - exp_moving_avg(decay_rate): Update target encoder parameters using exponential moving average.
    - get_trained_encoder(): Get a copy of the trained online encoder.
    """

    def __init__(self, encoder, predictor):
        super(BGRL, self).__init__()
        self.online_encoder = encoder
        self.predictor = predictor
        self.online_modules = nn.ModuleList([encoder, predictor])
        self.target_encoder = self.clone_encoder(encoder)

    def forward(self, x1, x2):
        """
        Forward pass for the BGRL model.

        Parameters:
        - x1 (torch.Tensor): First input tensor.
        - x2 (torch.Tensor): Second input tensor.

        Returns:
        - target_representation (torch.Tensor): Representation from the target encoder.
        - prediction (torch.Tensor): Prediction from the predictor network.
        """
        online_representation = self.online_encoder(x1)
        target_representation = self.target_encoder(x2)
        prediction = self.predictor(online_representation)
        return target_representation, prediction

    def clone_encoder(self, encoder):
        """
        Clone the encoder network.

        Parameters:
        - encoder (nn.Module): The encoder network to be cloned.

        Returns:
        - nn.Module: A cloned encoder network with requires_grad set to False.
        """
        cloned_encoder = copy.deepcopy(encoder)
        cloned_encoder.requires_grad_(False)
        return cloned_encoder

    @torch.no_grad()
    def exp_moving_avg(self, decay_rate=0.99):
        """
        Update target encoder parameters using exponential moving average.

        Parameters:
        - decay_rate (float): The decay rate for the exponential moving average (default: 0.99).
        """
        theta = list(self.online_encoder.parameters())
        phi = list(self.target_encoder.parameters())

        assert len(theta) == len(phi), "Parameter lists must be of the same length"

        for i in range(len(theta)):
            phi[i].data = decay_rate * phi[i].data + (1 - decay_rate) * theta[i].data

    @torch.no_grad()
    def get_trained_encoder(self):
        """
        Get a copy of the trained online encoder.

        Returns:
        - nn.Module: A deep copy of the online encoder network.
        """
        trained_encoder = copy.deepcopy(self.online_encoder)
        return trained_encoder
