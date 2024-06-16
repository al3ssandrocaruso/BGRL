import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for a list of prediction-target pairs.

    This loss function computes the cosine similarity between predicted and target tensors.

    Methods:
    - forward(predictions_targets): Compute the cosine similarity loss.

    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, predictions_targets):
        """
        Compute the cosine similarity loss.

        Parameters:
        - predictions_targets (list of tuples): A list where each tuple contains a prediction tensor and a target tensor.

        Returns:
        - torch.Tensor: The computed cosine similarity loss.

        Raises:
        - ValueError: If the input is not a list of tuples with prediction and target tensors.
        """
        if not isinstance(predictions_targets, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in predictions_targets):
            raise ValueError("Input must be a list of tuples, where each tuple contains a prediction tensor and a target tensor.")

        total_cos_sim = 0.0
        for prediction, target in predictions_targets:
            target_detached = target.detach()
            cos_sim = F.cosine_similarity(prediction, target_detached, dim=-1).mean()
            total_cos_sim += cos_sim

        loss = len(predictions_targets) - total_cos_sim
        return loss
