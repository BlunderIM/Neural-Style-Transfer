import torch
import torch.nn as nn


class VariationLoss(nn.Module):
    @staticmethod
    def forward(image):
        """
        Variation loss to encourage smoothness in the image.
        Computed as the sum of the squares of teh differences in the pixel values for all pairs of pixels that are next
        to each other (horizontally or vertically)

        Args:
            image (tensor): Image tensor wit shape (batch_size=1, 3, h, w)
        Returns:
            loss (float): variation loss

        """
        loss = torch.sum((image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2) + \
            torch.sum((image[:, :, :, 1:]) - image[:, :, :, :-1]) ** 2

        return loss
