import torch
import torch.nn as nn


class StyleLoss(nn.Module):
    def __init__(self, features):
        super(StyleLoss, self).__init__()
        self.features_list = features
        gram_matrix_list = []
        for idx in range(len(features)):
            temp_gram = self.gram_matrix(self.features_list[idx])
            gram_matrix_list.append(temp_gram)
        self.gram_matrix_list = gram_matrix_list

    @staticmethod
    def gram_matrix(features, normalize=False):
        """
        Compute the gram matrix given features

        Args:
            features (tensor): Tensor with features that has shape (batch_size=1, C, H, W)
            normalize (boolean): condition whether to normalize the gram matrix or not
        Returns:
            gram (tensor): The gram matrix
        """
        n, c, h, w = features.shape

        # Converting each channel as a matrix
        features = features.view(n, c, h*w)

        # Matrix multiplication
        features_transposed = torch.transpose(features, 1, 2)
        res = torch.bmm(features, features_transposed)

        if normalize:
            res = res/(c*h*w)

        return res

    def forward(self, style_current):
        """
        Computes style loss at specified layers via the Gram matrix
        The Gram matrix is an approximation to the covariance matrix

        Args:

            style_current (list): list of current features
        Returns:
            style_loss (float): style loss
        """
        style_loss = 0
        for idx in range(len(self.gram_matrix_list)):
            temp_gram = self.gram_matrix(style_current[idx])
            delta = (temp_gram - self.gram_matrix_list[idx]) ** 2
            style_loss += torch.mean(delta)

        return style_loss



