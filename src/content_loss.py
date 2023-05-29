import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    @staticmethod
    def forward(content_source, content_current):
        """
        Compute the content loss for style transfer
        Content loss measures how much the feature map of the generated image differs from the feature map of the
        source image

        Args:
            content_source (tensor): Batch containing features of images (batch_size=1, C_l, H_l, W_l)
            content_current (tensor): Batch containing features of images (batch_size=1, C_l, H_l, W_l)
        Returns:
           content_loss (float): content loss
        """
        content_loss = 0
        for idx in range(len(content_current)):
            temp_loss = torch.mean((content_source[idx] - content_current[idx])**2)
            content_loss += temp_loss

        return content_loss
