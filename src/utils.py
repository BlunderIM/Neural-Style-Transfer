import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch


def preprocess_image(image):
    """
    Adjust image size and convert to tensor

    Args:
        image (tensor): input image
    Returns:
        transformation (tensor): transformed image
    """
    transformation = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]
        )
    return transformation(image)


def extract_features(image, network, target_layers=None):
    """
    Extract features of an image using the provided convolutional neural network

    Args:
        image (tensor): Input image
        network (torchvision.models): Neural network
        target_layers (list): List of indices of layers
    Returns:
        features (list): list of features
    """
    if target_layers is None:
        target_layers = range(30)  # If no target layers are given, return the first 30 layers
    features = []
    previous_feature = image
    # Iterating over all layers upto 30. Not including features after the 30's layer
    for i, module in enumerate(network.features[:30]):
        output_feature = module(previous_feature)
        if i in target_layers:
            features.append(output_feature)
        previous_feature = output_feature

    return features


def take_snapshot(content_image, style_image, output_image, iteration_number):
    """
    Create a plot showing content_image, style_image, and the style_transfer_image

    Args:
        content_image (tensor): Content_image
        style_image (tensor): Style image
        output_image (tensor): Output image
        iteration_number (int): Iteration number used in naming the image
    Returns:
          None
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title('Content Image')
    ax[1].set_title('Generated Image')
    ax[2].set_title('Style Image')
    ax[0].imshow(content_image.squeeze().permute(1, 2, 0).cpu())
    ax[1].imshow(torch.clamp(output_image, 0, 1).squeeze().detach().permute(1, 2, 0).cpu())
    ax[2].imshow(style_image.squeeze().permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(f"../art/{iteration_number}_style_transfer.png", bbox_inches='tight');
    plt.close()






