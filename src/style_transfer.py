import numpy as np
import torch
import torchvision.models
import torchvision.models as models
import PIL
from content_loss import ContentLoss
from style_loss import StyleLoss
from variation_loss import VariationLoss
from utils import preprocess_image, extract_features, take_snapshot
import PIL
from PIL import Image


def style_transfer(content_image_path, style_image_path, cnn, content_layers_of_interest,
                   style_layers_of_interest, initialize_random, device, plot_intermediate_progress=False):
    """
    Perform style transfer

    Args:
        content_image_path (string): Location of content image
        style_image_path (string): Location of style image
        cnn (torchvision.models): Neural network
        content_layers_of_interest (list): Layers to use for content loss
        style_layers_of_interest (list): Layers to use for content loss
        initialize_random (bool): Conditional to randomly initialize the output image
        device (string): Defines whether to use GPU or CPU
        plot_intermediate_progress (bool): Conditional determine if intermediate snapshots should be saved
    Returns:
        output_image (tensor): Output image after style transfer

    """
    # Read and pre-process the images
    content_image = Image.open(content_image_path)
    content_image = preprocess_image(content_image).unsqueeze(0).to(device)
    style_image= Image.open(style_image_path)
    style_image = preprocess_image(style_image).unsqueeze(0).to(device)

    # Extracting features from the source content image
    source_content_target_features = extract_features(content_image, cnn, content_layers_of_interest)

    # Extracting gram matrices from the source style image
    source_style_target_features = extract_features(style_image, cnn, style_layers_of_interest)

    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss(source_style_target_features)
    variation_loss_fn = VariationLoss()

    # Initialize the output image
    if initialize_random:
        output_image = torch.empty(content_image.size(), device=device).uniform_()
    else:
        output_image = content_image.clone()
    output_image.requires_grad = True

    # Setting the hyperparameters
    initial_lr = 0.04
    decayed_lr = 0.001
    decay_lr_at = 180

    # Setting the optimizer
    optimizer = torch.optim.Adam([output_image], lr=initial_lr)

    # Implementing the update
    for it in range(1, 3001):

        output_content_target_features = extract_features(output_image, cnn, content_layers_of_interest)
        output_style_target_features = extract_features(output_image, cnn, style_layers_of_interest)
        optimizer.zero_grad()
        content_loss = content_loss_fn(source_content_target_features, output_content_target_features)
        style_loss = style_loss_fn(output_style_target_features)
        variation_loss = variation_loss_fn(output_image)
        total_loss = 1000*content_loss + 0.0001*style_loss + variation_loss*0.001
        print(f"Iteration # {it} loss: {total_loss}")
        total_loss.backward()

        if it == decay_lr_at:
            optimizer = torch.optim.Adam([output_image], lr=decayed_lr)

        if plot_intermediate_progress or it == 3000:
            take_snapshot(content_image, style_image, output_image, it)

        optimizer.step()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.cuda.FloatTensor
    else:
        device = torch.device("cpu")
        dtype = torch.FloatTensor

    network = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    for param in network.parameters():
        param.requires_grad = False

    network.type(dtype)

    params = {
        "content_image": "../data/dwight.jpg",
        "style_image": "../data/scream.jpg",
        "content_layers": [0, 5, 10, 19, 28],
        "style_layers": [0, 5, 10, 19, 28],
    }

    style_transfer(params["content_image"], params["style_image"], network, params["content_layers"],
                   params["style_layers"],
                   initialize_random=True, device=device, plot_intermediate_progress=False)





