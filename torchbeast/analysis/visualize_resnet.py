# Adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations
# TODO:
#  1. filter comparison between two models
#  2. optimise maximally activating input

"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import argparse
import logging
import numpy as np

import torch
from torch.optim import Adam

from PIL import Image

# from misc_functions import preprocess_image, recreate_image, save_image
# from torchbeast.polybeast import Net as ResNetPoly
from torchbeast.resnet_monobeast import ResNet as ResNetMono
from torchbeast.core.popart import PopArtLayer

########################################################################################################################
# From https://github.com/utkuozbulak/pytorch-cnn-visualizations                                                       #
########################################################################################################################


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        # im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


class CNNLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = torch.zeros([1])
        self.created_image = None
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # go through the CNN layers and count up to the selected_layer CONV layer, then register the hook for that layer
        # would be nice to only pass stuff through the network up to that point but seems difficult to do
        all_convs = []
        for i in range(0, len(model.feat_convs)):
            # the order of these needs to match that in the forward() function of the ResNet
            all_convs.append(model.feat_convs[i][0])
            all_convs.append(model.resnet1[i][1])
            all_convs.append(model.resnet1[i][3])
            all_convs.append(model.resnet2[i][1])
            all_convs.append(model.resnet2[i][3])

        # Hook the selected layer
        # self.model[self.selected_layer].register_forward_hook(hook_function)
        all_convs[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        # random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # random_image = np.uint8(np.random.uniform(0, 255, (1, 84, 84)))
        random_image = np.random.uniform(0, 255, (4, 84, 84))
        # Process image and return variable
        # processed_image = preprocess_image(random_image, False)
        processed_image = torch.tensor(random_image, requires_grad=True)
        processed_image = processed_image.unsqueeze(0).unsqueeze(0).detach().requires_grad_(True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            """
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            """
            self.model(processed_image, run_to_conv=self.selected_layer)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # self.created_image = recreate_image(processed_image)
            self.created_image = processed_image
            # Save image
            if i % 5 == 0:
                im_path = os.path.expandvars(os.path.expanduser("~/logs/optim_test"))
                im_path = im_path + '/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                im = np.squeeze(self.created_image.detach().numpy(), (0, 1))
                # im[im < 0] = 0
                # im[im > 1] = 1
                im = im[0]
                im -= im.min()
                im = np.divide(im, im.max())
                im = np.round(im * 255)
                im = np.uint8(im)
                save_image(im, im_path)
        # TODO: since this mostly seems to give noise, maybe try something like this:
        # https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
        # either that or use regularisation
        # TODO: look at RIAI slides again

########################################################################################################################
#                                                                                                                      #
########################################################################################################################


all_layers = []


def remove_sequential(network):
    for layer in network.children():
        if type(layer) in [torch.nn.Sequential, torch.nn.ModuleList]:
            remove_sequential(layer)
        if not list(layer.children()):
            all_layers.append(layer)


def filter_vis(model, flags):
    """
    for n, p in model.named_parameters():
        print(n)
    print(model)
    remove_sequential(model)
    for l in all_layers:
        print(l)
    # exit(0)
    """
    # Fully connected layer is not needed
    layer_vis = CNNLayerVisualization(model, flags.layer_index, flags.filter_index)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()


def filter_comp(model, flags):
    pass


def load_model(flags):
    # load one or more models depending on the mode
    paths = flags.model_load_path.split(",")
    if len(paths) == 1 and flags.mode == "filter_comp":
        raise ValueError("Need to supply paths to two models for filter comparison.")
    if len(paths) > 1 and flags.mode == "filter_vis":
        logging.warning("More than one model specified for filter visualisation. "
                        "Only the first model will be visualised.")
        paths = paths[:1]

    models = []
    for p in paths:
        # load parameters
        checkpoint = torch.load(p, map_location="cpu")

        # determine input for building the model
        # try:
        if "baseline.mu" not in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
            checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
            num_tasks = 1
        else:
            num_tasks = checkpoint["model_state_dict"]["baseline.mu"].shape[0]
        num_actions = checkpoint["model_state_dict"]["policy.weight"].shape[0]

        model = ResNetMono(observation_shape=None,
                           num_actions=num_actions,
                           num_tasks=num_tasks,
                           use_lstm=False,
                           use_popart=True,
                           reward_clipping="abs_one")
        model.eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        """
        except:
            model = ResNetMono(num_actions=flags.num_actions,
                               num_tasks=flags.num_tasks,
                               use_lstm=True,
                               use_popart=flags.use_popart,
                               reward_clipping="abs_one")
            model.eval()
            model.load_state_dict(checkpoint["model_state_dict"])
        """
        models.append(model)

    return models if len(models) > 1 else models[0]


if __name__ == '__main__':
    logging.basicConfig(format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s", level=0)

    parser = argparse.ArgumentParser(description="Visualizations for the ResNet")
    parser.add_argument("--model_load_path", default="./logs/torchbeast",
                        help="Path to the model that should be used for the visualizations.")
    parser.add_argument("--save_dir", default="~/logs/aaa-vis",
                        help=".")
    parser.add_argument("--mode", type=str, default="filter_vis",
                        choices=["filter_vis", "filter_comp"],
                        help="What visualizations to create.")
    parser.add_argument("--layer_index", type=int, default=0,
                        help="Layer for which to visualize a filter (only in mode 'filter_vis').")
    parser.add_argument("--filter_index", type=int, default=0,
                        help="Filter to visualize (only in mode 'filter_vis').")

    # correct model params
    parser.add_argument("--frame_height", type=int, default=84,
                        help="Height to which frames are rescaled.")
    parser.add_argument("--frame_width", type=int, default=84,
                        help="Width to which frames are rescaled.")
    parser.add_argument("--num_actions", type=int, default=6,
                        help="The number of actions of the loaded model(s).")

    flags = parser.parse_args()

    model = load_model(flags)
    if flags.mode == "filter_vis":
        filter_vis(model, flags)
    elif flags.mode == "filter_comp":
        filter_comp(model, flags)

