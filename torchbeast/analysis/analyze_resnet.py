import os
import re
import csv
import argparse
import logging
import pickle
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import torch
from torch.optim import Adam

from PIL import Image

# from misc_functions import preprocess_image, recreate_image, save_image
# from torchbeast.polybeast import Net as ResNetPoly
from torchbeast.resnet_monobeast import ResNet as ResNetMono
from torchbeast.core.popart import PopArtLayer

logging.getLogger('matplotlib.font_manager').disabled = True

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
        for i in range(0, len(self.model.feat_convs)):
            # the order of these needs to match that in the forward() function of the ResNet
            all_convs.append(self.model.feat_convs[i][0])
            all_convs.append(self.model.resnet1[i][1])
            all_convs.append(self.model.resnet1[i][3])
            all_convs.append(self.model.resnet2[i][1])
            all_convs.append(self.model.resnet2[i][3])

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


def filter_vis(flags):
    paths = flags.model_load_path.split(",")
    if len(paths) > 1:
        logging.warning("More than one model specified for filter visualisation. "
                        "Only the first model will be visualised.")
        paths = paths[:1]
    model = load_models(paths)

    # Fully connected layer is not needed
    layer_vis = CNNLayerVisualization(model, flags.layer_index, flags.filter_index)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()


def single_filter_comp(model_a, model_b, compute_optimal=True):
    models = [model_a, model_b]
    filter_list = [[] for _ in models]
    for m_idx, m in enumerate(models):
        for i in range(0, len(m.feat_convs)):
            filter_list[m_idx].append(m.feat_convs[i][0].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][3].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][3].weight.detach().numpy())
    filter_list_a = filter_list[0]
    filter_list_b = filter_list[1]

    distance_data = []
    default_dist_data = []
    optimal_dist_data = []
    for f_idx, f in enumerate(tqdm(filter_list_b)):
        # reshape filters
        all_filters = f.shape[0] * f.shape[1]
        filter_size = f.shape[2] * f.shape[3]
        original = np.reshape(filter_list_a[f_idx], (all_filters, filter_size))
        comparison = np.reshape(f, (all_filters, filter_size))

        # compute distances
        distances = cdist(original, comparison, metric="sqeuclidean")
        default_dist_sum = distances[0, :].sum()
        default_dist_mean = distances[0, :].mean()

        # compute optimal assignment
        if compute_optimal:
            row_idx, col_idx = linear_sum_assignment(distances)
            optimal_dist_sum = distances[row_idx, col_idx].sum()
            optimal_dist_mean = distances[row_idx, col_idx].mean()
        else:
            optimal_dist_sum = 0
            optimal_dist_mean = 0

        # print("Default distance sum/mean: {}/{}".format(default_dist_sum, default_dist_mean))
        # print("Optimal distance sum/mean: {}/{}".format(optimal_dist_sum, optimal_dist_mean))

        distance_data.append(distances)
        default_dist_data.append((f.shape, default_dist_sum, default_dist_mean))
        optimal_dist_data.append((f.shape, optimal_dist_sum, optimal_dist_mean))

    return distance_data, default_dist_data, optimal_dist_data


def parallel_filter_calc(combined_input):
    # TODO: change this data storage stuff so that both the mean and sum are stored
    #  (maybe make a nested dict for mean/sum or default/optimal)
    models, model_name, comparison_name = combined_input
    return single_filter_comp(models[comparison_name], models[model_name], not flags.comp_dist_only)


def filter_comp(flags):
    single_task_names = ["Carnival", "AirRaid", "DemonAttack", "NameThisGame", "Pong", "SpaceInvaders"]
    multi_task_name = "MultiTask"
    multi_task_popart_name = "MultiTaskPopart"
    model_names = single_task_names + [multi_task_name] + [multi_task_popart_name]

    # 1. get the model base directory
    # 2. get the paths for all subdirectories (only those matching a list of "model names")
    # 3. get lists of all model names: either x models every y checkpoints for all models
    #    or the same but only for the first 50 models (and the last)
    # 4. load models for every time step and do the filter comparison between
    #    a) all single-task and the vanilla multi-task model
    #    b) all single-task and the popart multi-task model
    #    c) the vanilla and popart multi-task models
    # 5. write values to separate CSV files and store them as well
    # 6. display stuff

    base_path = flags.model_load_path.split(",")
    if len(base_path) > 1:
        logging.warning("More than one path specified for filter progress visualization. "
                        "In this mode only the base directory is required. Using only the first path.")
    base_path = base_path[0]

    # get base directories for all intermediate models
    intermediate_paths = []
    for p in os.listdir(base_path):
        if p in model_names:
            intermediate_paths.append((p, os.path.join(base_path, p, "intermediate")))

    # get all model checkpoints that should be loaded
    logging.info("Determining checkpoints to load.")
    selected_model_paths = []
    for model_name, path in intermediate_paths:
        checkpoints = []
        for checkpoint in os.listdir(path):
            if not checkpoint.endswith(".tar"):
                continue
            checkpoint_n = int(re.search(r'\d+', checkpoint).group())
            checkpoints.append((checkpoint_n, checkpoint))
        checkpoints.sort()

        if flags.match_num_models and "MultiTask" in model_name:
            index = list(np.round(np.linspace(0, 50 - 1, flags.comp_num_models)).astype(int))  # TODO: make dynamic
            index.append(len(checkpoints) - 1)
        else:
            index = np.round(np.linspace(0, len(checkpoints) - 1, flags.comp_num_models)).astype(int)
        selected_models = [os.path.join(base_path, model_name, "intermediate", checkpoints[i][1]) for i in index]
        selected_model_paths.append((model_name, selected_models))

    # go through each time step
    logging.info("Starting the computation.")
    single_multi_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in single_task_names
    }
    for n in single_task_names:
        single_multi_data[n]["dist"] = []
    single_multipop_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in single_task_names
    }
    for n in single_task_names:
        single_multipop_data[n]["dist"] = []
    multi_multipop_data = {
        s: {
            t: [] for t in ["sum", "mean"]
        } for s in ["default", "optimal"]
    }
    multi_multipop_data["dist"] = []
    for t in range(flags.comp_num_models + (1 if flags.match_num_models else 0)):
        # load checkpoints for each model
        logging.info("Loading checkpoints for all models ({}/{}).".format(t + 1, flags.comp_num_models))
        models = {}
        for model_name, model_paths in selected_model_paths:
            models[model_name] = load_models(
                [model_paths[t - (1 if t == flags.comp_num_models and "multi" not in model_name.lower() else 0)]])

        # compare single and vanilla multi-task models
        logging.info("Comparing single-task and vanilla multi-task models ({}/{})."
                     .format(t + 1, flags.comp_num_models))
        with mp.Pool(len(single_task_names)) as pool:
            full_data = pool.map(parallel_filter_calc, [(models, mn, multi_task_name) for mn in single_task_names])
        for model_name, data in zip(single_task_names, full_data):
            dist, dd_data, od_data = data
            single_multi_data[model_name]["dist"].append(dist)
            single_multi_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
            single_multi_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
            single_multi_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
            single_multi_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        # compare single and multi-task PopArt models
        logging.info("Comparing single-task and multi-task PopArt models ({}/{})."
                     .format(t + 1, flags.comp_num_models))
        with mp.Pool(len(single_task_names)) as pool:
            full_data = pool.map(parallel_filter_calc, [(models, mn, multi_task_popart_name) for mn in single_task_names])
        for model_name, data in zip(single_task_names, full_data):
            dist, dd_data, od_data = data
            single_multipop_data[model_name]["dist"].append(dist)
            single_multipop_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
            single_multipop_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
            single_multipop_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
            single_multipop_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        # compare multi-task and multi-task PopArt models
        logging.info("Comparing vanilla multi-task and multi-task PopArt models ({}/{})."
                     .format(t + 1, flags.comp_num_models))
        dist, dd_data, od_data = single_filter_comp(models[multi_task_popart_name], models[multi_task_name],
                                                    not flags.comp_dist_only)
        print(dist[0].shape, len(dist))
        multi_multipop_data["dist"].append(dist)
        multi_multipop_data["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
        multi_multipop_data["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
        multi_multipop_data["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
        multi_multipop_data["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

    def update_dict(d):
        if type(d) == list:
            if type(d[0]) == list:
                result = [np.stack([d[time][filter] for time in range(len(d))]) for filter in range(len(d[0]))]
            else:
                result = np.stack(d)
            return result
        else:
            for k in d:
                d[k] = update_dict(d[k])
            return d

    # write data to files
    for dictionary, file_name in [(single_multi_data, "single_multi"),
                                  (single_multipop_data, "single_multipop"),
                                  (multi_multipop_data, "multi_multipop")]:
        dictionary = update_dict(dictionary)
        full_path = os.path.join(
            os.path.expanduser(flags.save_dir),
            "filter_comp", "{}_{}{}".format(flags.comp_num_models, "match" if flags.match_num_models else "no_match",
                                            "_dist_only" if flags.comp_dist_only else ""),
            file_name + ".pkl"
        )
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        with open(full_path, "wb") as f:
            pickle.dump(dictionary, f)
        logging.info("Wrote data to file '{}'.".format(full_path))


def _filter_comp(flags):
    paths = flags.model_load_path.split(",")
    if len(paths) == 1:
        raise ValueError("Need to supply paths to two models for filter comparison.")
    models = load_models(paths)

    filter_list = [[] for _ in models]
    for m_idx, m in enumerate(models):
        for i in range(0, len(m.feat_convs)):
            filter_list[m_idx].append(m.feat_convs[i][0].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][3].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][3].weight.detach().numpy())

    diffs = [[] for _ in models]
    for filters in filter_list[1:]:
        for f_idx, f in enumerate(filters):
            diffs = np.zeros(f.shape[:2])
            for oc_idx, out_channel in enumerate(f):
                for ic_idx, in_channel in enumerate(out_channel):
                    s = ((in_channel - filter_list[0][f_idx][oc_idx][ic_idx]) ** 2).sum()
                    diffs[oc_idx, ic_idx] = s
            plt.imshow(diffs, cmap="hot", interpolation="nearest")
            plt.show()

            all_filters = f.shape[0] * f.shape[1]
            filter_size = f.shape[2] * f.shape[3]
            original = np.reshape(filter_list[0][f_idx], (all_filters, filter_size))
            comparison = np.reshape(f, (all_filters, filter_size))
            distances = cdist(original, comparison, metric="sqeuclidean")
            # print(distances.shape)
            # TODO: search for matches only for corresponding in/out filters
            #  => does this even make sense? if the input is going to be different anyway, not beyond the first layer...

            # TODO: track changes between models over time => probably need to run on server, will take time
            row_idx, col_idx = linear_sum_assignment(distances)
            distance_sum = distances[row_idx, col_idx].sum()
            print("Total distance sum:", distance_sum)
            print("Without matching:", distances[0, :].sum())
            """
            ax = sns.heatmap(distances)
            for r_idx, c_idx in zip(row_idx, col_idx):
                ax.add_patch(Rectangle((r_idx, c_idx), 1, 1, fill=False, edgecolor="blue", lw=1))
            """
            # sns.clustermap(distances, center=0)
            plt.show()


def load_models(paths):
    models = []
    for p in paths:
        # load parameters
        checkpoint = torch.load(p, map_location="cpu")

        # determine input for building the model
        if "baseline.mu" not in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
            checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
            num_tasks = 1
        else:
            num_tasks = checkpoint["model_state_dict"]["baseline.mu"].shape[0]
        num_actions = checkpoint["model_state_dict"]["policy.weight"].shape[0]

        # construct model and transfer loaded parameters
        model = ResNetMono(observation_shape=None,
                           num_actions=num_actions,
                           num_tasks=num_tasks,
                           use_lstm=False,
                           use_popart=True,
                           reward_clipping="abs_one")
        model.eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model)

    return models if len(models) > 1 else models[0]


if __name__ == '__main__':
    logging.basicConfig(format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s", level=0)

    parser = argparse.ArgumentParser(description="Visualizations for the ResNet")
    parser.add_argument("--model_load_path", default="./logs/torchbeast",
                        help="Path to the model that should be used for the visualizations.")
    parser.add_argument("--save_dir", default="~/logs/resnet_vis",
                        help=".")
    parser.add_argument("--mode", type=str, default="filter_vis",
                        choices=["filter_vis", "filter_comp", "_filter_comp"],
                        help="What visualizations to create.")
    parser.add_argument("--layer_index", type=int, default=0,
                        help="Layer for which to visualize a filter.")
    parser.add_argument("--filter_index", type=int, default=0,
                        help="Filter to visualize (only in mode 'filter_vis').")
    parser.add_argument("--pairwise_comp", action="store_true",
                        help="Visualise difference between all pairwise filters, "
                             "not just corresponding ones (only in mode 'filter_comp').")

    parser.add_argument("--match_num_models", action="store_true",
                        help="...")
    parser.add_argument("--comp_num_models", type=int, default=10,
                        help="How many models...")
    parser.add_argument("--comp_dist_only", action="store_true",
                        help="How many models...")

    # correct model params
    parser.add_argument("--frame_height", type=int, default=84,
                        help="Height to which frames are rescaled.")
    parser.add_argument("--frame_width", type=int, default=84,
                        help="Width to which frames are rescaled.")
    parser.add_argument("--num_actions", type=int, default=6,
                        help="The number of actions of the loaded model(s).")

    flags = parser.parse_args()

    if flags.mode == "filter_vis":
        filter_vis(flags)
    elif flags.mode == "_filter_comp":
        _filter_comp(flags)
    elif flags.mode == "filter_comp":
        filter_comp(flags)

