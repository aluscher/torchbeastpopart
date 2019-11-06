# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

import argparse
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore")  # mute warnings, live dangerously ;)

import torch

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from skimage.transform import resize as imresize

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import torchbeast.polybeast as tb

parser = argparse.ArgumentParser(description="PyTorch Saliency for Scalable Agent")

parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--num_frames", default=50, type=int,
                    help=".")
parser.add_argument("--first_frame", default=200, type=int,
                    help=".")
parser.add_argument("--resolution", default=75, type=int,
                    help=".")
parser.add_argument("--density", default=2, type=int,
                    help=".")
parser.add_argument("--radius", default=2, type=int,
                    help=".")
parser.add_argument("--saliency_dir", default="/movies/",
                    help=".")

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

# choose an area NOT to blur
#searchlight = lambda I, mask: I * mask + torch.from_numpy(gaussian_filter(I, sigma=3)) * (1 - mask)
def searchlight(image, mask):
    ims = np.zeros([4, 84, 84])
    for i in range(4):
        ims[i] = torch.from_numpy(gaussian_filter(image[i].data.numpy()[0], sigma=3))
    imagep = torch.from_numpy(np.array([ims[0], ims[1], ims[2], ims[3]]))
    return image * mask + imagep * (1 - mask)

# choose an area to blur
#occlude = lambda I, mask: I * (1 - mask) + torch.from_numpy(gaussian_filter(I, sigma=3)) * mask
def occlude(image, mask):

    imagep = np.zeros([4, 84, 84])
    for i in range(4):
        imagep[i, :, :] = gaussian_filter(image[i], sigma=3)

    return image * (1 - mask) + imagep * mask


def rollout(model, env, max_ep_len=3e3, render=False):
    history = {"observation": [], "policy": [], "baseline": [], "core_state": [], "image": []}
    episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping

    observation = env.initial()

    with torch.no_grad():
        while not done and episode_length <= max_ep_len:
            episode_length += 1
            agent_outputs = model(observation, torch.tensor)
            policy_outputs, core_state = agent_outputs
            observation = env.step(policy_outputs[0])
            done = observation["done"]

            history["observation"].append(observation)
            history["policy"].append(policy_outputs[1].data.numpy()[0])
            history["baseline"].append(policy_outputs[2].data.numpy()[0])
            history["core_state"].append(core_state)
            history["image"].append(env.gym_env.render(mode='rgb_array'))

    return history


def get_mask(center, size, r):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    keep = x * x + y * y <= 1
    mask = np.zeros(size)
    mask[keep] = 1  # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    m = mask / mask.max()
    return np.array([m, m, m, m])


def run_through_model(model, history, ix, interp_func=None, mask=None, mode="policy", i=0, j=0):
    observation = history["observation"][ix].copy()
    frame = observation["frame"].squeeze().numpy() / 255.

    core_state = history["core_state"][ix]
    if mask is not None:
        frame = interp_func(frame, mask)  # perturb input I -> I"

    observation["frame"] = torch.from_numpy((frame * 255.).astype('uint8')).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        policy_outputs, _ = model(observation, core_state)
    policy = policy_outputs[1]
    baseline = policy_outputs[2]

    return baseline if mode == "baseline" else policy


def score_frame(model, history, ix, r, d, interp_func, mode="policy"):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)

    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    # saliency scores S(t,i,j)
    scores = np.zeros((int(84 / d) + 1, int(84 / d) + 1))
    for i in range(0, 84, d):
        for j in range(0, 84, d):
            mask = get_mask(center=[i, j], size=[84, 84], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode, i=i, j=j)
            scores[int(i / d), int(j / d)] = (L - l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    scores = imresize(scores, (84, 84)).astype(np.float32)
    scores = pmax * scores / scores.max()

    return scores


def saliency_on_atari_frame(saliency, frame, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur

    pmax = saliency.max()

    S = imresize(saliency, (160, 160))
    #S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    #S -= S.min();
    S = fudge_factor * pmax * S / S.max()

    image = frame.astype('uint16')
    image[25:185, :, channel] += S.astype('uint16')
    image = image.clip(1, 255).astype('uint8')

    return image


def get_env_meta(env_name):
    meta = {}
    if env_name == "PongNoFrameskip-v4":
        meta["baseline_ff"] = 600;
        meta["policy_ff"] = 500
    elif env_name == "SpaceInvadersNoFrameskip-v4":
        meta["baseline_ff"] = 400;
        meta["policy_ff"] = 400
    elif env_name == "AirRaidNoFrameskip-v4":
        meta["baseline_ff"] = 400;
        meta["policy_ff"] = 400
    else:
        print("environment '{}' not supported".format(env_name))
    return meta


def make_movie(model, env, flags):
    movie_title = "{}-{}-{}.mp4".format("saliency", flags.num_frames, flags.env)
    max_ep_len = flags.first_frame + flags.num_frames + 1
    torch.manual_seed(0)
    history = rollout(model, env, max_ep_len=max_ep_len)

    start = time.time()
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title=movie_title, artist="", comment="atari-saliency-video")
    writer = FFMpegWriter(fps=8, metadata=metadata)

    prog = "";
    total_frames = len(history["observation"])
    f = plt.figure(figsize=[6, 6 * 1.3], dpi=flags.resolution)
    #with writer.saving(f, flags.savedir + flags.saliency_dir + movie_title, flags.resolution):
    with writer.saving(f, "./movie.mp4", flags.resolution):
        for i in range(flags.num_frames):
            ix = flags.first_frame + i
            if ix < total_frames:  # prevent loop from trying to process a frame ix greater than rollout length

                policy_saliency = score_frame(model, history, ix, flags.radius, flags.density, interp_func=occlude, mode="policy")
                baseline_saliency = score_frame(model, history, ix, flags.radius, flags.density, interp_func=occlude, mode="baseline")

                frame = history["image"][ix]
                frame = saliency_on_atari_frame(policy_saliency, frame, fudge_factor=get_env_meta(flags.env)["policy_ff"], channel=2)
                frame = saliency_on_atari_frame(baseline_saliency, frame, fudge_factor=get_env_meta(flags.env)["baseline_ff"], channel=0)

                plt.imshow(frame);
                plt.title(flags.env, fontsize=15, fontname='DejaVuSans')
                plt.axis('off')
                writer.grab_frame();
                f.clear()

                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print("\ttime: {} | progress: {:.1f}%".format(tstr, 100 * i / min(flags.num_frames, total_frames)),
                      end="\r")
    print("\nfinished.")


if __name__ == "__main__":
    flags = parser.parse_args()

    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = tb.create_env(flags)
    env = tb.environment.Environment(gym_env)
    model = tb.Net(gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info(
        "making movie using checkpoint at %s %s", flags.savedir, flags.xpid
    )
    make_movie(model, env, flags)
