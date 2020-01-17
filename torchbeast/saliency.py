# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License
# https://github.com/greydanus/visualize_atari

import argparse
import logging
import os
import re

from PIL import Image

import torch

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from skimage.transform import resize as imresize

import torchbeast.polybeast as tb

parser = argparse.ArgumentParser(description="PyTorch Saliency for Scalable Agent")

parser.add_argument("--savedir", default="./logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--intermediate_model_id", default=None,
                    help="id for intermediate model: model.id.tar")
parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--num_frames", default=10000, type=int,
                    help=".")
parser.add_argument("--first_frame", default=0, type=int,
                    help=".")
parser.add_argument("--resolution", default=75, type=int,
                    help=".")
parser.add_argument("--density", default=2, type=int,
                    help=".")
parser.add_argument("--radius", default=2, type=int,
                    help=".")
parser.add_argument("--saliencydir", default="./movies/saliency_raw",
                    help=".")
parser.add_argument("--actions",
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


def rollout(model, env, max_ep_len=3e3, actions=None):
    history = {"observation": [], "policy": [], "baseline": [], "normalized_baseline": [], "core_state": [], "image": []}
    episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping

    observation = env.initial()

    with torch.no_grad():
        while not done and episode_length <= max_ep_len:
            agent_outputs = model(observation, torch.tensor)
            policy_outputs, core_state = agent_outputs
            action = policy_outputs[0] if len(actions) == 0 else torch.tensor(actions[episode_length])
            observation = env.step_no_task(action)
            done = observation["done"]

            history["observation"].append(observation)
            history["policy"].append(policy_outputs[1].data.numpy()[0])
            history["baseline"].append(policy_outputs[2].data.numpy()[0])
            history["normalized_baseline"].append(policy_outputs[3].data.numpy()[0])
            history["core_state"].append(core_state)
            history["image"].append(env.gym_env.render(mode='rgb_array'))

            episode_length += 1

    return history


def get_mask(center, size, r):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    keep = x * x + y * y <= 1
    mask = np.zeros(size)
    mask[keep] = 1  # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    m = mask / mask.max()
    return np.array([m, m, m, m])


def run_through_model(model, history, ix, interp_func=None, mask=None, mode="policy", task=0):
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
    return policy if mode == "policy" else baseline[:, :, task]


def score_frame(model, history, ix, r, d, interp_func, mode="policy", task=0):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)

    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode, task=task)

    # saliency scores S(t,i,j)
    scores = np.zeros((int(84 / d) + 1, int(84 / d) + 1))
    for i in range(0, 84, d):
        for j in range(0, 84, d):
            mask = get_mask(center=[i, j], size=[84, 84], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode, task=task)
            scores[int(i / d), int(j / d)] = (L - l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    scores = imresize(scores, (84, 84)).astype(np.float32)
    scores = pmax * scores / scores.max()

    return scores


def saliency_on_atari_frame(saliency, channel=0):
    q = np.quantile(saliency.flatten(), [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9])
    S = np.zeros_like(saliency)
    delta = 255. / 10.
    for i in range(1, len(q)):
        idx = saliency >= q[i]
        if i > 1:
            idx = np.logical_and(saliency >= q[i], saliency < q[i - 1])
        S[idx] = (saliency[idx] - q[i]) / (q[i - 1] - q[i]) * delta + (10 - i) * delta

    S = imresize(S, (160, 160))

    image = np.zeros([210, 160, 3], dtype='uint16')
    image[25:185, :, channel] += S.astype('uint16')
    image = image.clip(0, 255).astype('uint8')

    return image


def make_movie(model, env, flags):

    actions = []
    if flags.actions is not None:
        f = open(flags.actions, "r")
        for line in f:
            actions.append(int(line.replace("tensor([[[", "").replace("]]])", "")))
        f.close()

    max_ep_len = flags.first_frame + flags.num_frames + 1

    torch.manual_seed(0)
    history = rollout(model, env, max_ep_len=max_ep_len, actions=actions)

    total_frames = len(history["observation"])

    saliencypath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, flags.saliencydir))
    )
    if not os.path.exists(saliencypath):
        os.makedirs(saliencypath)

    for i in range(flags.num_frames):
        ix = flags.first_frame + i

        if ix < total_frames:
            policy_saliency = score_frame(model, history, ix, flags.radius, flags.density, interp_func=occlude, mode="policy", task=flags.task)
            baseline_saliency = score_frame(model, history, ix, flags.radius, flags.density, interp_func=occlude, mode="baseline", task=flags.task)

            frame_policy_saliency = saliency_on_atari_frame(policy_saliency, channel=0)
            frame_baseline_saliency = saliency_on_atari_frame(baseline_saliency, channel=2)

            frame_atari = history["image"][ix]
            filename = saliencypath + "/" + "{}_{}_{}_{}_{}".format("Atari", flags.xpid, flags.intermediate_model_id, flags.env, str(ix).zfill(5)) + ".png"
            im = Image.fromarray(frame_atari).resize([160, 210])
            im.save(filename)

            frame_saliency = frame_policy_saliency + frame_baseline_saliency
            filename = saliencypath + "/" + "{}_{}_{}_{}_{}".format("Saliency", flags.xpid, flags.intermediate_model_id, flags.env, str(ix).zfill(5)) + ".png"
            im = Image.fromarray(frame_saliency)
            im.save(filename)

            print("\tprogress: {:.1f}%".format(100 * i / min(flags.num_frames, total_frames)), end="\r")
    print("\nfinished.")


def create_env_det(env_name, full_action_space=False, noop=20):
    return tb.atari_wrappers.wrap_pytorch(
        tb.atari_wrappers.wrap_deepmind(
            tb.atari_wrappers.make_atari_det(env_name, full_action_space=full_action_space, noop=noop),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


task_map = {
    "AirRaidNoFrameskip-v4": 0
    , "CarnivalNoFrameskip-v4": 1
    , "DemonAttackNoFrameskip-v4": 2
    , "NameThisGameNoFrameskip-v4": 3
    , "PongNoFrameskip-v4": 4
    , "SpaceInvadersNoFrameskip-v4": 5
}


if __name__ == "__main__":
    flags = parser.parse_args()

    if flags.xpid is None:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, "latest", "model.tar"))
        )
        meta = checkpointpath.replace("model.tar", "meta.json")
    else:
        if flags.intermediate_model_id is None:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
            )
            meta = checkpointpath.replace("model.tar", "meta.json")
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s/%s" % (flags.savedir, flags.xpid, "intermediate", "model." + flags.intermediate_model_id + ".tar"))
            )
            meta = re.sub(r"model.*tar", "meta.json", checkpointpath).replace("/intermediate", "")
    flags_orig = tb.read_metadata(meta)
    args_orig = flags_orig["args"]
    num_actions = args_orig.get("num_actions")
    num_tasks = args_orig.get("num_tasks", 1)
    use_lstm = args_orig.get("use_lstm", False)
    use_popart = args_orig.get("use_popart", False)
    reward_clipping = args_orig.get("reward_clipping", "abs_one")

    task = 0
    if num_tasks > 1:
        task = task_map[flags.env]
    flags.task = task

    gym_env = create_env_det(flags.env)
    gym_env.seed(0)
    env = tb.Environment(gym_env)
    model = tb.Net(num_actions=num_actions, num_tasks=num_tasks, use_lstm=use_lstm, use_popart=use_popart, reward_clipping=reward_clipping)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    if 'baseline.mu' not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
        checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info(
        "making movie using checkpoint at %s %s", flags.savedir, flags.xpid
    )
    flags.use_popart = use_popart
    make_movie(model, env, flags)
