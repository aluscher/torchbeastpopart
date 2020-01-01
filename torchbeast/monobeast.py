# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from torchbeast.attention_augmented_agent import AttentionAugmentedAgent
from torchbeast.resnet_monobeast import ResNet


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors per environment (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--agent_type", type=str, default="aaa",
                    help="The type of network to use for the agent.")
parser.add_argument("--frame_height", type=int, default=84,
                    help="Height to which frames are rescaled.")
parser.add_argument("--frame_width", type=int, default=84,
                    help="Width to which frames are rescaled.")
parser.add_argument("--aaa_input_format", type=str, default="gray_stack", choices=["gray_stack", "rgb_last", "rgb_stack"],
                    help="Color format of the frames as input for the AAA.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--save_model_every_nsteps", default=0, type=int,
                    help="Save model every n steps")

# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    env: str,
    full_action_space: bool,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        # create the environment from command line parameters
        # => could also create a special one which operates on a list of games (which we need)
        gym_env = create_env(env, frame_height=flags.frame_height, frame_width=flags.frame_width,
                             gray_scale=(flags.aaa_input_format == "gray_stack"), full_action_space=full_action_space)
        # NOTE: this part of the act() function is only called once when the actor thread/process
        # is started, so it would probably not be a good idea to just distribute the different
        # games over different environments, but that each environment contains all games

        # generate a seed for the environment (NO HUMAN STARTS HERE!), could just
        # use this for all games wrapped by the environment for our application
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)

        # wrap the environment, this is actually probably the point where we could
        # use multiple games, because the other environment is still one from Gym
        env = environment.Environment(gym_env)

        # get the initial frame, reward, done, return, step, last_action
        env_output = env.initial()

        # perform the first step
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            # get a buffer index from the queue for free buffers (?)
            index = free_queue.get()
            # termination signal (?) for breaking out of this loop
            if index is None:
                break

            # Write old rollout end.
            # the keys here are (frame, reward, done, episode_return, episode_step, last_action)
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            # here the keys are (policy_logits, baseline, action)
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            # I think the agent_state is just the RNN/LSTM state (which will be the "initial" state for the next step)
            # not sure why it's needed though because it really just seems to be the initial state before starting to
            # act; however, it might be randomly initialised, which is why we might want it...
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout (ONLY UP TO A FIXED LENGTH, IS THIS WHAT WE WANT?)
            for t in range(flags.unroll_length):
                timings.reset()

                # forward pass without keeping track of gradients to get the agent action
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                # agent acting in the environment
                # TODO: does that mean the same action isn't executed for 4 time steps?
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # writing the respective outputs of the current step (see above for the list of keys)
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")

            # after finishing a trajectory put the index in the "full queue",
            # presumably so that the data can be processed/sent to the learner
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    # need to make sure that we wait until batch_size trajectories/rollouts have been put into the queue
    with lock:
        timings.time("lock")
        # get the indices of actors "offering" trajectories/rollouts to be processed by the learner
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")

    # create the batch as a dictionary for all the data in the buffers (see act() function for list of
    # keys), where each entry is a tensor of these values stacked across actors along the first dimension,
    # which I believe should be the "batch dimension" (see _format_frame())
    batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}

    # similar thing for the initial agent states, where I think the tuples are concatenated to become torch tensors
    initial_agent_state = (torch.cat(ts, dim=1) for ts in zip(*[initial_agent_state_buffers[m] for m in indices]))
    timings.time("batch")

    # once the data has been "transferred" into batch and initial_agent_state,
    # signal that the data has been processed to the actors
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")

    # move the data to the right device (e.g. GPU)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True) for t in initial_agent_state)
    timings.time("device")

    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    stats,
    lock=threading.Lock(),
):
    """Performs a learning (optimization) step."""
    with lock:
        # forward pass with gradients
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        # if specified, clip rewards between -1 and 1
        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        # the "~"/tilde operator is apparently kind of a complement or # inverse, so maybe this just reverses
        # the "done" tensor? in that case would discounting only be applied when the game was NOT done?
        # TODO: print this out to see what it actually does
        discounts = (~batch["done"]).float() * flags.discounting

        # get the V-trace returns; I hope nothing needs to be changed about this, but I think
        # once one has the V-trace returns it can just be plugged into the PopArt equations
        # TODO: would still be good to properly understand what is happening inside this method
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # policy gradient loss
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )

        # value function/baseline loss (1/2 * squared difference between V-trace and value function)
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )

        # entropy loss for getting a "diverse" action distribution (?), "normal entropy" over action distribution
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        # get the returns only for finished episodes (where the game was played to completion)
        episode_returns = batch["episode_return"][batch["done"]]
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size

        # do the backward pass (WITH GRADIENT NORM CLIPPING) and adjust hyperparameters (scheduler, ?)
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        # update the actor model with the new parameters
        actor_model.load_state_dict(model.state_dict())

        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    # TODO: should probably print this out to get what exactly is happening here; in particular I don't really
    #  understand if the specs dictionary is even really used for anything else but to create the buffers;
    #  I'm also not sure what exactly the number of buffers influences here, so should figure that out
    T = flags.unroll_length
    specs = dict(  # seems like these "inner" dicts could also be something else...
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}

    # basically create a bunch of empty torch tensors according to the sizes in the specs dicts above
    # and do this for the specified number of buffers, so that there will be a list of length flags.num_buffers
    # for each key
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )
    if flags.save_model_every_nsteps > 0:
        os.makedirs(checkpointpath.replace("model.tar", "intermediate"), exist_ok=True)

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # set the right agent class
    if flags.agent_type.lower() in ["aaa", "attention_augmented", "attention_augmented_agent"]:
        Net = AttentionAugmentedAgent
        logging.info("Using the Attention-Augmented Agent architecture.")
    elif flags.agent_type.lower() in ["rn", "res", "resnet", "res_net"]:
        Net = ResNet
        logging.info("Using the ResNet architecture (monobeast version).")
    else:
        Net = AtariNet
        logging.warning("No valid agent type specified. Using the default agent.")

    environments = flags.env.split(",")

    # create a dummy environment, mostly to get the observation and action spaces from
    gym_env = create_env(environments[0], frame_height=flags.frame_height, frame_width=flags.frame_width,
                     gray_scale=(flags.aaa_input_format == "gray_stack"))
    observation_space_shape = gym_env.observation_space.shape
    action_space_n = gym_env.action_space.n
    full_action_space = False
    for environment in environments[1:]:
        gym_env = create_env(environment)
        if gym_env.action_space.n != action_space_n:
            logging.warning("Action spaces don't match, using full action space.")
            full_action_space = True
            action_space_n = 18
            break

    # create the model and the buffers to pass around data between actors and learner
    model = Net(observation_space_shape, action_space_n, use_lstm=flags.use_lstm, rgb_last=(flags.aaa_input_format == "rgb_last"))
    buffers = create_buffers(flags, observation_space_shape, model.num_actions)

    # I'm guessing that this is required (similarly to the buffers) so that the
    # different threads/processes can all have access to the parameters etc. (?)
    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # create stuff to keep track of the actor processes
    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i, environment in enumerate(environments):
        for j in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(
                    flags,
                    environment,
                    full_action_space,
                    i*flags.num_actors + j,
                    free_queue,
                    full_queue,
                    model,
                    buffers,
                    initial_agent_state_buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

    learner_model = Net(observation_space_shape, action_space_n, use_lstm=flags.use_lstm,
                        rgb_last=(flags.aaa_input_format == "rgb_last")).to(device=flags.device)

    # the hyperparameters in the paper are found/adjusted using population-based training,
    # which might be a bit too difficult for us to do; while the IMPALA paper also does
    # some experiments with this, it doesn't seem to be implemented here
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        # step in particular needs to be from the outside scope, since all learner threads can update
        # it and all learners should stop once the total number of steps/frames has been processed
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            s = learn(flags, model, learner_model, batch, agent_state, optimizer, scheduler, stats)
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B  # so this counts the number of frames, not e.g. trajectories/rollouts

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    # populate the free queue with the indices of all the buffers at the start
    for m in range(flags.num_buffers):
        free_queue.put(m)

    # start as many learner threads as specified => could in principle do PBT (somehow?)
    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,))
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    def save_model():
        save_model_path = checkpointpath.replace(
            "model.tar", "intermediate/model." + str(stats.get("step", 0)).zfill(9) + ".tar")
        logging.info("Saving model to %s", save_model_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            save_model_path,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        last_savemodel_nsteps = 0
        while step < flags.total_steps:
            start_step = stats.get("step", 0)
            start_time = timer()
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min. TODO: probably change this to count steps
                checkpoint()
                last_checkpoint_time = timer()

            if flags.save_model_every_nsteps > 0 and end_step >= last_savemodel_nsteps + flags.save_model_every_nsteps:
                # save model every save_model_every_nsteps steps
                save_model()
                last_savemodel_nsteps = end_step

            sps = (end_step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = ("Return per episode: %.1f. " % stats["mean_episode_return"])
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                end_step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)  # send quit signal to actors
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")))

    # set the right agent class
    if flags.agent_type.lower() in ["aaa", "attention_augmented", "attention_augmented_agent"]:
        Net = AttentionAugmentedAgent
        logging.info("Using the Attention-Augmented Agent architecture.")
    elif flags.agent_type.lower() in ["rn", "res", "resnet", "res_net"]:
        Net = ResNet
        logging.info("Using the ResNet architecture (monobeast version).")
    else:
        Net = AtariNet
        logging.warning("No valid agent type specified. Using the default agent.")

    if len(flags.env.split(",")) != 1:
        raise Exception("Only one environment allowed for testing")

    gym_env = create_env(flags.env, frame_height=flags.frame_height, frame_width=flags.frame_width,
                         gray_scale=(flags.aaa_input_format == "gray_stack"))
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, use_lstm=flags.use_lstm,
                rgb_last=(flags.aaa_input_format == "rgb_last"))
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info("Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns))


class AtariNet(nn.Module):

    def __init__(self, observation_shape, num_actions, use_lstm=False, **kwargs):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            # notdone has shape (time_steps, batch_size)
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
            # pretty sure flatten() is just used to merge time and batch again
        else:
            core_output = core_input
            core_state = tuple()

        # core_output should have shape (T * B, hidden_size) now?
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


# Net = AtariNet
Net = AttentionAugmentedAgent


def create_env(env, frame_height=84, frame_width=84, gray_scale=True, full_action_space=False):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(env, full_action_space=full_action_space),
            clip_rewards=False,
            frame_stack=True,
            frame_height=frame_height,
            frame_width=frame_width,
            gray_scale=gray_scale,
            scale=False,
        )
    )


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
