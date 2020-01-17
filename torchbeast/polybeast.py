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
import collections
import logging
import os
import signal
import subprocess
import threading
import time
import timeit
import traceback

import re

from PIL import Image


os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import nest
import numpy as np

import torch
from libtorchbeast import actorpool
from torch import nn
from torch.nn import functional as F

from torchbeast.core import file_writer
from torchbeast.core import vtrace
from torchbeast.core.environment import Environment
from torchbeast.core.file_writer import read_metadata
from torchbeast.core.popart import PopArtLayer

from torchbeast import atari_wrappers

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render", "record", "env_info"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--start_servers", dest="start_servers", action="store_true",
                    help="Spawn polybeast_env servers automatically.")
parser.add_argument("--no_start_servers", dest="start_servers", action="store_false",
                    help="Don't spawn polybeast_env servers automatically.")
parser.set_defaults(start_servers=True)
parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment. Ignored if --no_start_servers is passed.")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="./logs/torchbeast-local",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=16, type=int, metavar="N",
                    help="Number of actors.")
parser.add_argument("--total_steps", default=10000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--num_inference_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--num_actions", default=6, type=int, metavar="A",
                    help="Number of actions.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--max_learner_queue_size", default=None, type=int, metavar="N",
                    help="Optional maximum learner queue size. Defaults to batch_size.")
parser.add_argument("--use_popart", action="store_true",
                    help="Use PopArt Layer.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01, type=float, # 0.0006
                    help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99, type=float,
                    help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048, type=float,   # 0.0006
                    metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--beta", default=0.0001, type=float,
                    help="PopArt parameter")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--save_model_every_nsteps", default=0, type=int,
                    help="Save model every n steps")

# Test settings.
parser.add_argument("--num_episodes", default=100, type=int,
                    help="Number of episodes for Testing.")
parser.add_argument("--intermediate_model_id", default=None,
                    help="id for intermediate model: model.id.tar")
parser.add_argument("--actions",
                    help="Use given action sequence.")

# yapf: enable


atari_environments = np.array(["AdventureNoFrameskip-v4"
                                  , "AirRaidNoFrameskip-v4"
                                  , "AlienNoFrameskip-v4"
                                  , "AmidarNoFrameskip-v4"
                                  , "AssaultNoFrameskip-v4"
                                  , "AsterixNoFrameskip-v4"
                                  , "AsteroidsNoFrameskip-v4"
                                  , "AtlantisNoFrameskip-v4"
                                  , "BankHeistNoFrameskip-v4"
                                  , "BattleZoneNoFrameskip-v4"
                                  , "BeamRiderNoFrameskip-v4"
                                  , "BerzerkNoFrameskip-v4"
                                  , "BowlingNoFrameskip-v4"
                                  , "BoxingNoFrameskip-v4"
                                  , "BreakoutNoFrameskip-v4"
                                  , "CarnivalNoFrameskip-v4"
                                  , "CentipedeNoFrameskip-v4"
                                  , "ChopperCommandNoFrameskip-v4"
                                  , "CrazyClimberNoFrameskip-v4"
                                  # , "DefenderNoFrameskip-v4"  # doesn't work
                                  , "DemonAttackNoFrameskip-v4"
                                  , "DoubleDunkNoFrameskip-v4"
                                  , "ElevatorActionNoFrameskip-v4"
                                  , "EnduroNoFrameskip-v4" # not in torchbeast paper
                                  , "FishingDerbyNoFrameskip-v4"
                                  , "FreewayNoFrameskip-v4" # not in torchbeast paper
                                  , "FrostbiteNoFrameskip-v4"
                                  , "GopherNoFrameskip-v4"
                                  , "GravitarNoFrameskip-v4"
                                  , "HeroNoFrameskip-v4"
                                  , "IceHockeyNoFrameskip-v4"
                                  , "JamesbondNoFrameskip-v4"
                                  , "JourneyEscapeNoFrameskip-v4"
                                  , "KangarooNoFrameskip-v4"
                                  , "KrullNoFrameskip-v4"
                                  , "KungFuMasterNoFrameskip-v4"
                                  , "MontezumaRevengeNoFrameskip-v4"
                                  , "MsPacmanNoFrameskip-v4"
                                  , "NameThisGameNoFrameskip-v4"
                                  , "PhoenixNoFrameskip-v4"
                                  , "PitfallNoFrameskip-v4"
                                  , "PongNoFrameskip-v4"
                                  , "PooyanNoFrameskip-v4"
                                  , "PrivateEyeNoFrameskip-v4"
                                  , "QbertNoFrameskip-v4"
                                  , "RiverraidNoFrameskip-v4"
                                  , "RoadRunnerNoFrameskip-v4"
                                  , "RobotankNoFrameskip-v4"
                                  , "SeaquestNoFrameskip-v4"
                                  , "SkiingNoFrameskip-v4"  # not in torchbeast paper
                                  , "SolarisNoFrameskip-v4"  # not in torchbeast paper
                                  , "SpaceInvadersNoFrameskip-v4"
                                  , "StarGunnerNoFrameskip-v4"
                                  #, "SurroundNoFrameskip-v4"  # doesn't work
                                  , "TennisNoFrameskip-v4"
                                  , "TimePilotNoFrameskip-v4"
                                  , "TutankhamNoFrameskip-v4"
                                  , "UpNDownNoFrameskip-v4"
                                  , "VentureNoFrameskip-v4"  # not in torchbeast paper
                                  , "VideoPinballNoFrameskip-v4"
                                  , "WizardOfWorNoFrameskip-v4"
                                  , "YarsRevengeNoFrameskip-v4"
                                  , "ZaxxonNoFrameskip-v4"])


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


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
        target=torch.flatten(actions, 0, 2),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(actions)
    return torch.sum(cross_entropy * advantages.detach())


class Net(nn.Module):
    def __init__(self, num_actions, num_tasks=1, use_lstm=False, use_popart=False, reward_clipping="abs_one"):
        super(Net, self).__init__()
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.use_lstm = use_lstm
        self.use_popart = use_popart
        self.reward_clipping = reward_clipping

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 4
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = PopArtLayer(core_output_size, num_tasks if self.use_popart else 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        if self.reward_clipping == "abs_one":
            clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        elif self.use_popart:
            clipped_reward = inputs["reward"].view(T * B, 1)

        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline, normalized_baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)

        baseline = baseline.view(T, B, self.num_tasks)
        normalized_baseline = normalized_baseline.view(T, B, self.num_tasks)
        action = action.view(T, B, 1)

        return (action, policy_logits, baseline, normalized_baseline), core_state


def inference(flags, inference_batcher, model, lock=threading.Lock()):  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            frame, reward, done, task, *_ = batched_env_outputs
            frame = frame.to(flags.actor_device, non_blocking=True)
            reward = reward.to(flags.actor_device, non_blocking=True)
            done = done.to(flags.actor_device, non_blocking=True)
            task = task.to(flags.actor_device, non_blocking=True)
            agent_state = nest.map(
                lambda t: t.to(flags.actor_device, non_blocking=True), agent_state
            )
            with lock:
                outputs = model(
                    dict(frame=frame, reward=reward, done=done, task=task), agent_state
                )
            outputs = nest.map(lambda t: t.cpu(), outputs)
            batch.set_outputs(outputs)


EnvOutput = collections.namedtuple(
    "EnvOutput", "frame rewards done task episode_step episode_return"
)
AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline normalized_baseline")
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    flags,
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(flags.learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        frame, reward, done, task, *_ = env_outputs

        lock.acquire()  # Only one thread learning at a time.
        learner_outputs, unused_state = model(
            dict(frame=frame, reward=reward, done=done, task=task), initial_agent_state
        )

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(env_outputs.rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = env_outputs.rewards

        discounts = (~env_outputs.done).float() * flags.discounting

        task = torch.nn.functional.one_hot(env_outputs.task.long(), flags.num_tasks)
        clipped_rewards = clipped_rewards[:, :, None]
        discounts = discounts[:, :, None]

        mu = model.baseline.mu[None, None, :]
        sigma = model.baseline.sigma[None, None, :]

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
            normalized_values=learner_outputs.normalized_baseline,
            mu=mu,
            sigma=sigma
        )

        with torch.no_grad():
            normalized_vs = (vtrace_returns.vs - mu) / sigma

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages * task,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            (normalized_vs - learner_outputs.normalized_baseline) * task
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        if flags.use_popart:
            model.baseline.update_parameters(vtrace_returns.vs, task)

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["step_env"] = stats.get("step_env", 0) + task.sum((0, 1))
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["mu"] = mu[0, 0, :]
        stats["sigma"] = sigma[0, 0, :]

        stats["learner_queue_size"] = learner_queue.size()

        plogger.log(stats)

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


def train(flags):
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

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.learner_device = torch.device("cuda:0")
        flags.actor_device = torch.device("cuda:1")
    else:
        logging.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static.
    learner_queue = actorpool.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = actorpool.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    model = Net(num_actions=flags.num_actions, num_tasks=flags.num_tasks, use_lstm=flags.use_lstm, use_popart=flags.use_popart, reward_clipping=flags.reward_clipping)
    model = model.to(device=flags.learner_device)

    actor_model = Net(num_actions=flags.num_actions, num_tasks=flags.num_tasks, use_lstm=flags.use_lstm, use_popart=flags.use_popart, reward_clipping=flags.reward_clipping)
    actor_model.to(device=flags.actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = actorpool.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=actor_model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    # Load state from a checkpoint, if possible.
    if os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                flags,
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
                plogger,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(flags, inference_batcher, actor_model),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            checkpointpath,
        )

    def savemodel():
        savemodelpath = checkpointpath.replace("model.tar", "intermediate/model." + str(stats.get("step")).zfill(9) + ".tar")
        logging.info("Saving model to %s", savemodelpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            savemodelpath,
        )

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        last_checkpoint_time = timeit.default_timer()
        last_savemodel_nsteps = 0
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)
            if start_step >= flags.total_steps:
                break
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timeit.default_timer() - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timeit.default_timer()

            if flags.save_model_every_nsteps > 0 and end_step >=  last_savemodel_nsteps + flags.save_model_every_nsteps:
                # save model every save_model_every_nsteps steps
                savemodel()
                last_savemodel_nsteps = end_step

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])
        checkpoint()

    # Done with learning. Stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actorpool_thread.join()

    for t in learner_threads + inference_threads:
        t.join()


def test(flags):

    if flags.xpid is None:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, "latest", "model.tar"))
        )
    else:
        if flags.intermediate_model_id is None:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
            )
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s/%s" % (flags.savedir, flags.xpid, "intermediate", "model." + flags.intermediate_model_id + ".tar"))
            )
    flags_orig = read_metadata(re.sub(r"model.*tar", "meta.json", checkpointpath).replace("/intermediate", ""))
    args_orig = flags_orig["args"]
    num_actions = args_orig.get("num_actions")
    num_tasks = args_orig.get("num_tasks", 1)
    use_lstm = args_orig.get("use_lstm", False)
    use_popart = args_orig.get("use_popart", False)
    reward_clipping = args_orig.get("reward_clipping", "abs_one")

    model = Net(num_actions=num_actions, num_tasks=num_tasks, use_lstm=use_lstm, use_popart=use_popart, reward_clipping=reward_clipping)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    if 'baseline.mu' not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
        checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_action_space = False
    if flags.num_actions == 18:
        full_action_space = True
    gym_env = create_env(flags.env, full_action_space)
    env = Environment(gym_env)

    observation = env.initial()
    returns = []

    with torch.no_grad():
        i = 0
        while len(returns) < flags.num_episodes:
            if flags.mode == "test_render":
                time.sleep(0.05)
                env.gym_env.render()
            agent_outputs = model(observation, torch.tensor)
            policy_outputs, _ = agent_outputs
            #observation = env.step(policy_outputs[0])
            action = torch.tensor(np.random.randint(0, num_actions))
            observation = env.step(action)
            if observation["done"].item():
                returns.append(observation["episode_return"].item())
                logging.info(
                    "Episode ended after %d steps. Return: %.1f",
                    observation["episode_step"].item(),
                    observation["episode_return"].item(),
                )
                print(flags.xpid, flags.intermediate_model_id,  flags.env, observation["episode_step"].item(), observation["episode_return"].item(), sep=",", flush=True)
            i = i + 1

    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", flags.num_episodes, sum(returns) / len(returns)
    )


def record(flags):
    torch.manual_seed(0)
    if flags.xpid is None:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, "latest", "model.tar"))
        )
    else:
        if flags.intermediate_model_id is None:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
            )
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s/%s" % (flags.savedir, flags.xpid, "intermediate", "model." + flags.intermediate_model_id + ".tar"))
            )
    flags_orig = read_metadata(re.sub(r"model.*tar", "meta.json", checkpointpath).replace("/intermediate", ""))
    args_orig = flags_orig["args"]
    num_actions = args_orig.get("num_actions")
    num_tasks = args_orig.get("num_tasks", 1)
    use_lstm = args_orig.get("use_lstm", False)
    use_popart = args_orig.get("use_popart", False)
    reward_clipping = args_orig.get("reward_clipping", "abs_one")

    actions = []
    if flags.actions is not None:
        f = open(flags.actions, "r")
        for line in f:
            actions.append(int(line.replace("tensor([[[", "").replace("]]])", "")))
        f.close()

    model = Net(num_actions=num_actions, num_tasks=num_tasks, use_lstm=use_lstm, use_popart=use_popart, reward_clipping=reward_clipping)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    if 'baseline.mu' not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
        checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_action_space = False
    if flags.num_actions == 18:
        full_action_space = True
    gym_env = create_env_det(flags.env, full_action_space)
    gym_env.seed(0)
    env = Environment(gym_env)

    observation = env.initial()

    folder = re.sub(r"/model.*tar", "/movies/play_raw/", checkpointpath.replace("/intermediate", ""))
    with torch.no_grad():
        for i in range(10000):
            #time.sleep(0.05)
            #env.gym_env.render()
            filename = folder + flags.xpid + "_" + ( str(flags.intermediate_model_id).zfill(9) if flags.intermediate_model_id is not None else "None" ) + "_" + flags.env + "_" + str(i).zfill(5) + ".png"
            img = Image.fromarray(env.gym_env.ale.getScreenRGB2(), 'RGB').resize([160, 210])
            img.save(filename, format='png')
            agent_outputs = model(observation, torch.tensor)
            policy_outputs, _ = agent_outputs
            action = policy_outputs[0]
            if len(actions) > 0:
                action = torch.tensor(actions[i])
            observation = env.step(action)
            if observation["done"].item():
                print("episode_return:", observation["episode_return"], "step:", observation["episode_step"])
                break
            i = i + 1

    env.close()


def env_info(flags):
    envs = atari_environments

    l = envs.shape[0]
    print(l, "environments")
    for i in range(l):
        env = create_env(envs[i])
        print(env)
        print(env.action_space)
        print(env.get_action_meanings())
    env = create_env(envs[0], full_action_space=True)
    print(env)
    print(env.action_space)
    print(env.get_action_meanings())
    env.close()


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    if flags.env == "six":
        flags.env = "AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4,NameThisGameNoFrameskip-v4,PongNoFrameskip-v4,SpaceInvadersNoFrameskip-v4"
    elif flags.env == "three":
        flags.env = "AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4"
    flags.num_tasks = len(flags.env.split(",")) if flags.use_popart else 1

    if flags.start_servers and flags.mode == "train":
        if flags.env == "all":
            flags.env = ",".join(atari_environments)
            if flags.num_actors != atari_environments.shape[0]:
                flags.num_actors = atari_environments.shape[0]
                logging.info("Changed number of environment servers to '%s'", str(atari_environments.shape[0]))
        command = [
            "python",
            "-m",
            "torchbeast.polybeast_env",
            f"--num_servers={flags.num_actors}",
            f"--pipes_basename={flags.pipes_basename}",
            f"--env={flags.env}",
        ]
        if flags.use_popart:
            command.append("--multitask")
        logging.info("Starting servers with command: " + " ".join(command))
        server_proc = subprocess.Popen(command)

    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)

    if flags.mode == "test" or flags.mode == "test_render":
        test(flags)

    if flags.mode == "record":
        record(flags)

    if flags.mode == "env_info":
        env_info(flags)

    if flags.start_servers and flags.mode == "train":
        # Send Ctrl-c to servers.
        server_proc.send_signal(signal.SIGINT)


def create_env_det(env_name, full_action_space=False, noop=20):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari_det(env_name, full_action_space=full_action_space, noop=noop),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )

def create_env(env_name, full_action_space=False):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(env_name, full_action_space=full_action_space),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
