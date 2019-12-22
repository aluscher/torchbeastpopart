"""
Adapted from https://github.com/cjlovering/Towards-Interpretable-Reinforcement-Learning-Using-Attention-Augmented-Agents-Replication
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Tuple


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """Initialize stateful ConvLSTM cell.

        Parameters
        ----------
        input_channels : ``int``
            Number of channels of input tensor.
        hidden_channels : ``int``
            Number of channels of hidden state.
        kernel_size : ``int``
            Size of the convolutional kernel.

        Paper
        -----
        https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf

        Referenced code
        ---------------
        https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
        """
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, prev_hidden=()):
        # TODO: should consider moving this to the constructor because this seems like really bad practice
        if self.Wci is None:
            _, _, height, width = x.shape
            hidden = self.hidden_channels
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(x.device)
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(x.device)
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(x.device)

        h, c = prev_hidden

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        return ch, cc

    def initial_state(self, batch_size, hidden, height, width):
        return self.init_hidden(batch_size, hidden, height, width)

    def init_hidden(self, batch_size, hidden, height, width):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True)
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True)
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True)
        return (
            torch.zeros(batch_size, hidden, height, width),
            torch.zeros(batch_size, hidden, height, width)
        )


class VisionNetwork(nn.Module):

    def __init__(self, frame_height, frame_width, in_channels=3, hidden_channels=128):
        super(VisionNetwork, self).__init__()
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels

        # padding s.t. the output shapes match the paper. TODO: might have to be adjusted...
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(in_channels=self._in_channels, out_channels=32, kernel_size=(8, 8), stride=4, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=2)
        )
        self.vision_lstm = ConvLSTMCell(input_channels=64, hidden_channels=self._hidden_channels, kernel_size=3)

    def initial_state(self, batch_size, dummy_frame):
        cnn_output = self.vision_cnn(dummy_frame)
        height, width = tuple(cnn_output.shape[2:])
        return self.vision_lstm.initial_state(batch_size, self._hidden_channels, height, width)

    def forward(self, x, prev_vision_core_state):
        x = x.permute(0, 3, 1, 2)
        vision_core_output, vision_core_state = self.vision_lstm(self.vision_cnn(x), prev_vision_core_state)
        return vision_core_output.permute(0, 2, 3, 1), (vision_core_output, vision_core_state)


class QueryNetwork(nn.Module):
    def __init__(self, num_queries, c_k, c_s):
        super(QueryNetwork, self, ).__init__()
        # TODO: Add proper non-linearity. => seems like there is "proper nonlinearity" (with ReLUs)
        self._num_queries = num_queries
        self._c_o = c_k + c_s
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self._num_queries * self._c_o),
            nn.ReLU(),
            nn.Linear(self._num_queries * self._c_o, self._num_queries * self._c_o)
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, self._num_queries, self._c_o)


class SpatialBasis:
    # TODO: Implement Spatial.
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=27, width=20, channels=64):
        # TODO: go through the math here, but at least it already seems implemented...
        self._height = height
        self._width = width
        self._channels = channels
        self._s = None

        self.init()

    def __call__(self, x):
        batch_size, x_height, x_width, *_ = x.size()
        re_init = False
        if self._height != x_height:
            self._height = x_height
            re_init = True
        if self._width != x_width:
            self._width = x_width
            re_init = True
        if re_init:
            self.init()

        # Stack the spatial bias (for each batch) and concat to the input.
        s = torch.stack([self._s] * batch_size).to(x.device)
        return torch.cat([x, s], dim=3)

    def init(self):
        h, w, d = self._height, self._width, self._channels

        p_h = torch.mul(torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()) * (np.pi / h)
        p_w = torch.mul(torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()) * (np.pi / w)

        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 8  # size of U, V.
        u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum('hwu,hwv->hwuv', torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self._s = out


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class AttentionAugmentedAgent(nn.Module):
    def __init__(
            self,
            observation_shape,
            num_actions,
            use_lstm: bool = True,
            hidden_size: int = 256,
            c_v: int = 120,
            c_k: int = 8,
            c_s: int = 64,
            num_queries: int = 4,
    ):
        """Agent implementing the attention agent."""
        super(AttentionAugmentedAgent, self).__init__()
        self.hidden_size = hidden_size
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries

        self.vision = VisionNetwork(self.observation_shape[1], self.observation_shape[2],
                                    in_channels=self.observation_shape[0])
        self.query = QueryNetwork(num_queries, c_k, c_s)
        # TODO: Implement SpatialBasis. I think it's implemented, isn't it?
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1031 x 512
            nn.Linear((c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
        )

        self.policy_core = nn.LSTM(hidden_size, hidden_size)
        self.prev_output = None
        self.prev_hidden = None

        self.policy_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        self.values_head = nn.Sequential(nn.Linear(hidden_size, 1))

    def reset(self):
        self.vision.reset()
        self.prev_output = None
        self.prev_hidden = None

    def initial_state(self, batch_size):
        dummy_frame = torch.zeros(1, *self.observation_shape)
        vision_core_initial_state = tuple(s.unsqueeze(0) for s in self.vision.initial_state(batch_size, dummy_frame))
        policy_core_initial_state = tuple(
            torch.zeros(self.policy_core.num_layers, batch_size, self.policy_core.hidden_size)
            for _ in range(2)
        )
        return vision_core_initial_state + policy_core_initial_state

    def forward(self, inputs, state=(), test=False):
        # TODO: change all this to the input being a dictionary just like in the example network
        # TODO: if possible use the optimisation of concatenating things in time
        #  => might actually not be worth the trouble though, because we have "nested" LSTMs
        #  => in any case, can/should probably only happen in the vision network (?)

        # input frames are formatted: (time_steps, batch_size, frame_stack, height, width)
        # the original network is designed for (batch_size, height, width, num_channels)
        # there are a couple options to solve this:
        # - use grayscale, stack frames, use those as channels
        # - use full colour, stack frames, resulting in 4 * 3 = 12 channels
        # - use full colour, don't stack frames (similar to original paper)
        # IMPORTANT NOTE: for the latter, the original paper still sends the same action 4 times,
        # so the following might be a better option (as far as implementation goes)
        # - use full colour, stack frames, use only the last one
        # => for now, I'm just going to use the first method

        # (time_steps, batch_size, frame_stack, height, width)
        x: torch.Tensor = inputs["frame"]
        if test: print("x shape:", x.shape)
        time_steps, batch_size, *_ = x.shape
        if test: print("time_steps, batch_size:", time_steps, batch_size)
        # (time_steps, batch_size, frame_stack, height, width)
        x = x.float() / 255.0
        # (time_steps, batch_size, height, width, frame_stack) to match the design of the network
        x = x.permute(0, 1, 3, 4, 2)
        if test: print("x shape after permuting:", x.shape)

        # (time_steps * batch_size, 1) => probably needs to be "expanded" (unsqueezed in at least one dimension)
        prev_reward = inputs["reward"].view(time_steps, batch_size, 1)
        if test: print("prev_reward shape:", prev_reward.shape)
        # (time_steps * batch_size, num_actions)
        prev_action = F.one_hot(inputs["last_action"].view(time_steps, batch_size), self.num_actions).float()  # 1-hot
        if test: print("prev_action shape:", prev_action.shape)
        # TODO: add shape here
        not_done = (~inputs["done"]).float()
        if test: print("not_done shape:", not_done.shape)

        vision_core_output_list = []
        vision_core_state = tuple(s.squeeze(0) for s in state[:2])
        if test:
            print("state shapes:", [s.shape for s in state])
        for x_batch, not_done_batch in zip(x.unbind(), not_done.unbind()):
            # x_batch should have shape (batch_size, height, width, frame_stack)
            # not_done_batch should have shape (batch_size)
            # => both of these could also be (1, ...), not sure about that

            # to do this the way it has been done in the torchbeast code the hidden states
            # of both the "inner" and "outer" LSTM should be returned by and fed to the
            # forward() function explicitly, so that "done" states can be zeroed out

            not_done_batch = not_done_batch.view(-1, 1, 1, 1)
            vision_core_state = tuple(not_done_batch * s for s in vision_core_state)
            if test:
                print("not_done_batch shape:", not_done_batch.shape)
                print("vision_core_state shapes:", vision_core_state[0].shape, vision_core_state[1].shape)

            # continue with all the stuff below...

            # 1 (a). Vision.
            # --------------
            # (n, h, w, c_k + c_v) / (batch_size, height, width, c_k + c_v)
            vision_core_output, vision_core_state = self.vision(x_batch, vision_core_state)
            vision_core_output_list.append(vision_core_output)
            # for clarity vision_core_output.unsqueeze(0) might be better, because it would be clear that this
            # is the result for one time step, but since we merge time and batch in the following steps anyway,
            # we can also just "discard" the time dimension and get the same result when we concatenate
            # the results for each time step

            if test:
                print("\nO shape:", vision_core_output.shape)
                print("vision_core_state shapes:", vision_core_state[0].shape, vision_core_state[1].shape)
        vision_core_state = tuple(s.unsqueeze(0) for s in vision_core_state)

        vision_core_output = torch.cat(vision_core_output_list)
        if test: print("\nvision_core_output shape:", vision_core_output.shape)

        # (n, h, w, c_k), (n, h, w, c_v) / (batch_size, height, width, c_k), (batch_size, height, width, c_v)
        keys, baseline = vision_core_output.split([self.c_k, self.c_v], dim=3)
        if test: print("keys/values shapes:", keys.shape, baseline.shape)
        # (n, h, w, c_k + c_s), (n, h, w, c_v + c_s) /
        # (batch_size, height, width, c_k + c_s), (batch_size, height, width, c_v + c_s)
        keys, baseline = self.spatial(keys), self.spatial(baseline)
        if test: print("keys/values shapes after spatial:", keys.shape, baseline.shape)

        # reshape the keys and values tensors so that they can be separated in the time dimension
        keys = keys.view(time_steps, batch_size, *keys.shape[1:])
        baseline = baseline.view(time_steps, batch_size, *baseline.shape[1:])
        if test: print("keys/values shapes after view:", keys.shape, baseline.shape)

        policy_core_output_list = []
        policy_core_state = state[2:]
        for keys_batch, values_batch, prev_reward_batch, prev_action_batch, not_done_batch in zip(
                keys.unbind(), baseline.unbind(), prev_reward.unbind(), prev_action.unbind(), not_done.unbind()):

            not_done_batch = not_done_batch.view(1, -1, 1)
            if test:
                print("\npolicy_core_state shapes:", policy_core_state[0].shape, policy_core_state[1].shape)
                print("not_done_batch shape:", not_done_batch.shape)

            policy_core_state = tuple(not_done_batch * s for s in policy_core_state)
            if test: print("\npolicy_core_state shapes:", policy_core_state[0].shape, policy_core_state[1].shape)

            # 1 (b). Queries.
            # --------------
            # (n, num_queries, c_k + c_s) / (batch_size, num_queries, c_k + c_s)
            queries = self.query(policy_core_state[0])
            if test:
                print("queries shape:", queries.shape)
                print("queries after reshaping:", queries.transpose(2, 1).unsqueeze(1).shape)

            # 2. Answer.
            # ----------
            # (n, h, w, num_queries) / (batch_size, height, width, num_queries)
            answer = torch.matmul(keys_batch, queries.transpose(2, 1).unsqueeze(1))
            if test: print("answer (1) shape:", answer.shape)
            # (n, h, w, num_queries) / (batch_size, height, width, num_queries)
            answer = spatial_softmax(answer)
            if test: print("answer (2) shape:", answer.shape)
            # (n, 1, 1, num_queries) / (batch_size, 1, 1, num_queries)
            answer = apply_alpha(answer, values_batch)  # TODO: what does this do? => probably equation (5) in the paper...
            if test: print("answer (3) shape:", answer.shape)

            # (n, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + num_actions) /
            # (batch_size, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + num_actions)
            # TODO: check whether one-hot vs single value encoding matters
            if test:
                print("chunks 1:", [c.shape for c in torch.chunk(answer, self.num_queries, dim=1)])
                print("chunks 2:", [c.shape for c in torch.chunk(queries, self.num_queries, dim=1)])
                print("rewards:", prev_reward_batch.unsqueeze(1).float().shape)
                print("actions:", prev_action_batch.unsqueeze(1).float().shape)
            answer = torch.cat(
                torch.chunk(answer, self.num_queries, dim=1)
                + torch.chunk(queries, self.num_queries, dim=1)
                + (prev_reward_batch.unsqueeze(1).float(), prev_action_batch.unsqueeze(1).float()),
                dim=2,
            ).squeeze(1)
            if test: print("answer after concatenating:", answer.shape)
            # (n, hidden_size) / (batch_size, hidden_size)
            answer = self.answer_processor(answer)
            if test: print("answer after answer processor:", answer.shape)

            # 3. Policy.
            # ----------
            # (n, hidden_size) / (batch_size, hidden_size)
            policy_core_output, policy_core_state = self.policy_core(answer.unsqueeze(0), policy_core_state)
            if test: print("policy_core_output shape:", policy_core_output.squeeze(0).shape)
            policy_core_output_list.append(policy_core_output.squeeze(0))
            # squeeze() is needed because the LSTM input has an "extra" dimensions for the layers of the LSTM,
            # of which there is only one in this case; therefore, the concatenated input vector has an extra
            # dimension and the output as well

        output = torch.cat(policy_core_output_list)

        # 4, 5. Outputs.
        # --------------
        # (n, num_actions) / (batch_size, num_actions)
        policy_logits = self.policy_head(output)
        # (n, 1) / (batch_size, 1)
        baseline = self.values_head(output)

        if test:
            print("output shape:", output.shape)
            print("policy_logits shape: {}".format(policy_logits.shape))
            print("baseline shape: {}".format(baseline.shape))

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        if test:
            print("action shape: {}".format(action.shape))

        policy_logits = policy_logits.view(time_steps, batch_size, self.num_actions)
        baseline = baseline.view(time_steps, batch_size)
        action = action.view(time_steps, batch_size)

        if test:
            print("policy_logits shape: {}".format(policy_logits.shape))
            print("baseline shape: {}".format(baseline.shape))
            print("action shape: {}\n".format(action.shape))

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            vision_core_state + policy_core_state
        )
