import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
import torch.nn.functional as F
from torch.distributions import Normal


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action

        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            actions_policy = self.forward(obs)
            log_probs = F.log_softmax(actions_policy, dim=-1)
            probs = torch.exp(log_probs)

            # Sampling
            m = torch.distributions.Categorical(probs)
            sampled_action = m.sample()
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            means = self.forward(obs)
            normal_distribution = Normal(means, torch.exp(self.logstd))
            sampled_action = normal_distribution.sample()

        return ptu.to_numpy(sampled_action)
    
    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            action = self.logits_net(obs)
            while action.isnan().all():
                assert 1 ==0
                action = self.logits_net(obs+torch.normal(mean=0,std=0.2,size=obs.shape))

        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            action = self.mean_net(obs)
        return action

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        
        assert 1 == 0
        # TODO: implement the policy gradient actor update.
        actions_policy = self.forward(obs)
        #advantages = torch.rand_like(advantages,requires_grad = True)
        log_probs = F.log_softmax(actions_policy, dim=-1)

        # Assume 'actions' are the indices of the actions taken, in the same batch order as log_probs
        # Gather only the log probabilities of actions that were actually taken
        gathered_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze()

        # Compute the loss
        loss = -torch.mean(gathered_log_probs * advantages)



        self.optimizer.zero_grad() # zero's out gradients
        loss.backward() # populate gradients
        self.optimizer.step() # update each parameter via gradient descent

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        if self.discrete:
            # TODO: implement the policy gradient actor update.
            actions_policy = self.forward(obs)
            #advantages = torch.rand_like(advantages,requires_grad = True)
            log_probs = F.log_softmax(actions_policy, dim=-1)

            # Assume 'actions' are the indices of the actions taken, in the same batch order as log_probs
            # Gather only the log probabilities of actions that were actually taken
            #gathered_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze()
            gathered_log_probs = log_probs.gather(1, actions.unsqueeze(-1).long()).squeeze()

            # Compute the loss
            loss = -torch.mean(gathered_log_probs * advantages)
        else: #continuous

            # TODO: define the forward pass for a policy with a continuous action space.
            means = self.forward(obs)
            # Split into means and stds
            normal_distribution = Normal(means, torch.exp(self.logstd))
            log_prob = normal_distribution.log_prob(actions)   
            joint_log_prob = log_prob.sum(axis=1)

            loss = -torch.mean(joint_log_prob * advantages)


        self.optimizer.zero_grad() # zero's out gradients
        loss.backward() # populate gradients
        self.optimizer.step() # update each parameter via gradient descent



        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
