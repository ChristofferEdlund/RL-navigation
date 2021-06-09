"""
Created on 2021-05-05. 11:52

@author: Christoffer Edlund
"""

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from datatypes import SumTree

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(2**17)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double=False, double_update_frq=10):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.double = double
        self.double_update_frq = double_update_frq
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.lr_iter = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory

        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def weighted_mse(self, pred, target, weights):
        return (weights * (pred - target)**2).mean()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        self.optimizer.zero_grad()

        pred_Qs = self.qnetwork_local(states).gather(1, actions)

        # Double Q-learning
        if self.double:
            pred2_Qs = self.qnetwork_local(next_states)
            best_actions = pred2_Qs.detach().argmax(1).unsqueeze(1)
            target_Qs = self.qnetwork_target(next_states).gather(1, best_actions).detach() * (1 - dones)

        else:
            target_Qs = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) * (1 - dones) # [0].unsqueeze(1)

        target_Qs = rewards + gamma * target_Qs
        # Account for the important sampling weights
        #print(is_weights.dtype)
        loss = F.mse_loss(pred_Qs, target_Qs)

        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        
        if not self.double:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        else:
            (self.lr_iter % self.double_update_frq) == 0
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.lr_iter = self.lr_iter + 1

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class PrioAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, priority_alpha=0.6, priority_beta=0.4, double=False, double_update_frq=10):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.double = double
        self.double_update_frq = double_update_frq
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.priority_alpha = priority_alpha
        self.td_epsilon = 1e-2

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # PDQN paper have the learning rate reduced by a factor of 4

        # Replay memory
        self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.priority_alpha, priority_beta)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.lr_iter = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory

        # For new experiences in Prioritized Experience Replay, set prio to max(prios), (or epsilon)
        td_error = self.memory.max_priority()
        
        # If this is the first experience recorded, put priority to 1
        if td_error == 0:
            td_error = 1
            
        # Clipping the reward as done in the PBE paper
        reward = np.clip(reward, -1, 1)
        self.memory.add(state, action, reward, next_state, td_error, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def weighted_mse(self, pred, target, weights):
        return (weights * (pred - target)**2).mean()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, is_weights = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        self.optimizer.zero_grad()

        pred_Qs = self.qnetwork_local(states).gather(1, actions)
        
        # Double Q-learning
        if self.double:
            pred2_Qs = self.qnetwork_local(next_states)
            best_actions = pred2_Qs.detach().argmax(1).unsqueeze(1)
            target_Qs = self.qnetwork_target(next_states).gather(1, best_actions).detach() * (1 - dones)

        else:
            target_Qs = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) * (1 - dones) # [0].unsqueeze(1)
        

        target_Qs = rewards + gamma * target_Qs
        # Account for the important sampling weights

        loss = self.weighted_mse(pred_Qs, target_Qs, torch.tensor(is_weights).to(device).float())  # <- Change to L2 loss, gradient clipping

        loss.backward()
        self.optimizer.step()

        # ------------ Update the experience replay buffer -------- #
        td_errors = torch.abs(target_Qs.cpu().numpy() - pred_Qs.detach().cpu()) + self.td_epsilon

        td_errors = torch.clamp(td_errors, 0, 1).numpy()
        for td_error, idx in zip(td_errors, indices):

            self.memory.update(idx, td_error[0])


        # ------------------- update target network ------------------- #

        
        if not self.double:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        else:
            (self.lr_iter % self.double_update_frq) == 0
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.lr_iter = self.lr_iter + 1

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def update_isbeta(self, beta):
        self.memory.update_beta(beta)
            
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, epsilon=0.00001, priority_alpha=0.6, priority_beta=0.4):
        """Initialize a PriorityReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = SumTree(max_buffer=buffer_size, alpha=priority_alpha, beta=priority_beta)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, td_error, done):
        """Add a new experience to memory."""

        td_error = abs(td_error)

        e = self.experience(state, action, reward, next_state, done)
        self.memory.add(e, td_error)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # Todo, sample by probability

        experiences, indices, is_weights = self.memory.sample(self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones, indices, is_weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update(self, idx, priority):
        self.memory.update(idx, priority)
    
    def update_beta(self, beta):
        self.memory.update_beta(beta)
        
        
    def max_priority(self):
        return self.memory.max_priority