"""
Created on 2021-05-10. 21:31 

@author: Christoffer Edlund
"""
import numpy as np


class SumTree:
    def __init__(self, max_buffer=int(1e5), alpha=0.6, beta=0.4):
        # self.current_size = init_size
        self.max_buffer = max_buffer
        self.occupy_idx = 0
        self.residents = 0

        self.nodes, self.root = self.create_tree(np.zeros(max_buffer))
        self.root = self.root[0]
        self.alpha = alpha
        self.beta = beta
        self._max_priority = 0
        
    @property
    def max_priority(self):
        return self._max_priority
        

    def _get_prob(self, priority):
        return (priority ** self.alpha)

    def create_tree(self, init_values):
        assert len(init_values) % 2 == 0, "Error, initiate tree with even number of nodes, preferably 2^n"
        idx = 0
        leafs = []
        for priority in init_values:
            leafs.append(SumNode(is_leaf=True, idx=idx, priority=priority))

            idx += 1
            # TODO make idx something sensiable
            # construct tree from nodes

        nodes = leafs

        while len(nodes) > 1: 
            node_iter = iter(nodes)
            # This assumes that nodes is divisable by 2.
            nodes = [SumNode(n1, n2) for n1, n2 in zip(node_iter, node_iter)]

        return leafs, nodes

    def add(self, data, priority):
        # In the original paper, priority max(nodes.priority).
        
        # If the new priority is larger than the max_priority, store it.
        priority = self._get_prob(priority)
        self._max_priority = max(self._max_priority, priority)

        # Check that tree is big enough
        if self.occupy_idx == self.max_buffer:
            self.occupy_idx = 0

        node = self.nodes[self.occupy_idx]
        node.data = data
        if node.priority is None:
            delta_prio = priority
        else:
            delta_prio = priority - node.priority
        node.priority = priority

        # Propegate priority
        self._propegate_priority(node, delta_prio)

        self.occupy_idx += 1
        self.residents = min(self.residents + 1, self.max_buffer)

    def update(self, idx, priority, data=None):
        priority = self._get_prob(priority)

        # Update max priority variable (used for new experinces)
        self._max_priority = max(self._max_priority, priority)

        node = self.nodes[idx]
        w = node.priority
        node.priority = priority
        if data:
            node.data = data
        self._propegate_priority(node, priority - w)

    def _propegate_priority(self, node, delta_priority):

        while node.parent is not None:
            node.parent.priority += delta_priority
            node = node.parent

    def sum_priority(self):
        return self.root.priority

    def lookup(self, prob):
        assert prob <= self.root.priority, f"Prob is higher than all prioritys combined. Prob: {prob}, Sum priority: {self.sum_priority()}"
        node = self.root

        while not node.is_leaf:
            lw = node.left.priority
            if lw > prob:
                node = node.left
            else:
                node = node.right
                prob = prob - lw

        return node.data, node.idx, node.priority
    
    def __len__(self):
        return self.residents


    def sample(self, n):
        sample_prob = np.random.uniform(0, self.sum_priority(), size=n)

        data = []
        idx = []
        prio = []
        for s in sample_prob:
            d, i, p = self.lookup(s)
            data.append(d)
            idx.append(i)
            prio.append(p)

        # Calculate the important samplng weights as PER - article

        sampling_probability = np.array(prio) / np.array(self.sum_priority())
        is_weights = (sampling_probability * self.residents) ** -self.beta
        is_weights = is_weights.astype(np.float64) / is_weights.max() # <- For stability reasons, we always normalize weights by 1/max(w) - from PDQN paper

        return data, idx, is_weights

    def update_beta(self, beta):
        self.beta = beta


class SumNode():
    def __init__(self, left=None, right=None, is_leaf=False, idx=None, priority=None, data=None):
        self.left = left
        self.right = right
        self.idx = idx  # Used for leaf nodes
        self.priority = priority
        self.data = data
        self.is_leaf = is_leaf

        if not is_leaf:
            self.priority = self.left.priority
            if self.right is not None:
                self.priority += self.right.priority

        self.parent = None

        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self
