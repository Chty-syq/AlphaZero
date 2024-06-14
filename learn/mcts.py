import copy

import numpy as np

from common import utils
from game.board import Board


class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.child = {}
        self.n_visits = 0
        self.q = 0
        self.prior_p = prior_p

    def calculate_uct(self, c_puct):
        return self.q + c_puct * self.prior_p * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)

    def select(self, c_puct):
        return max(self.child.items(), key=lambda item: item[1].calculate_uct(c_puct))

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.child:
                self.child[action] = TreeNode(self, prob)

    def back_propagate(self, leaf_value):
        self.n_visits += 1
        self.q += (leaf_value - self.q) / self.n_visits
        if self.parent:
            self.parent.back_propagate(-leaf_value)


class MCTS:
    def __init__(self, agent, args):
        self.root = TreeNode(None, 1.0)
        self.agent = agent

        self.c_puct = args.c_puct
        self.n_rollout = args.n_rollout
        self.n_actions = args.n_actions
        self.tau = args.tau

    def rollout(self, board: Board):
        node = self.root
        while node.child:
            action, node = node.select(self.c_puct)
            board.move(action)

        action_probs, leaf_value = self.agent.predict(board)
        end, winner = board.game_end()
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0.0 if winner < 0 else -1.0

        node.back_propagate(-leaf_value)

    def get_move_probs(self, board: Board):
        for _ in range(self.n_rollout):
            self.rollout(copy.deepcopy(board))

        actions, n_visits = zip(*[(action, node.n_visits) for action, node in self.root.child.items()])
        probs = utils.softmax(np.log(np.array(n_visits) + 1e-10) / self.tau)
        return list(actions), probs

    def update(self, action):
        if action in self.root.child:
            self.root = self.root.child[action]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
