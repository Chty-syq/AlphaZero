import numpy as np

from learn.mcts import MCTS


class MCTSPlayer:
    def __init__(self, agent, args):
        self.mcts = MCTS(agent, args)

        self.n_actions = args.n_actions
        self.epsilon = args.epsilon
        self.noise = args.dirichlet_noise

    def reset(self):
        self.mcts.update(-1)

    def choose_action(self, board, evaluate=False):
        actions, probs = self.mcts.get_move_probs(board)
        move_probs = np.zeros(self.n_actions)
        move_probs[actions] = probs

        if not evaluate:
            probs = (1 - self.epsilon) * probs + self.epsilon * np.random.dirichlet(self.noise * np.ones_like(probs))  # 引入噪声
            action = np.random.choice(actions, p=probs)
            self.mcts.update(action)
        else:
            action = np.random.choice(actions, p=probs)
            self.mcts.update(-1)

        return action, move_probs
