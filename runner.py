import random
from collections import deque

import numpy as np

from common.arguments import get_args
from game.board import Board
from game.game import Game
from learn.agent import Agent
from player.mcts_player import MCTSPlayer


class Runner:
    def __init__(self, args):
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.train_step = args.train_step
        self.width = args.width
        self.height = args.height
        self.kl_threshold = args.kl_threshold
        self.save_cycle = args.save_cycle

        self.board = Board(args.width, args.height, args.n_in_row)
        self.game = Game(self.board, args)
        self.agent = Agent(args)
        self.mcts_player = MCTSPlayer(self.agent, args)

        self.buffer = deque(maxlen=args.buffer_size)

    def generate_episodes(self):
        episode = self.game.self_play(self.mcts_player)
        for state, mcts_prob, winner in zip(*episode.values()):
            for k in [1, 2, 3, 4]:
                # rotation
                state_p = np.array([np.rot90(s, k) for s in state])
                mcts_prob_p = np.rot90(np.flipud(mcts_prob.reshape(self.height, self.width)), k)
                self.buffer.append((state_p, np.flipud(mcts_prob_p).flatten(), winner))
                # flipping
                state_p = np.array([np.fliplr(s) for s in state_p])
                mcts_prob_p = np.fliplr(mcts_prob_p)
                self.buffer.append((state_p, np.flipud(mcts_prob_p).flatten(), winner))

    def policy_update(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, mcts_probs, winners_z = zip(*batch)
        states, mcts_probs, winners_z = np.stack(states), np.stack(mcts_probs), np.stack(winners_z)
        probs_o, value_o = self.agent.predict_batch(states)
        for idx in range(self.train_step):
            loss = self.agent.train_step(states, mcts_probs, winners_z)
            probs_n, value_n = self.agent.predict_batch(states)
            kl = np.mean(np.sum(probs_o * (np.log(probs_o + 1e-10) - np.log(probs_n + 1e-10)), axis=1))  # KL-Divergence
            if kl > self.kl_threshold * 4.0:
                break

        return loss, kl

    def train(self):
        for idx in range(self.n_epochs):
            self.generate_episodes()
            if len(self.buffer) > self.batch_size:
                loss, kl = self.policy_update()
                self.agent.scheduler.step(kl)
                print("[Epoch {}] loss: {:6f}, lr: {:6f}, kl: {:5f}".format(idx + 1, loss, self.agent.get_current_lr(), kl))
            else:
                print("[Epoch {}]".format(idx + 1))
            if (idx + 1) % self.save_cycle == 0:
                self.agent.save_model()

    def test(self):
        self.game.human_play(self.mcts_player)


def main():
    args = get_args()
    print("using device: " + str(args.device))

    runner = Runner(args)
    # runner.train()
    runner.test()


if __name__ == "__main__":
    main()
