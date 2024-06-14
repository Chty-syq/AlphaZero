import numpy as np

from game.graphic import Graphic
from player.mcts_player import MCTSPlayer


class Game:
    def __init__(self, board, args):
        self.board = board
        self.graphic = Graphic(args)

    def self_play(self, player: MCTSPlayer):
        self.board.init()

        states, mcts_probs, cur_players = [], [], []
        end, winner = self.board.game_end()
        while not end:
            action, move_probs = player.choose_action(self.board)

            states.append(self.board.get_full_state())
            mcts_probs.append(move_probs)
            cur_players.append(self.board.cur_player)

            self.graphic.move(self.board.cur_player, self.board.index_to_location(action))
            self.board.move(action)

            end, winner = self.board.game_end()

        player.reset()
        self.graphic.clear()
        
        winners_z = np.zeros_like(cur_players)
        if winner >= 0:
            winners_z[np.array(cur_players) == winner] = 1.0
            winners_z[np.array(cur_players) != winner] = -1.0

        episode = dict(states=states, mcts_probs=mcts_probs, winners_z=winners_z)
        return episode
