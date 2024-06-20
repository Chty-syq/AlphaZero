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

    def human_play(self, player: MCTSPlayer):
        def on_click(event):
            if self.board.game_end()[0]:
                return
            w = round(event.x / self.graphic.line_dist) - 1
            h = round(event.y / self.graphic.line_dist) - 1
            dist = np.linalg.norm([(w + 1) * self.graphic.line_dist - event.x, (h + 1) * self.graphic.line_dist - event.y])
            if dist > 0.4 * self.graphic.line_dist:
                return
            action = self.board.location_to_index((h, w))
            self.graphic.move(self.board.cur_player, self.board.index_to_location(action))
            self.board.move(action)
            if self.board.game_end()[0]:
                return
            action, _ = player.choose_action(self.board, evaluate=True)
            self.graphic.move(self.board.cur_player, self.board.index_to_location(action))
            self.board.move(action)

        self.board.init()
        self.graphic.canvas.bind("<Button-1>", on_click)
        self.graphic.handle()
