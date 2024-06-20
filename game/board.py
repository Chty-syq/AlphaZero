import numpy as np


class Board:
    def __init__(self, width, height, k):
        """
        :param width,height: 棋盘大小
        :param k: 连续k子获胜
        """
        self.width = width
        self.height = height
        self.k = k

        self.cur_player, self.states, self.availables, self.last_move = 0, {}, [], -1

    def init(self):
        if self.width < self.k or self.height < self.k:
            raise Exception("Invalid board")
        self.cur_player = 0
        self.states = {}
        self.availables = list(range(self.width * self.height))
        self.last_move = -1

    def location_to_index(self, location):
        h, w = location[0], location[1]
        return h * self.width + w

    def index_to_location(self, index):
        h, w = index // self.width, index % self.width
        return h, w

    def get_full_state(self):
        full_state = np.zeros((4, self.height, self.width))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.cur_player]
            move_oppo = moves[players != self.cur_player]
            move_last = self.last_move
            full_state[0][move_curr // self.width, move_curr % self.width] = 1.0
            full_state[1][move_oppo // self.width, move_oppo % self.width] = 1.0
            full_state[2][move_last // self.width, move_last % self.width] = 1.0
        if len(self.states) % 2 == 0:
            full_state[3][:, :] = 1.0

        return full_state[:, ::-1, :]

    def move(self, index):
        self.states[index] = self.cur_player
        self.availables.remove(index)
        self.cur_player ^= 1
        self.last_move = index

    def is_win(self):
        for index, player in self.states.items():
            h, w = self.index_to_location(index)
            if w < self.width - self.k + 1 and \
                    len(set(self.states.get(self.location_to_index((h, w + i)), -1) for i in range(self.k))) == 1:
                return True
            if h < self.height - self.k + 1 and \
                    len(set(self.states.get(self.location_to_index((h + i, w)), -1) for i in range(self.k))) == 1:
                return True
            if w < self.width - self.k + 1 and h < self.height - self.k + 1 and \
                    len(set(self.states.get(self.location_to_index((h + i, w + i)), -1) for i in range(self.k))) == 1:
                return True
            if w >= self.k - 1 and h < self.height - self.k + 1 and \
                    len(set(self.states.get(self.location_to_index((h + i, w - i)), -1) for i in range(self.k))) == 1:
                return True

        return False

    def game_end(self):
        if self.is_win():
            return True, self.cur_player ^ 1
        if not self.availables:
            return True, -1

        return False, -1
