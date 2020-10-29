###########################################################################################################################################
# 代码实现 Board 类
# Board类用于存储当前棋盘的状态，它实际上也是MCTS算法的根节点。
###########################################################################################################################################
import numpy as np
class Board(object):

    def __init__(self, width = 15, height = 15, n_in_row=5):
        self.width = width                                                                     # 棋盘宽度
        self.height = height                                                                   # 棋盘高度
        self.states =  np.zeros(self.width*self.height)                                                                       # 记录当前棋盘的状态，键是位置，值是棋子，这里用玩家来表示棋子类型
        self.n_in_row = n_in_row                                                               # 表示几个相同的棋子连成一线算作胜利
        pass

    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)     # 棋盘不能过小

        self.availables = list(range(self.width * self.height))                                # 表示棋盘上所有合法的位置，这里简单的认为空的位置即合法
        for m in self.availables:
            self.states[m] = -1                                                                # -1表示当前位置为空
            pass
        pass

    def move_to_location(self, move):                                                          # 由位置索引计算位置
        h = move // self.width
        w = move % self.width
        return [w,h]                                                                          # 位置是一个list

    def location_to_move(self, location):                                                      # 由位置计算位置索引
        if (len(location) != 2):                                                               # list应该包括 宽度和高度
            return -1
        w = location[0]
        h = location[1]
        move = h * self.width + w                                                              # 计算索引
        if (move not in range(self.width * self.height)):                                      # 超出给定的位置
            return -1
        if h>self.height-1:
            return -1
        elif w>self.width-1:
            return -1
        return move

    def update(self, player, move):                                                            # player在move处落子，更新棋盘
        self.states[move] = player                                                             # 更新棋盘
        self.availables.remove(move)                                                           # 从可用动作移除
        pass

    pass







