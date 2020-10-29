###########################################################################################################################################
# Human 类
# 用于获取玩家的输入，作为落子位置。
###########################################################################################################################################
class Human(object):

    def __init__(self, board, player):
        self.board = board                                                                   # 棋盘状态
        self.player = player                                                                 # 玩家

    def get_action(self):
        try:
            print('轮到人类玩家',self.player,'落子',end='')
            location = [int(n, 10) for n in input("!     请落子: ").split(",")]                 # 人类玩家输入下棋位置
            #print('\n')
            move = self.board.location_to_move(location)                                     # 转化为索引
        except Exception as e:
            move = -1
        if move == -1 or move not in self.board.availables:                                  # 下棋位置 无效  或者 该处已经由落子  或者超出落子位置
            print("invalid move,please try again!!!")                                                            # 打印log
            move = self.get_action()                                                         # 迭代调用
        return move                                                                          # 返回索引

    def update_board(self,board):
        self.board = board

    def __str__(self):                                                                       # 如果输出该类的对象时候，就会 输出这个函数的返回值！！！
        return "Human"