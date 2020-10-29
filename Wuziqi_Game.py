
import random
import time
import copy
from Wuziqi_AI import MCTS

from Wuziqi_Human import Human
import matplotlib.pyplot as plt
import numpy as np

from Wuziqi_constant import has_a_winner

from Wuziqi_constant import board_hight
from Wuziqi_constant import board_width
x_dim1 = board_hight* board_width + 1  # 输入维度 (棋盘状态 + 当前落子的玩家)
y_dim1 = board_hight* board_width + 1  # 输入维度 (所有动作的可能性 + 胜率)


###########################################################################################################################################
# Game 类
# 控制游戏的进行，并在终端显示游戏的实时状态。
###########################################################################################################################################
class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board                                                               # 棋局
        self.player = [1, 2]                                                             # player1 and player2
        self.n_in_row = int(kwargs.get('n_in_row', 5))                                   # 最大连子数
        self.simu_time = float(kwargs.get('simu_time', 5))                               # 最大模拟时间
        self.simu_num = int(kwargs.get('simu_num', 1000))                                # 每次模拟最大的步数

        self.memory_size = 500000
        self.batch_size = 100
        self.memory_input = np.zeros((self.memory_size,x_dim1))                                    # NN输入样本
        self.memory_num = 0
        self.memory_output = np.zeros((self.memory_size,y_dim1))                                   # NN输出样本

        self.save_path = "./model"  #  数据存储路径

        self.memory_input = np.load(self.save_path + '/memory001.npy')                      # 载入NN输入样本
        self.memory_output = np.load(self.save_path + '/memory002.npy')                          # 载入NN输出样本
        self.memory_num = np.load(self.save_path + '/memory003.npy')
        #print(self.memory_num)

    def start(self,model,index,Is_learning):
        #p1, p2 = self.init_player()                                                      # 随机初始化玩家顺序
        p1 = self.player[0]
        p2 = self.player[1]

        if index==0:
            # 自我博弈开启   # 注意下面一行也要开启
            if model == '1':  # 自我博弈
                self.AI = MCTS(self.board, [p1, p2], self.n_in_row, self.simu_time, self.simu_num)  # AI1玩家
                self.human = self.AI
            elif model == '2':
                self.AI = MCTS(self.board, [p1, p2], self.n_in_row, self.simu_time, self.simu_num)  # AI1玩家
                self.human = Human(self.board, p2)  # 人类玩家
            else:
                self.AI = Human(self.board, p1)
                self.human = Human(self.board, p2)  # 人类玩家

        players = {}  # 2个玩家
        players[p1] = self.AI
        players[p2] = self.human
        turn = [p1, p2]
        random.shuffle(turn)                                                             # 玩家和电脑的出手顺序随机 shuffle用于随机排列

        self.board.init_board()                                                           # 初始化游戏棋局
        self.AI.update_board(self.board)                                                  # 两者都更新棋盘
        self.human.update_board(self.board)

        sample1_x = np.empty([0,x_dim1])
        sample1_y = np.empty([0,y_dim1])

        while (1):
            p = turn.pop(0)                                                              #  选择玩家
            turn.append(p)
            player_in_turn = players[p]                                                  # 当前下棋的玩家

            #自我博弈开启   更新当前玩家序号  因为不能用深拷贝  就用这种简单的办法
            if model == 1:
                player_in_turn.player = p
                pass

            move = player_in_turn.get_action()                                           # 获取动作

            x_state = np.zeros([1,x_dim1])           # 构造样本数据
            x_state[0,0:x_dim1-1] = self.board.states
            x_state[0,-1] = p
            sample1_x = np.r_[sample1_x,x_state]

            y_move = np.zeros([1,y_dim1])
            y_move[0,move] = 1
            y_move[0,-1] = 1
            sample1_y = np.r_[sample1_y,y_move]

            self.board.update(p, move)                                                   # 更新棋盘

            self.AI.update_board(self.board)                                                  # 两者都更新棋盘
            self.human.update_board(self.board)

            # self.graphic(self.board, human, AI)                                        # 输出棋盘
            self.plot_board(self.board, self.human, self.AI )
            end, winner = self.game_end()                                                # 判断游戏是否结束
            if end:                                                                      # 游戏结束

                xx = sample1_x[players[winner].player-1::2,:]
                yy = sample1_y[players[winner].player-1::2,:]
                len = np.shape(xx)[0]
                counter = self.memory_num % self.memory_size #求更新的位置
                self.memory_input[counter:counter+len,:] = xx
                self.memory_output[counter:counter+len,:] = yy
                self.memory_num += len

                np.save(self.save_path + '/memory001.npy', self.memory_input)
                np.save(self.save_path + '/memory002.npy', self.memory_output)
                np.save(self.save_path + '/memory003.npy', self.memory_num)

                if model=='1':   # AI对战
                    if Is_learning:
                        self.learningNN()
                    if winner != -1:  # 不是平局
                        print("Game end. Winner is", players[winner],players[winner].player)
                    else:
                        print("Game end. No Winner")
                    pass
                elif model=='2':        # 人机对战
                    if Is_learning:
                        self.learningNN()
                    if winner != -1:  # 不是平局
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. No Winner")
                    pass
                else:              # 人人对战
                    if winner != -1:  # 不是平局
                        print("Game end. Winner is", players[winner],players[winner].player)
                    else:
                        print("Game end. No Winner")
                    pass

                break  # 退出当前局

    def learningNN(self):
            print('开始训练!')
            if self.memory_num > self.memory_size:  # 随机采样
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_num, size=self.batch_size)
            batch_memory_x = self.memory_input[sample_index, :]  # 采样batch_size个样本
            batch_memory_y = self.memory_output[sample_index, :]  # 采样batch_size个样本
            self.AI.NN.learn(batch_memory_x,batch_memory_y)
            self.AI.NN.save()
            print('训练结束!')

    def init_player(self):                                                               # 初始化玩家
        plist = list(range(len(self.player)))                                            # 玩家的序号
        index1 = random.choice(plist)                                                    # 随机选择一个玩家
        plist.remove(index1)                                                             # 移除所选玩家
        index2 = random.choice(plist)                                                    # 在选择另一个玩家
        return self.player[index1], self.player[index2]

    def game_end(self):                                                              # 检查游戏是否结束
        win, winner = has_a_winner(self.n_in_row,self.board)                                        # 游戏结束
        if win:                                                                          # 游戏结束 返回胜利者
            return True, winner
        elif not len(self.board.availables):                                             # 游戏平局 没有可走的位置了 棋局已满了
            print("Game end. Tie")
            return True, -1
        return False, -1

    def graphic(self, board, human, ai):                                                 # 在终端绘制棋盘，显示棋局的状态
        width = board.width
        height = board.height

        print("Human Player", human.player, "with X".rjust(3))                           # rjust() 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串。如果指定的长度小于字符串的长度则返回原字符串
        print("AI    Player", ai.player, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                if board.states[loc] == self.player[0]:
                    print('X'.center(8), end='')
                elif board.states[loc] == self.player[1]:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')


    def plot_board(self, board, human, ai):
        width = board.width
        height = board.height
        #plt.clf()
        plt.figure(1)
        plt.cla()

        for i in range(height - 1, -1, -1):
            for j in range(width):
                loc = i * width + j
                if board.states[loc] == self.player[0] : #human.player
                    plt.scatter(j , i , s = 200 , c='r', marker='x')
                elif board.states[loc] == self.player[1]:  #ai.player
                    plt.scatter(j , i , s = 200 , c='black', marker='o')
                else:
                    pass

        # plt.axis('off')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.text(11,15.9, ("Human Player:  X"), color="r")
        plt.text(11,15.3, ("AI    Player:  O"), color="r")
        plt.xticks(np.arange(-1,width+1))
        plt.yticks(np.arange(-1,height+1))
        # plt.xticks([])
        # plt.yticks([])
        plt.grid(True)
        plt.title('AI VS Human')

        plt.ion()  # 打开交互模式
        plt.savefig('model/result.jpg')  # 保存图片

        plt.show()
        plt.pause(1)

        pass

