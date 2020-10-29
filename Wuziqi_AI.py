###########################################################################################################################################
# 蒙特卡洛树搜索（MCTS）仅展开根据 UCB 公式所计算过的节点，并且会采用一种自动的方式对性能指标好的节点进行更多的搜索。具体步骤概括如下：
# 1.由当前局面建立根节点，生成根节点的全部子节点，分别进行模拟对局；
# 2.从根节点开始，进行最佳优先搜索；
# 3.利用 UCB 公式计算每个子节点的 UCB 值，选择最大值的子节点；
# 4.若此节点不是叶节点，则以此节点作为根节点，重复 2；
# 5.直到遇到叶节点，如果叶节点未曾经被模拟对局过，对这个叶节点模拟对局；否则为这个叶节点随机生成子节点，并进行模拟对局；
# 6.将模拟对局的收益（一般胜为 1 负为 0）按对应颜色更新该节点及各级祖先节点，同时增加该节点以上所有节点的访问次数；
# 7.回到 2，除非此轮搜索时间结束或者达到预设循环次数；
# 8.从当前局面的子节点中挑选平均收益最高的给出最佳着法。
# 由此可见 UCT 算法就是在设定的时间内不断完成从根节点按照 UCB 的指引最终走到某一个叶节点的过程。而算法的基本流程包括了选择好的分支（Selection）、
# 在叶子节点上扩展一层（Expansion）、模拟对局（Simulation）和结果回馈（Backpropagation）这样四个部分。
# UCT 树搜索还有一个显著优点就是可以随时结束搜索并返回结果，在每一时刻，对 UCT 树来说都有一个相对最优的结果。
###########################################################################################################################################

import random
import time
import copy
import math
import numpy as np

from BPnet import BPnet

###########################################################################################################################################
# MCTS 类
# 核心类，用于实现基于UCB的MCTS算法。
###########################################################################################################################################

from Wuziqi_constant import board_hight
from Wuziqi_constant import board_width
from Wuziqi_constant import has_a_winner

s_dim = board_hight* board_width                                                              # 棋局状态维度
a_dim = board_hight* board_width                                                              # 棋局动作维度

class Node(object):
    def __init__(self, board,N1,W1,Q1,P1):
        self.board = board
        self.N = N1                                                                           # 每个动作的计数
        self.W = W1                                                                           # 每个动作的价值总和
        self.Q = Q1                                                                           # 每个动作的平均值
        self.P = P1                                                                           # 每个动作的概率
        self.parents = None                                                                   # 父节点
        pass

    def update_value(self,v,at):       # 这个at表示在当前状态下 执行了第几个动作！
        self.N[at] += 1
        self.W += v
        self.Q = self.W/self.N
        pass


class MCTS(object):
    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=1000):
        self.board = board                                                                    # 初始化棋盘
        self.play_turn = play_turn                                                            # 出手顺序 [list包括两个玩家]
        self.calculation_time = float(time)                                                   # 最大运算时间
        self.max_actions = max_actions                                                        # 每次模拟对局最多进行的步数
        self.n_in_row = n_in_row                                                              # 胜利时候的连子数

        self.save_path = "./model"  #  数据存储路径


        self.player = play_turn[0]                                                            # 轮到电脑出手，所以出手顺序中第一个总是电脑
        self.confident = 1.96                                                                 # UCB中的常数
        # self.plays = {}                                                                     # 记录着法参与模拟的次数，键形如(player, move)，即（玩家，落子）
        # self.wins = {}                                                                      # 记录着法获胜的次数
        self.nodes = []
        #self.nodes = np.load(self.save_path + '/memory004.npy',allow_pickle=True)
        # 载入Nodes

        #self.max_depth = 1

        self.tao = 0.1

        self.NN = BPnet()

        self.NN.restore()

    def update_board(self,board):
        self.board = board

    def get_action(self):                                                                     # return move  利用模拟 返回动作

        if len(self.board.availables) == 1:
            return self.board.availables[0]                                                   # 棋盘只剩最后一个落子位置，直接返回

        simulations = 0                                                                       # 仿真次数
        begin = time.time()                                                                   # 开始时间
        while time.time() - begin < self.calculation_time:                                    # 仿真时间小于
            board_copy = copy.deepcopy(self.board)                                            # 模拟会修改board的参数，所以必须进行深拷贝，与原board进行隔离
            play_turn_copy = copy.deepcopy(self.play_turn)                                    # 每次模拟都必须按照固定的顺序进行，所以进行深拷贝防止顺序被修改
            self.run_simulation(board_copy, play_turn_copy)                                   # 进行MCTS
            simulations += 1                                                                  # 仿真次数+1

        print("\ntotal simulations=", simulations)                                            # 仿真次数

        move = self.select_best_one_move(self.board)                                                    # 选择此次最佳着法
        location = self.board.move_to_location(move)                                          # 计算位置
        print("AI move: %d,%d\n" % (location[0], location[1]))
        return move

    def run_simulation(self, board, play_turn):

        # plays = self.plays                                                                   # 玩家的下棋位置统计
        # wins = self.wins                                                                     # 胜利次数统计

        init_state= copy.deepcopy(board.states)

        action_list = []            # 记录每次仿真从根结点开始 的动作 序列

        player = self.get_player(play_turn)                                                  # 获取当前出手的玩家
        winner = -1                                                                          # 胜者
        # Is_expand = True
        last_node = None                                                                      # 上一时刻的Node
        # Simulation
        for t in range(1, self.max_actions + 1):                                             # 每次模拟的步数

            availables = board.availables  # 可用动作

            current_node = None                       # 找到当前状态所对应的结点
            for node in self.nodes:
                if (np.array(board.states)==np.array(node.board.states)).all():
                    current_node = copy.deepcopy(node)
                    break
                    pass
                pass

            if current_node != None:                                               # 当前状态已经扩展  直接根据UCB选择动作
                # print('当前状态已经扩展  直接根据UCB选择动作')
                last_node = current_node
                U = self.confident*current_node.P*sum(current_node.P)/(1+current_node.N)
                QU = current_node.Q+U
                availables_QU = [ QU[index] for index in availables]               # 仅从可用动作中选取
                at_index = np.argmax(availables_QU)
                move = availables[at_index]
                action_list.append(move)
                pass
            else:
                # 当前状态并未扩展，根据NN选择动作与v 执行 回传 并结束此次模拟 进入下一次模拟
                # print('扩展结点')
                sample = np.zeros([1,s_dim+1])
                sample[0,0:s_dim] = board.states[:]
                sample[0,-1] = player

                pv = self.NN.predict(sample)
                p = pv[0,0:a_dim]
                v = pv[0,-1]
                pp = [p[index] for index in availables]
                # print(np.argmax(pp))

                action_list.append(np.argmax(pp))

                all0 = np.zeros(a_dim)
                if board != None:
                    newnode = Node(board,np.ones(a_dim),all0,all0,p)
                else:
                    break
                newnode.board = board
                newnode.parents = last_node
                self.nodes.append(newnode)
                # print('利用NN回传')
                self.back_propagation(newnode,action_list,v,init_state)

                #np.save(self.save_path + '/memory004.npy', self.nodes)

                break     # 退出此次模拟

            board.update(player, move)

            is_full = not len(availables)        # 是否所有的动作都已经走完
            win, winner = has_a_winner(self.n_in_row,board)   # 检查是否产生胜者
            if is_full or win:  # 游戏结束，没有落子位置或有玩家获胜
                #  回传  胜者为1 负者为 -1
                # print('分出胜负回传')
                # print(current_node)
                if current_node != None:
                    self.back_propagation(current_node,action_list,1,init_state)
                #np.save(self.save_path + '/memory004.npy', self.nodes)

                break

            player = self.get_player(play_turn)  # 更换选手


    def back_propagation(self,current_node,move,v,init_state):
        leng = len(move)
        index = -1  # 倒序
        while(1):
            if leng==0:
                break
            if current_node==None:
                break
            if (np.array(current_node.board.states) == np.array(init_state)).all():  #这不是根节点
                at = move[index]
                current_node.N[at] += 1
                current_node.W[at] += v
                v = v * (-1)                                        # 该节点为胜者，上节点必为败者
                index = index - 1
                current_node.Q = current_node.W / (current_node.N)  # 加1 为了 防止0/0  而且可以增大 动作的搜索
                for tt, node in enumerate(self.nodes):
                    if (np.array(current_node.board.states) == np.array(node.board.states)).all():
                        self.nodes[tt] = current_node
                        break
                #current_node = current_node.parents
                break

            at = move[index]
            current_node.N[at] += 1
            current_node.W[at] += v
            v = v*(-1)       # 该节点为胜者，上节点必为败者
            index = index -1
            current_node.Q = current_node.W/(current_node.N)   # 加1 为了 防止0/0  而且可以增大 动作的搜索
            for tt,node in enumerate(self.nodes):
                current_node_state = current_node.board.states
                node_state = node.board.states
                if (np.array(current_node_state)==np.array(node_state)).all():
                    self.nodes[tt] = current_node
                    break
            current_node = current_node.parents
            if current_node==None:
                break
            for node in self.nodes:
                if (np.array(current_node.board.states)==np.array(node.board.states)).all():
                    current_node = node
                    break
            leng = leng-1
        pass


    def get_player(self, players):                                     # 利用堆栈 轮流选择下棋
        p = players.pop(0)
        players.append(p)
        return p

    def select_best_one_move(self,board):

        current_node = None                                                     # 找到当前状态所对应的结点
        for node in self.nodes:
            if (np.array(board.states) == np.array(node.board.states)).all():
                current_node = node
                break
                pass
            pass
        ntao = current_node.N**(1/self.tao)
        pi = ntao/np.sum(ntao)
        while(1):
            move = np.random.choice(range(a_dim), p= pi)
            if move in board.availables:                                        # 保证动作可行
                return move

    def __str__(self):                                            # 如果输出该类的对象时候，就会 输出这个函数的返回值！！！
        return "AI"

    pass