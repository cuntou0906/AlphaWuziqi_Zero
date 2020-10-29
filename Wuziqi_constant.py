board_width = 15                                  # 棋盘宽度
board_hight = 15                                  # 棋盘高度
n_in_num = 5                                      # 最大连子数
simu_time = 10                                     # 每次AI决策的时间
simu_num = 1000                                   # 每次Mcts搜索的步数


def has_a_winner(n_in_row,board):      # 检查是否有玩家获胜

    moved = list(set(range(board.width * board.height)) - set(board.availables))  # 已下的棋子
    if (len(moved) < n_in_row * 2 - 1):  # 已下的棋子 小于  最大连子数的两倍-1 那么一定没有 获胜方
        return False, -1

    width = board.width  # 棋盘宽度
    height = board.height  # 棋盘高度
    states = board.states  # 棋盘状态
    n = n_in_row  # 最大连子数
    for m in moved:
        h = m // width
        w = m % width
        player = states[m]  # 当前棋子所属的玩家

        # (w in range(width - n + 1) 保证 棋子不太靠右，足够容纳self.n_in_row 个棋子
        if (w in range(width - n + 1) and
                len(set(states[i] for i in range(m, m + n))) == 1):  # 横向连成一线
            return True, player

        if (h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * width, width))) == 1):  # 竖向连成一线
            return True, player

        if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * (width + 1), width + 1))) == 1):  # 右斜向上连成一线
            return True, player

        if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states[i] for i in range(m, m + n * (width - 1), width - 1))) == 1):  # 左斜向下连成一线
            return True, player

    return False, -1