from Wuziqi_Board import  Board
from Wuziqi_Game import Game
import time


from Wuziqi_constant import board_hight
from Wuziqi_constant import board_width
from Wuziqi_constant import n_in_num
from Wuziqi_constant import simu_time
from Wuziqi_constant import simu_num

if __name__ == '__main__':
    print('棋盘大小为',board_width,'*',board_hight,'!')
    print('率先完成',n_in_num,'连子的一方获胜！')
    print('玩家下棋的方式eg： i,j  。其中: i的范围：0~',board_width,', j的范围：0~',board_hight,'\n')
    print('训练选项： 1：训练 2：不训练')

    print('模式选项： 1：AI对战 2：人机对战 3：人人对战\n')
    while (1):
        Islearning = input('请选择是否开始训练：')
        if Islearning == '1':
            Is_learning = True
            print('开启训练模式！\n')
            break
        elif Islearning == '2':
            Is_learning = False
            print('关闭训练模式!\n')
            break
        else:
            print('模式选择错误！ 请重新选择！')
            pass
    time.sleep(1)
    while(1):
        model = input('请输入模式选项：')
        if model == '1':
            print('当前模式为AI对战！')
            break
        elif model == '2':
            print('当前模式为人机对战！')
            break
        elif model == '3':
            print('当前模式为人人对战')
            Is_learning = False           # 模式3不可以训练
            break
        else:
            print('模式选择错误！ 请重新选择！')
            pass
        pass



    Wuziqi = Board(board_width,board_width,n_in_num)
    game = Game(Wuziqi , n_in_row = n_in_num , simu_time = simu_time , simu_num = simu_num)
    current_index = 0
    total_VS = 5
    for current_index in range(total_VS):
        print('\n第',current_index+1,'局开始！')
        game.start(model,current_index,Is_learning)
        time.sleep(5)