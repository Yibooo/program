# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import P_Generation as P_G
import numpy as np
import random
import sys

### simple8.py の目的 ###

# sinple7 では100単位でまとめてみたけど、
# なんか良い感じにアプデされなかったンゴ。
# その理由を探すのと、改善を試みる！！！


LEARNING_COUNT = 1000    
GAMMA          = 0.8
GOAL_STATE     = 5

### パケット生成 ###

UE_Num = 50

P_1 = 3      # 1000[ms] = 1[sec]
P_0 = P_1 * 5

Amt_Avg = 3    # 1サイクルでの平均パケット生成量   7
Amt_IoT = 1    # 1サイクルでの平均パケット生成量   5

Time_Range = 2000

# 各々のUEで生成したパケットの足し合わせ
arr_sum   = list(np.zeros(Time_Range))

for i in range(UE_Num):
  if(i%4 != 0):  # Text, HD, 8K(ULでの生成パケット数は統一)
    ue_packet = P_G.UE_Poisson(int(Time_Range), Amt_Avg)
  else:          # IoTデバイスからの周期的なパケット生成
    ue_packet = P_G.IoT_Generate(int(Time_Range), P_0, P_1, Amt_IoT)

  arr_sum = [x + y for (x, y) in zip(arr_sum, ue_packet)]  # 配列の要素ごと足し合わせ


### eNBに送る（ 帯域幅の制限 ）###
### eNBで等間隔で一定数のパケットを処理する ###

val       = 5
bw_limit  = [80,90,100,110,120,130]  # Action の数
num_p_eNB = 0
p_process = 110   # eNBで単位時間あたりに処理するパケット数
eNB_arr   = []
reward_log= []
rw_range  = 100   # 行動選択してからReward出すまでのRange
                  # このRange間の平均のNum_of_packetを使用

unit_size = 100
Buf_limit = 2000  # State の状態数
cong_thres= 1800  # Congestion Threshold
reward_arr= [20,40,60,80,100,120,140,160,180,200, \
             220,240,260,280,300,320,340,360,-200,-200]


# Initial Q-value
# Q = np.zeros((Buf_limit,len(bw_limit)))
Q = np.zeros((int(Buf_limit/unit_size),len(bw_limit))) # 20行 6列 行列

def initiate():
    global val, bw_limit, num_p_eNB, p_process, eNB_arr, Buf_limit
    val       = 5
    num_p_eNB = 0
    eNB_arr   = []


def file_arr(arr):
    file = open('Q_Matrix.txt', 'w')     # 書き込みモードでオープン
    for i in range(len(Q)): file.write(str(arr[i]) + "\n")
    file.close()


def row_count(arr):
  num = []
  for i in range(len(arr)):
    count = 0
    for j in range(len(arr[i])):
      if(arr[i][j] != 0): count += 1
    num.append(count)
  return num


class QLearning(object):
    def __init__(self):
      return

    ###  学習フェーズ  ###
    def learn(self):
      global val,bw_limit,num_p_eNB,p_process,eNB_arr,Buf_limit,cong_thres
      global reward_log, rw_range
      state = self._getRandomState()     # 最初のStateをランダムに決定

      for j in range(LEARNING_COUNT):    # ここのCountどうしよ
        initiate()
        for i in range(len(arr_sum)):
          reward = 0

        # if(num_p_eNB <= Buf_limit):
          if(i>50): val = action     # val,action,stateを整理すべし

          num_p_eNB += min(bw_limit[val],arr_sum[i]) - p_process
          if(num_p_eNB < 0):     num_p_eNB = 0
          # Thres_cong = 1800 で、Buffer_linit = 2000だから必要
          if(num_p_eNB >= 2000): num_p_eNB = 1999
          eNB_arr.append(num_p_eNB)


          if(i>1 and i%50 == 0):      # Policy Interval
            if(i==50):  action = val

            avg_num_p = sum(eNB_arr[-rw_range:])/rw_range
            state = int((avg_num_p-1)/unit_size)  # これで合ってるか不安
                                                  # stateは 0~20
            reward = reward_arr[state]
            reward_log.append(reward)

            possible_actions = self._getPossibleActions(action)
            action           = random.choice(possible_actions)
            # Update Q-value
            # Q(s,a) = r(s,a) + Gamma * max[Q(next_s, possible_actions)]
            next_possible_actions = self._getPossibleActions(action)
            # max_Q_next_s_aの第一引数がactionなのはオカシイ     
            # 関数の定義されてる側でQ[state,action]ってなってるのに     
            # これに対する代替案が必要 → 一回整理しないとな。。。
            max_Q_next_s_a        = self._getMaxQvalue(action, next_possible_actions)
            Q[int(state), action] = reward + GAMMA * max_Q_next_s_a

            if(j < 10 and i <= 300):
              print("************")
              print("Val = " + str(val))
              print(reward)
              print(max_Q_next_s_a)
              print(type(max_Q_next_s_a))
              print(state)
              print(action)
              print(Q[int(state), action])
              print("************")


        # ゴールしたら、またランダムな場所からスタート
        if state == GOAL_STATE:  state = self._getRandomState()

    def _getRandomState(self):
        return random.randint(0, Buf_limit)

    def _getPossibleActions(self, val):
        if(val == 0):
          return [0,1]
        elif(val == 5):
          return [4,5]
        else:
          return [int(val)-1, int(val), int(val)+1]

    def _getMaxQvalue(self, state, possible_actions):
        return max([Q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        print(Q.astype(int)) # convert float to int for redability

    def runGreedy(self, start_action = 0):
        global val, bw_limit, num_p_eNB, p_process, eNB_arr, Buf_limit
        action = start_action

        for i in range(Time_Range):

        # if(num_p_eNB <= Buf_limit):
          num_p_eNB += min(bw_limit[val],arr_sum[i]) - p_process     
          if(num_p_eNB < 0):     num_p_eNB = 0
          if(num_p_eNB >= 2000): num_p_eNB = 1999
          eNB_arr.append(num_p_eNB)



          ############### 報酬が最大となる行動を選択 ###############
          # get best action which maximaizes Q-value(s, a)
          # Q[state, action] : 行動価値関数
          # 現状態(s)から次状態(a)を選択し,最適行動を取り続けた場合の割引累積報酬の期待値
          # 現状態(s)を固定で、行動(a=Action)を色々とった時のQの要素で一番大きい奴を抽出

          if(i>1 and i%50 == 0):      # Policy Interval
            if(i==50):  action = val

            avg_num_p = sum(eNB_arr[-rw_range:])/rw_range
            state = int((avg_num_p-1)/unit_size)

            possible_actions = self._getPossibleActions(action)

            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:
              a,state = int(a),int(state)
              if Q[state][a] > max_Q:     # 次の状態への報酬が最も高い行動を選択
                best_action_candidates = [a,]
                max_Q = Q[state][a]
              elif Q[state][a] == max_Q:  # 同じ報酬の値の行動があったらそれを選択
                best_action_candidates.append(a)

            # get a best action from candidates randomly
            best_action = random.choice(best_action_candidates)  # 報酬の高い次の状態から選択
            # print("-> choose action: %d" % best_action)

        plt.plot(eNB_arr)
        plt.title("Num eNB packet(L-Count=%d)" % LEARNING_COUNT)
        plt.show()


if __name__ == "__main__":
    #####  Learning Phase  #####
    QL = QLearning()   # オブジェクトの生成
    QL.learn()         # 学習フェーズ
    file_arr(Q)

    #####  Test Phase  #####
    start_val = 5
    initiate()
    QL.dumpQvalue()          # Q-Tableの表示
    QL.runGreedy(start_val)  # 完成したQを基にTest
                             # Test結果の#num_p_eNB"を図示

    # plt.plot(reward_log[0     : 10000])
    # plt.show()
    # plt.plot(reward_log[-10000:-1])
    plt.plot(row_count(Q), lw=0.5)
    plt.title("Num Updated Q-Matrix(L-Count=%d)" % LEARNING_COUNT)
    plt.show()


