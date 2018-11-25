# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import P_Generation as P_G
import numpy as np
import random
import sys

### update3.py の目的 ###

# いい感じにupdateされた Q-Matrixによる行動選択が
# RunGreedy にも反映されるようにする。

Time_Range     = 5000
LEARNING_COUNT = 1000    
P_Interval     = 50  # 行動選択してからReward出すまでのInterval
                     # このInterval間の平均のNum_of_packetを使用

unit_size  = 100
Buf_limit  = 2000
cong_thres = 1800  # 使わないけど便宜上書いておく
reward_arr = [20,40,60,80,100,120,140,160,180,200, \
             220,240,260,280,300,320,340,100,-300,-300]
             # Thres_Congは上記の報酬設定で自動規定される
             # マイナス報酬の数とcong_thresは完全に相関する感じ


### パケット生成 ###
P_1     = 3      # 1000[ms] = 1[sec]
P_0     = P_1 * 5
Amt_Avg = 3     # 1サイクルでの平均パケット生成量   7
Amt_IoT = 1     # 1サイクルでの平均パケット生成量   5
UE_Num  = 50

# 各々のUEで生成したパケットの足し合わせ
arr_sum   = list(np.zeros(Time_Range))

for i in range(UE_Num):
  if(i%4 != 0):  # Text, HD, 8K(ULでの生成パケット数は統一)
    ue_packet = P_G.UE_Poisson(int(Time_Range), Amt_Avg)
  else:          # IoTデバイスからの周期的なパケット生成
    ue_packet = P_G.IoT_Generate(int(Time_Range), P_0, P_1, Amt_IoT)

  # 配列の要素ごと足し合わせ
  arr_sum = [x + y for (x, y) in zip(arr_sum, ue_packet)]


GAMMA     = 0.8
val       = 5
bw_limit  = [80,90,100,110,120,130]  # Action の数
num_p_eNB = 0
p_process = 110   # eNBで単位時間あたりに処理するパケット数
eNB_arr   = []
reward_log= []

# Initial Q-value
# Q = np.zeros((Buf_limit,len(bw_limit)))
Q = np.zeros((int(Buf_limit/unit_size),len(bw_limit))) # 20行6列の行列

def initiate():
    global val, bw_limit, num_p_eNB, p_process, eNB_arr, Buf_limit
    val       = 5
    num_p_eNB = 0
    eNB_arr   = []


def file_arr(arr):
    file = open('Q_Matrix.txt', 'w')  # 書き込みモードでオープン
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
      global val,bw_limit,num_p_eNB,p_process,eNB_arr,Buf_limit
      global reward_log, P_Interval
      state = self._getRandomState()

      for j in range(LEARNING_COUNT):
        initiate()

        for i in range(Time_Range):

          if(i>P_Interval): val = action
          num_p_eNB += min(bw_limit[val],arr_sum[i]) - p_process
          if(num_p_eNB < 0):     num_p_eNB = 0
          if(num_p_eNB >= 2000): num_p_eNB = 1999
          eNB_arr.append(num_p_eNB)
          # Thres_cong = 1800 で、Buffer_linit = 2000だから必要

          if(i>1 and i%P_Interval == 0):      # Policy Interval
            if(i==P_Interval):
              reward   = 0
              action   = val
              prv_state= 0

            prv_state  = state
            prv_reward = reward
            prv_action = action

            avg_num_p  = sum(eNB_arr[-P_Interval:])/P_Interval
            state      = int((avg_num_p-1)/unit_size)  # state 0~20

            reward     = reward_arr[state]
            reward_log.append(reward)

            psb_actions= self._getPossibleActions(action)
            action     = random.choice(psb_actions)

            # if(j < 10 and i <= 300):
            print("************")
            print(prv_reward)
            print(prv_state)
            print(prv_action)
            print(reward)
            print(state)
            print(action)


            # 本来：self._updateQ(action, reward, state, next_state)
            # 一個前の情報を用いてQ値更新 ( next_stateを現行stateで代用 )
            self._updateQ(prv_action, prv_reward, prv_state, state)


    def _updateQ(self, action, reward, state, next_state):
      global Q
      next_psb_actions = self._getPossibleActions(action)
      max_Q_next_s_a   = self._getMaxQvalue(next_state, next_psb_actions)
      Q[int(state), action] = reward + GAMMA * max_Q_next_s_a


    def _getRandomState(self):
        return random.randint(0, (Buf_limit/unit_size)-1)

    def _getPossibleActions(self, val):
        if(val == 0):   return [0,1]
        elif(val == 5): return [4,5]
        else:     return [int(val)-1, int(val), int(val)+1]

    def _getMaxQvalue(self, state, possible_actions):
        return max([Q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        print(" *** Q-Matrix *** ")
        print(Q.astype(int)) # convert float to int for redability


    def runGreedy(self, start_action = 0):
        global val, bw_limit, num_p_eNB, p_process, eNB_arr, Buf_limit

        for i in range(Time_Range):

        # if(num_p_eNB <= Buf_limit):
          num_p_eNB += min(bw_limit[val],arr_sum[i]) - p_process
          if(num_p_eNB < 0):     num_p_eNB = 0
          if(num_p_eNB >= 2000): num_p_eNB = 1999
          eNB_arr.append(num_p_eNB)

          ##### Get best action which maximaizes Q(s, a) #####
          # Q[state, action] : 行動価値関数

          if(i>1 and i%P_Interval == 0):      # Policy Interval
            if(i==P_Interval):  action   = val


            avg_num_p = sum(eNB_arr[-P_Interval:])/P_Interval
            state = int((avg_num_p-1)/unit_size)

            possible_actions = self._getPossibleActions(action)

            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:
              a,state = int(a),int(state)
              if Q[state][a] > max_Q:
                best_action_candidates = [a,]
                max_Q = Q[state][a]
              elif Q[state][a] == max_Q:
                best_action_candidates.append(a)

            best_action = random.choice(best_action_candidates)

            # if(i<=P_Interval*6):
            print("*   *   *   *   *")
            print("current state    :" + str(state))
            print("current action   :" + str(action))
            print("possible actions :" + str(possible_actions))
            print("best actions     :" + str(best_action_candidates))

            action      = best_action    # これがないと循環しないだろw

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
    QL.runGreedy(start_val)  # 完成したQを基にTest
                             # Test結果の#num_p_eNB"を図示
    QL.dumpQvalue()          # Q-Tableの表示

    # plt.plot(reward_log[0     : 10000])
    # plt.show()
    # plt.plot(reward_log[-10000:-1])
    plt.plot(row_count(Q), lw=0.5)
    plt.title("Num Updated Q-Matrix(L-Count=%d)" % LEARNING_COUNT)
    plt.show()


