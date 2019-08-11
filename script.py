import sys
sys.path.append('/data')

import numpy as np
import requests
import random
import math
import json
from collections import defaultdict
from rl_model.architectures import SimpleConv
from rl_model.dqn import DQNModel, state_to_tensor
from rl_model.r import get_moves
from rule_utils.rule_based_model import choose_action as rulebased_choose_action

"""
{
requests:[
    {
        own: [0, 1, 2, 3, 4] // 自己最初拥有哪些牌
        history: [[0, 1, 2]/* 上上家 */, []/* 上家 */], // 总是两项，每一项都是数组，分别表示上上家和上家出的牌，空数组表示跳过回合或者还没轮到他。
        publiccard: [0, 1, 2, 3, 4] // 自己最初拥有哪些牌
    },
    {
        "history": [[0, 1, 2]/* 上上家 */, []/* 上家 */], // 总是两项，每一项都是数组，分别表示上上家和上家出的牌，空数组表示跳过回合。
    },
    {...}
]
}
"""

class GameState():
    def __init__(self):
        self.hand = None
        self.out = None
        self.up_out = None
        self.down_out = None
        self.self_out = None
        self.other_hand = None
        self.last_move = [0]*15 # 上一个有效出牌，全零表示主动权
        self.last_pid = -1 # 上一个有效出牌的玩家编号，-1表示主动权
        self.last_move_ = np.zeros(15, dtype=int)  # 上一个出牌，不管有效与否
        self.last_last_move_ = np.zeros(15, dtype=int) # 上上个出牌，不管有效与否
        self.player_role = -1  # 当前玩家编号，-1表示主动权
        self.left_cards = [0, 0, 0]  # 三家剩余牌数量

    def convert_to_payload(self):
        pass
        

def get_initial_info(r):
    # 判断自己是什么身份，地主0 or 农民甲1 or 农民乙2
    hand = r['own']
    pid = 0
    o = r['history']
    if len(o[0]) == 0:
        if len(o[1]) != 0:
            pid = 1
    else:
        pid = 2
    return pid, hand
    
def get_rank(cardno):
    if cardno==52:
        return 13
    if cardno==53:
        return 14
    return cardno//4

def cardno_to_list(cards):
    lst = [0]*15
    for c in cards:
        lst[get_rank(c)]+=1
    return lst

def choose_action(state):
    move_list = get_moves(state.hand, state.last_move)
    #move = model.choose_action(state_to_tensor(state), move_list, ignore_eps=True)
    move = rulebased_choose_action(state)
    return move


def choose_action_by_payload(payload):
    return None

full_input = json.loads(input())
requ = full_input["requests"]
resp = full_input['responses']

# restore
pid, hand = get_initial_info(requ[0])
history = []
for i in range(len(resp)):
    history.extend(requ[i]['history'])
    history.append(resp[i])
    for j in resp[i]:
        hand.remove(j)
if len(requ)==1:
    history.extend(requ[-1]['history'][2-pid:]) # 只有第一行有padding
else:
    history.extend(requ[-1]['history'])

last_move = []
last_pid = -1
if len(requ[-1]['history'][-1])>0:
    last_move = requ[-1]['history'][-1]
    last_pid = (pid-1)%3
elif len(requ[-1]['history'][-2])>0:
    last_move = requ[-1]['history'][-2]
    last_pid = (pid+1)%3

# prepare
#model = DQNModel((15, 4, 11), SimpleConv)
#if pid==0:
#    model.load(r"/data/rl_model/lord_selfvs_nornn_onlyweights.h5")
#else:
#    model.load(r"/data/rl_model/peas_selfvs_nornn_onlyweights.h5")
    
history = np.array(list(map(cardno_to_list, history)), dtype=int)
last_move = cardno_to_list(last_move)
hand_list = cardno_to_list(hand)
zeros = np.zeros(15, dtype=int)

state = GameState()
state.hand = hand_list
state.out = (zeros+history.sum(axis=0)).tolist()
state.up_out = (zeros+history[(pid-1)%3::3].sum(axis=0)).tolist()
state.down_out = (zeros+history[(pid+1)%3::3].sum(axis=0)).tolist()
state.self_out = (zeros+history[pid::3].sum(axis=0)).tolist()
state.last_move = last_move
state.other_hand = (np.array([4] * 13 + [1, 1], dtype=int) - state.hand - state.out).tolist()
state.last_pid = last_pid
#state.last_move_, state.last_last_move_ = history[-1], history[-2]
state.left_cards = [20-history[0::3].sum(), 17-history[1::3].sum(), 17-history[1::3].sum()]

# policy
move = choose_action(state)  # By GameState
#move = choose_action_by_payload(state.convert_to_payload)   # By Payload

# output
tmp = [[] for _ in range(15)]
for k in hand:
    tmp[get_rank(k)].append(k)
ans = []
for i in range(len(move)):
    if move[i]>0:
        ans.extend(tmp[i][:move[i]])
print(json.dumps({
        "response": ans
    }))

