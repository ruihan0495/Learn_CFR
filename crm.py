# A demo for counterfactual regret minimization applied on Rock, Scissor and Papers

############## GAME CONSTANTS #######################
from random import random
import numpy as np


ROCK = 0
SCISSORS = 1
PAPER = 2

NUM_ACTIONS = 3

other_strategy = [1/3, 1/3, 1/3]

GAME_UTILITY = np.array([[0, 1, -1],[-1, 0, 1],[1, -1, 0]])

############## CRM IMPLEMENTATIONS ####################
regret_sum = np.zeros(NUM_ACTIONS)
strategy = np.zeros(NUM_ACTIONS)
strategy_sum = np.zeros(NUM_ACTIONS)


class CRM:

    def __init__(self) -> None:
        self.regret_sum = regret_sum
        self.strategy = strategy
        self.strategy_sum = strategy_sum

    def get_strategy(self):
        for action in range(NUM_ACTIONS):
            self.strategy[action] = self.regret_sum[action] if self.regret_sum[action] > 0 else 0
        total_regrets = np.sum(self.strategy)     
        if total_regrets > 0:
            for action in range(NUM_ACTIONS):
                self.strategy[action] /= total_regrets
        else:
            self.strategy = [1/NUM_ACTIONS] * NUM_ACTIONS
        self.strategy_sum += self.strategy
        return self.strategy 

    def get_action(self):
        return np.random.choice([ROCK, SCISSORS, PAPER], p=self.strategy)

    def get_avg_strategy(self):
        strategy = np.zeros(NUM_ACTIONS)
        for action in range(NUM_ACTIONS):
            strategy[action] = self.strategy_sum[action] if self.strategy_sum[action] > 0 else 0
        total_regrets = np.sum(strategy_sum)
        if total_regrets > 0:
            for action in range(NUM_ACTIONS):
                strategy[action] /= total_regrets 
        else:
            strategy = [1/NUM_ACTIONS] * NUM_ACTIONS
        return strategy

    def train(self, iterations, other_strategy=other_strategy):
        for i in range(iterations):
            other_action = np.random.choice([ROCK, SCISSORS, PAPER], p=other_strategy)

            # get regret matched mixed-strategy actions
            strategy = self.get_strategy()
            action = self.get_action()

            # compute action utilities
            utility = GAME_UTILITY[action,:]

            # accumulate action regrets
            self.regret_sum += utility - utility[other_action]
            print(self.regret_sum)
        print(self.get_avg_strategy())


def train(iterations, me, other):
    my_utility = np.zeros(NUM_ACTIONS)
    other_utility = np.zeros(NUM_ACTIONS)
    for _ in range(iterations):
        # get regret matched mixed-strategy actions
        my_strategy = me.get_strategy()
        my_action = me.get_action()
        print(my_strategy, my_action)
        other_strategy = other.get_strategy()
        other_action = other.get_action()
        print(other_strategy, other_action)
        # compute action utilities
        my_utility = GAME_UTILITY[my_action,:]
        other_utility = GAME_UTILITY[other_action,:]
        # print("utilities", my_utility, other_utility)
        # accumulate action regrets
        me.regret_sum += my_utility - my_utility[other_action]
        other.regret_sum += other_utility - other_utility[my_action]
        # print("regret sum", me.regret_sum, other.regret_sum)
    print(me.get_avg_strategy(), other.get_avg_strategy())
    return me.get_avg_strategy(), other.get_avg_strategy()


if __name__ == "__main__":
    me = CRM()
    opponent = CRM()
    # me.train(10000)
    train(1000, me, opponent)

