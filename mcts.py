import numpy as np

"""
An easy implementation based on
https://github.com/salimandre/Monte-Carlo-Tree-Search
https://github.com/int8/monte-carlo-tree-search
"""
class GameState:
    def __init__(self, state, visit_cnt, value, ucb_const) -> None:
        self.visit_cnt = visit_cnt
        self.value = value
        self.ucb_const = ucb_const
        self.state_info = state #定义一个游戏状态
        

class Node:
    def __init__(self, state, parent) -> None:
        self.game_state = state
        self.parent = parent
        self.children = {}

    def get_value(self):
        """
        Returns the ucb value of this node
        """
        visit_cnt = self.game_state.visit_cnt + 1
        parent_visit_cnt = self.parent.game_state.visit_cnt
        ucb_value = self.game_state.value/visit_cnt + self.game_state.ucb_const * np.sqrt(np.log(parent_visit_cnt)/visit_cnt)
        return ucb_value

    def select_best_child(self):
        """
        Select action among children that gives maximum ucb value
        Return: A tuple of (action, next_node)
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value())

    def is_leaf_node(self):
        """
        return True if this node is leaf
        """
        return self.children == {}

    def expand(self, action, state):
        """
        expand a leaf node
        """
        self.children.update(action, Node(state,self))

    def update(self, value):
        """
        update this node's values as well as it's ancestors'   
        """
        self.game_state.visit_cnt += 1
        self.game_state.value += value
        node = self
        while node.parent:
            node = node.parent
            node.update(value)

class MCTS:
    def __init__(self, root) -> None:
        self.root = root

    def traverse(self, state):
        node = self.root
        while not node.is_leaf_node:
            # select the best child
            action, node = node.select_best_child()
            state.step(action)

        # if node is leaf, expand the node
        # how to expand an action ????
        node.expand(action)
        
    def simulate(self, node, state):
        reward = 0
        while True:
            action = state.sample_action()
            state = state.step(action)
            # check winner
            winner, reward = state.get_winner()
            if winner:
                break
        node.update(reward)

