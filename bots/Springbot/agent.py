import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import ActionName, Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent, Interactive_TensorRTS
from datetime import datetime
import numpy as np
from collections import defaultdict
    
class New_Agent(Agent):
    def __init__(self, initial_observation: Observation, action_space: Dict[str, CategoricalActionSpace | SelectEntityActionSpace | GlobalCategoricalActionSpace]):
        super().__init__(initial_observation, action_space)
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
    def take_turn(self, current_game_State : Observation) -> Mapping[ActionName, Action]:
        MAX_TIME_PER_TURN = 300 # ms

        root = MonteCarloTreeSearchNode(state = current_game_State, act_space = self.action_space, MAX_TIME_PER_TURN=MAX_TIME_PER_TURN)
        selected_node = root.best_action().parent_action
        # action_choice = 1 if self.can_beat_opponent(current_game_State) else 0
        action_choice = selected_node
        print(action_choice)
        return map_move(action_choice[0], action_choice[1])

## Base implementation used is originaly from https://ai-boson.github.io/mcts/
## modified for use here.
class MonteCarloTreeSearchNode():
    def __init__(self, state, act_space, MAX_TIME_PER_TURN = 300, parent=None, parent_action=None):
        self.state = state
        self.action_space = act_space
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.MAX_ROLLOUT_TIME = 30
        self.simulations = int(MAX_TIME_PER_TURN/self.MAX_ROLLOUT_TIME)
        return
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions(self.state)
        return self._untried_actions
    
    def q(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses
    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        action = self._untried_actions.pop()
        next_state_game = Interactive_TensorRTS(enable_printouts=False)
        next_state_game.set_state(self.state)
        next_state = next_state_game.act(map_move(action[0], action[1]))
        child_node = MonteCarloTreeSearchNode(
            next_state, self.action_space, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
    
    def is_terminal_node(self):
        return self.state.done
    
    def rollout(self):

        current_rollout_state = self.state
        rollout_game = Interactive_TensorRTS(enable_printouts=False)
        rollout_game.set_state(current_rollout_state)
        rollout_start_time = datetime.now()
        delta = rollout_start_time - rollout_start_time
        while not rollout_game.is_game_over and (delta.total_seconds() * 1000) < self.MAX_ROLLOUT_TIME:
            
            possible_moves = self.get_legal_actions(current_rollout_state)
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = rollout_game.act(map_move(action[0], action[1]))

            ## move randomly as player two
            action = self.rollout_policy(possible_moves)
            current_rollout_state = rollout_game.act(map_move(action[0], action[1]), False, True)

            delta = datetime.now() - rollout_start_time
            # print(delta, action)

        return current_rollout_state.reward
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = self.simulations
        
        
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)

    def get_legal_actions(self, state): 
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        actionmask = state.actions["Move"].mask
        if actionmask is None:
            return list(enumerate(self.action_space["Move"].index_to_label))
        return list(filter(lambda x, i: actionmask[i] == True, enumerate(self.action_space["Move"].index_to_label)))
    
    def game_result(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return self.state.reward
        # return np.ceil(self.state.reward/10)
    
    def move(self,action):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        self.state.act()

def map_move(idx : int , label : str):
    action_map = {}
    action_map["Move"] = GlobalCategoricalAction(idx, label)
    return action_map

def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    # return Random_Agent(init_observation, action_space)
    return New_Agent(init_observation, action_space)

def student_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Silas Springer'
