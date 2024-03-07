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

    
from mcts import mcts

class New_Agent(Agent):
    def __init__(self, initial_observation: Observation, action_space: Dict[str, CategoricalActionSpace | SelectEntityActionSpace | GlobalCategoricalActionSpace]):
        super().__init__(initial_observation, action_space)
    
    # def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
    #     return super().on_game_start(is_player_one, is_player_two)
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
    def take_turn(self, current_game_State : Observation) -> Mapping[ActionName, Action]:
        # tree = mcts(timeLimit=1000)
        tree = mcts(iterationLimit=1000)
        # action_choice = 1
        action_choice = tree.search(initialState=MCTS_State(current_game_State, list(enumerate(self.action_space["Move"].index_to_label)) ))
        # print(action_choice)
        return map_move(action_choice[0], action_choice[1])

class MCTS_State(Observation):
    def __init__(self, state: Observation, action_space:  list):
        self.state = state
        self.action_space = action_space
    def __copy__(self):
        return MCTS_State(self.state.copy(), self.action_space)
    def __deepcopy__(self, memo):
        return MCTS_State(self.state.deepcopy(), self.action_space)
    def getPossibleActions(self):
        actionmask = self.state.actions["Move"].mask
        if actionmask is None:
            return self.action_space
        return list(filter(lambda x, i: actionmask[i] == True, enumerate(self.action_space)))
    def takeAction(self, action):
        _game = Interactive_TensorRTS(enable_printouts=False)
        _game.set_state(self.state)
        _game.act(map_move(action[0], action[1]))
        opp_rand_move = random.choice(self.action_space)
        _game.act(map_move(opp_rand_move[0], opp_rand_move[1]))
        return MCTS_State(_game.act(map_move(action[0], action[1])), self.action_space)
    def isTerminal(self):
        return self.state.done
    def getReward(self):
        return self.state.reward


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


