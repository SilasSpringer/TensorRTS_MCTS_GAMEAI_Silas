import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent

class Random_Agent(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        action_choice = random.randrange(0, 2)
        if (action_choice == 1): 
            mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        else: 
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        
        return mapping
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    return Random_Agent(init_observation, action_space)

def student_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'My Name_2'
