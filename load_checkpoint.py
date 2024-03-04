import random
from typing import Dict, List, Mapping, Tuple, Set

from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent

from TensorRTS import TensorRTS

if __name__ == "__main__": # This is to load a model
    # Change the checkpoint to your own folder name.
    checkpoint = load_checkpoint('checkpoints/latest-step000000008192')
    agent = RogueNetAgent(checkpoint.state.agent)
    env = TensorRTS()
    obs = env.reset()
    action, predicted_return = agent.act(obs)    
    print(f"ACTION: {action}")
    print(f"Predicted_return: {predicted_return}")