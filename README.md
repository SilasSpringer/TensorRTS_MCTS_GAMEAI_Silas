# Example implementation of TensorRTS game from drchangliu's [starter code](https://github.com/drchangliu/RL4SE/tree/main/enn/TensorRTS)

Utilizes one additional package from the original demo implementation, attainable using
```
pip install mcts
```

Adds an MCTS bot

Adds funcitonality described in PA4 with specific rates of conversion for booming andrushing, and additional benefits for attacking tensors.
Adds `set_state()` function to TensorRTS environment to allow easy simulation.

Note that occasional failures due to an issue in the mcts library may occur due to it trying to get the best child of the root node while it has no children. A temporary fix could be to add 
```
        if len(self.root.children) == 0:
            return random.choice(self.root.state.action_space)
```
to line 59 (just before the line `bestChild = self.getBestChild(self.root, 0)`)
in `/usr/local/python/3.10.13/lib/python3.10/site-packages/mcts.py` (your path will differ)