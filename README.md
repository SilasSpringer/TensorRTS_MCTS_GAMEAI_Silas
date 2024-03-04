# Example implementation of TensorRTS game from drchangliu's [starter code](https://github.com/drchangliu/RL4SE/tree/main/enn/TensorRTS)

Utilizes one additional package from the original demo implementation, attainable using
```
pip install mcts
```

Adds an MCTS bot

Adds funcitonality described in PA4 with specific rates of conversion for booming andrushing, and additional benefits for attacking tensors.
Adds `set_state()` function to TensorRTS environment to allow easy simulation.
