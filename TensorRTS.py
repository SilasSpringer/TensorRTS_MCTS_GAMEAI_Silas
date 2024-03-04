import random
import abc
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation

from entity_gym.runner import CliRunner
from entity_gym.env import *

class TensorRTS(Environment):
    """
LinearRTS, the first epoch of TensorRTS, is intended to be the simplest RTS game.
    """

    def __init__(
        self,
        mapsize: int = 64,
        attackerspeedadvantage: int = 2,
        attackeradvantagefactor: float = 1.8,
        econtomilitaryconvrate: float = 0.2,
        econboomrate: float = 0.2,
        nclusters: int = 6,
        ntensors: int = 2,
        maxdots: int = 9,
        enable_prinouts : bool = True
    ):
        self.enable_printouts = enable_prinouts
        
        if self.enable_printouts:
            print(f"LinearRTS -- Mapsize: {mapsize}")

        self.mapsize = mapsize
        self.maxdots = maxdots
        self.nclusters = nclusters

        self.attackerspeedadvantage = attackerspeedadvantage
        self.attackeradvantagefactor = attackeradvantagefactor
        self.econtomilitaryconvrate = econtomilitaryconvrate
        self.econboomrate = econboomrate

        self.clusters: List[List[int]] = []  # The inner list has a size of 2 (position, number of dots).
        self.tensors: List[List[int ]] = [] # The inner list has a size of 4 (position, dimension, x, y).

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "Cluster": Entity(features=["position", "dot"]),
                "Tensor": Entity(features=["position", "dimension", "x", "y"]),
            }
        )

    def action_space(cls) -> Dict[ActionName, ActionSpace]:
        return {
            "Move": GlobalCategoricalActionSpace(
                ["advance", "retreat", "rush", "boom"],
            ),
        }

    def reset(self) -> Observation:
        positions = set()
        while len(positions) < self.nclusters // 2:
            position, b = random.choice(
                [[position, b] for position in range(self.mapsize // 2) for b in range(1, self.maxdots)]
            )
            if position not in positions:
                positions.add(position)
                self.clusters.append([position, b])
                self.clusters.append([self.mapsize - position - 1, b])
        self.clusters.sort()
 
        position = random.randint(0, self.mapsize // 2)
        self.tensors = [[position, 1, 2, 0], [self.mapsize - position - 1, 1, 2, 0]]

        if self.enable_printouts:
            self.print_universe()

        return self.observe()
    
    def set_state(self, state : Observation):
        self.done = state.done
        self.reward = state.reward
        self.clusters = state.features["Cluster"]
        self.tensors = state.features["Tensor"]
        # self.print_universe()

    def tensor_power(self, tensor_index) -> float :
        f = self.tensors[tensor_index][3] * self.tensors[tensor_index][3] +  self.tensors[tensor_index][2] * \
            self.attackeradvantagefactor * (1 + abs((self.tensors[tensor_index][0] - 0.5*self.mapsize)/self.mapsize))

        f = self.tensors[tensor_index][3] * self.tensors[tensor_index][3] +  self.tensors[tensor_index][2]
        if self.enable_printouts:
            print(f"TP({tensor_index})=TP({self.tensors[tensor_index]})={f}")
        return f

    def observe(self) -> Observation:
        done = self.tensors[0][0] >= self.tensors[1][0]
        if done:
            reward = 10 if self.tensor_power(0) > self.tensor_power(1) else 0 if self.tensor_power(0) == self.tensor_power(1) else -10
        else:
            # reward = 1.0 if self.tensors[0][1] > self.tensors[1][1] else 0.0
            # reward += 0.1*self.tensors[0][0] # reward for going towards opponent
            reward = 0
        return Observation(
            entities={
                "Cluster": (
                    self.clusters,
                    [("Cluster", i) for i in range(len(self.clusters))],
                ),
                "Tensor": (
                    self.tensors,
                    [("Tensor", i) for i in range(len(self.tensors))],
                ),
            },
            actions={
                "Move": GlobalCategoricalActionMask(),
            },
            done=done,
            reward=reward,
        )

    def act(self, actions: Mapping[ActionName, Action], trigger_default_opponent_action : bool = True, is_player_two : bool = False) -> Observation:
        action = actions["Move"]
        player_tensor = self.tensors[0]
        direction = 1
        if is_player_two:
            player_tensor = self.tensors[1]
            direction = -1

        assert isinstance(action, GlobalCategoricalAction)
        if action.label == "advance" and self.tensors[0][0] < self.mapsize:
            player_tensor[0] += self.attackerspeedadvantage*direction
            player_tensor[2] += self.collect_dots(player_tensor[0])
        elif action.label == "retreat" and player_tensor[0] > 0:
            player_tensor[0] -= 1*direction
            player_tensor[2] += self.collect_dots(player_tensor[0])
        elif action.label == "boom":
            player_tensor[2] += max(1, self.econboomrate*player_tensor[2])
        elif action.label == "rush":
            if player_tensor[2] >= 1:
                player_tensor[1] = 2 # the number of dimensions is now 2
                militaryconversion = max(1, self.econtomilitaryconvrate*player_tensor[2])
                player_tensor[2] -= militaryconversion
                player_tensor[3] += militaryconversion

        if trigger_default_opponent_action:
            self.opponent_act()
        
        if self.enable_printouts:
            self.print_universe()

        return self.observe()

    def opponent_act(self):         # This is the rush AI.
        if self.tensors[1][2]>0 :   # Rush if possile
            self.tensors[1][2] -= 1
            self.tensors[1][3] += 1
            self.tensors[1][1] = 2      # the number of dimensions is now 2
        else:                       # Otherwise Advance.
            self.tensors[1][0] -= 1
            self.tensors[1][2] += self.collect_dots(self.tensors[1][0])

        return self.observe()

    def collect_dots(self, position):
        low, high = 0, len(self.clusters) - 1

        while low <= high:
            mid = (low + high) // 2
            current_value = self.clusters[mid][0]

            if current_value == position:
                dots = self.clusters[mid][1]
                self.clusters[mid][1] = 0
                return dots
            elif current_value < position:
                low = mid + 1
            else:
                high = mid - 1

        return 0        

    def print_universe(self):
        #    print(self.clusters)
        #    print(self.tensors)
        for j in range(self.mapsize):
            print(f" {j%10}", end="")
        print(" #")
        position_init = 0
        for i in range(len(self.clusters)):
            for j in range(position_init, self.clusters[i][0]):
                print("  ", end="")
            print(f" {self.clusters[i][1]}", end="")
            position_init = self.clusters[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

        position_init = 0
        for i in range(len(self.tensors)):
            for j in range(position_init, self.tensors[i][0]):
                print("  ", end="")
            print(f"{self.tensors[i][2]}", end="")
            if self.tensors[i][3]>=0:
                print(f"-{self.tensors[i][3]}", end="")
            position_init = self.tensors[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

class Interactive_TensorRTS(TensorRTS): 
    def __init__(self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,
        maxdots: int = 9, 
        enable_printouts : bool = True): 
        self.is_game_over = False

        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts=enable_printouts)

    def act(self, actions: Mapping[ActionName, Action],  trigger_default_opponent_action : bool = True, is_player_two : bool = False, print_universe : bool = False) -> Observation:
        obs_result = super().act(actions, False, is_player_two)

        if (obs_result.done == True):
            self.is_game_over = True

        return obs_result

class Agent(metaclass=abc.ABCMeta):
    def __init__(self, initial_observation : Observation, action_space : Dict[ActionName, ActionSpace]):
        self.previous_game_state = initial_observation
        self.action_space = action_space

    @abc.abstractmethod
    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]: 
        """Pure virtual function in which an agent should return the move that they will make on this turn.

        Returns:
            str: name of the action that will be taken
        """
        pass

    @abc.abstractmethod
    def on_game_start(self) -> None: 
        """Function which is called for the agent before the game begins.
        """
        pass

    @abc.abstractmethod
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        """Function which is called for the agent once the game is over.

        Args:
            did_i_win (bool): set to True if this agent won the game.
        """
        pass

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

class GameResult():
    def __init__(self, player_one_win : bool = False, player_two_win : bool = False, tie : bool = False):
        self.player_one_win = player_one_win
        self.player_two_win = player_two_win
        self.tie = tie

class GameRunner(): 
    def __init__(self, environment = None, enable_printouts : bool = False):
        self.game = Interactive_TensorRTS(enable_printouts=enable_printouts)
        self.game.reset()

        self.player_one = None
        self.player_two = None
        self.results : GameResult = None
    
    def assign_players(self, first_agent : Agent, second_agent : Agent = None):
        self.player_one = first_agent

        if second_agent is not None:
            self.player_two = second_agent

    def run(self): 
        assert(self.player_one is not None)

        game_state = self.game.observe()
        self.player_one.on_game_start()
        if self.player_two is not None: 
            self.player_two.on_game_start()

        while(self.game.is_game_over is False):
            #take moves and pass updated environments to agents
            game_state = self.game.act(self.player_one.take_turn(game_state))
            
            if (self.game.is_game_over is False):
                if self.player_two is None: 
                    game_state = self.game.opponent_act()
                else:
                    #future player_two code
                    game_state = self.game.act(self.player_two.take_turn(game_state), False, True)

        # who won? 
        tie = False
        win_p_one = False
        win_p_two = False

        p_one = self.game.tensor_power(0)
        p_two = self.game.tensor_power(1)

        if p_one > p_two: 
            win_p_one = True
        elif p_two > p_one:
            win_p_two = True
        else:
            tie = True

        self.results = GameResult(win_p_one, win_p_two, tie)
        self.player_one.on_game_over(win_p_one, tie)
        if self.player_two is not None:
            self.player_two.on_game_over(win_p_two, tie)

if __name__ == "__main__":  # This is to run wth agents
    runner = GameRunner()
    init_observation = runner.set_new_game()
    random_agent = Random_Agent(init_observation, runner.game.action_space())

    runner.assign_players(random_agent)
    runner.run()
    
if __name__ == "__main__":  #this is to run cli
    env = TensorRTS()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()    