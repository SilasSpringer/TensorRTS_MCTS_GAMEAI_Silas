import os
import sys
import json
import inspect
import importlib
import random
import math
from typing import Dict, List, Mapping, Tuple, Set
from TensorRTS import Agent, GameRunner, GameResult

from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *

NUM_GAMES_PER_ROUND = 3
NUM_GAMES_WIN_BY_TO_VICTORY = 1

class Bot(): 

    def load_module(path_to_script):
        """Try and load from the provided script

        Args:
            path_to_script (_type_): Path to the script 
        """

        #load module
        spec = importlib.util.spec_from_file_location('agent_hook', bot_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
    
    def load_student_name(path) -> str:
        spec = importlib.util.spec_from_file_location('student_name_hook', bot_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.student_name_hook()
    
    def create_instance(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
        return self.agent_module.agent_hook(init_observation, action_space)

    def __init__(self, path_to_agent_script : str) -> None:
        self.agent_module = Bot.load_module(path_to_agent_script)
        self.student_name : str = Bot.load_student_name(path_to_agent_script)
        self.path_to_agent_script = path_to_agent_script

class Matchup(): 
    def __init__(self, first_bot : Bot, second_bot : Bot) -> None:
        self.first_bot = first_bot
        self.second_bot = second_bot
        self.games : list[GameRunner] = []
        self.final_result : GameResult = None

    def fight(self, num_rounds : int, num_rounds_must_win_by_to_win : int) -> None: 
        assert(self.first_bot is not None and self.second_bot is not None)
        done = False

        num_rounds_played = 0
        num_wins_p_one = 0
        num_wins_p_two = 0

        while done is False:
            if num_rounds_played >= num_rounds and abs(num_wins_p_one - num_wins_p_two) >= num_rounds_must_win_by_to_win:
                done = True
                break
            new_game = GameRunner()
            
            #create bot instances
            observation = new_game.game.observe()
            action_space = new_game.game.action_space()
            player_one = self.first_bot.create_instance(observation, action_space)
            player_two = self.second_bot.create_instance(observation, action_space)

            new_game.assign_players(player_one, player_two)
            new_game.run()

            result = new_game.results

            if result.player_one_win:
                num_wins_p_one += 1
            elif result.player_two_win:
                num_wins_p_two += 1

            num_rounds_played += 1

        if num_wins_p_one > num_wins_p_two: 
            self.final_result = GameResult(player_one_win=True)
        elif num_wins_p_two > num_wins_p_one:
            self.final_result = GameResult(player_two_win=True)
        else:
            raise Exception('Number of wins for each player is the same. This should not happen.')
        
    def stringify_print_results(self) -> list[str]: 
        results : list[str] = []
        display_string = f'{self.first_bot.student_name} -- {self.second_bot.student_name}'
        results.append(display_string)
        if self.final_result.player_one_win:
            results.append(' [winner]')
        else:
            loc = display_string.find('--')
            display_string = '[winner]'.ljust(loc + 1)
            results.append(display_string)

        return results

class Bracket(): 
    def __init__(self) -> None:
        self.run = False
        self.matchups : list[Matchup] = []

    def add_matchup(self, first_bot : Bot, second_bot : Bot) -> None: 
        self.matchups.append(Matchup(first_bot, second_bot))

    def execute_game(self, num_rounds : int, num_rounds_must_win_by_to_win : int): 
        self.run = True

        for match in self.matchups:
            match.fight(num_rounds, num_rounds_must_win_by_to_win)

    def stringify_print_results(self) -> list[str]:
        assert(self.run)

        lines = []

        lines.append('Results of bracket: ')
        for match in self.matchups:
            match_result = match.stringify_print_results()
            
            #add padding
            for line in match_result: 
                lines.append(f'\t{line}')
        
        return lines

class Tournament():
    def __init__(self, bots : list[Bot], num_rounds_per_match : int, num_rounds_must_win_by : int) -> None:
        self.winner : Bot = None
        self.all_bots = bots
        self.in_bots = bots
        self.rounds : list[Bracket] = []

        self.num_rounds_per_match = num_rounds_per_match
        self.num_rounds_must_win_by = num_rounds_must_win_by
    
    def create_next_round(in_bots : list[Bot]) -> Bracket: 
        new_round = Bracket()

        already_selected = []
        for i in range(0, math.floor(len(in_bots)/2)): 
            selected_indicies = random.sample(range(len(in_bots)), 2)
            found_match = False

            while not found_match:
                if selected_indicies[0] not in already_selected and selected_indicies[1] not in already_selected: 
                    found_match = True

                    already_selected.append(selected_indicies[0])
                    already_selected.append(selected_indicies[1])

                    new_round.add_matchup(in_bots[selected_indicies[0]], in_bots[selected_indicies[1]])

        return new_round

    def run(self): 
        while len(self.in_bots) != 1: 
            print('Building next round')
            self.rounds.append(Tournament.create_next_round(self.in_bots))
            print('Running round')
            self.rounds[-1].execute_game(self.num_rounds_per_match, self.num_rounds_must_win_by)

            #remove losers from remaining bots
            for matchup in self.rounds[-1].matchups:
                if matchup.final_result.player_one_win: 
                    self.in_bots.remove(matchup.second_bot)
                elif matchup.final_result.player_two_win: 
                    self.in_bots.remove(matchup.second_bot)
                else:
                    raise Exception('Tie detected which should never happen.')
                
        self.winner = self.in_bots[0]

    def print_results(self) -> None: 
        results : list[str] = []
        results.append('-Beginning Tournament Prinout-')
        results.append("Tournament Settings: ")
        results.append(f'Winner: {self.winner.student_name}')
        results.append(f'\tNumber of games in each match: {self.num_rounds_per_match}')
        results.append(f'\tNumber of games win by for victory: {self.num_rounds_must_win_by}')
        results.append("Bracket Results:")

        num_rounds = len(self.rounds)
        for i in range(0, num_rounds):
            str_results = self.rounds[i].stringify_print_results()

            results.append(f'\tRound: {i+1}')
            for line in str_results: 
                results.append(f'\t\t{line}')

        for line in results:
            print(line)

        with open('Results.txt', 'w+') as result_file: 
            for line in results: 
                result_file.write(f'{line}\n')

game_runner = GameRunner(enable_printouts=False)

if __name__ == "__main__":
    bots : list[Bot] = []

    bot_root = os.path.join(os.getcwd(), "bots")

    if not os.path.isdir(bot_root): 
        print("Bot directory does not exist.")
        exit()

    for path in os.listdir(bot_root): 
        bot_script = None
        target_dir = os.path.join(bot_root, path)

        #find script
        for ele in os.listdir(target_dir): 
            ele_path = os.path.join(target_dir, ele)
            if '.py' in os.path.basename(ele_path) and '__init__.py' not in ele_path:
                bot_script = ele_path
                break
        
        if bot_script is None: 
            print(f'Failed to find script within provided directory - {path}')
            exit()
  
        #create bot 
        new_bot = Bot(bot_script)
        assert(new_bot is not None)
        bots.append(new_bot)

    tournament = Tournament(bots, NUM_GAMES_PER_ROUND, NUM_GAMES_WIN_BY_TO_VICTORY)
    tournament.run()
    tournament.print_results()