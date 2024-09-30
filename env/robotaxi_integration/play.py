#!/usr/bin/env python3

""" Front-end script for replaying the Snake agent's behavior on a batch of episodes. """

import json
import sys, os

from robotaxi_integration.robotaxi.gameplay.environment import Environment
from robotaxi_integration.robotaxi.gui import PyGameNavigation, PyGameSurveillance
from robotaxi_integration.robotaxi.utils.cli import HelpOnFailArgumentParser
from robotaxi_integration.robotaxi.agent import HumanAgent, RandomActionAgent

gui = None

def parse_command_line_args(args):

    parser = HelpOnFailArgumentParser(description='Snake AI replay client.', epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json')

    parser.add_argument('--mode', type=str, choices=['navigation', 'surveillance'], help='Choose mode to run',)
    parser.add_argument('--interface', type=str, choices=['cli', 'gui'], default='gui', help='Interface mode (command-line or GUI).',)
    parser.add_argument('--agent', required=False, type=str, choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding', 'reward-learning', 'a2c'], default='random', help='Player agent to use.',)
    parser.add_argument('--model', type=str, help='File containing a pre-trained agent model.',)
    parser.add_argument('--level', type=str, default='./robotaxi_integration/levels/8x8-blank.json', help='JSON file containing a level definition.',)
    parser.add_argument('--num-episodes', type=int, default=1, help='The number of episodes to run consecutively.',)
    parser.add_argument('--save_frames', action="store_true", default=False, help='save frames as jpg files in screenshots/ folder.')
    parser.add_argument('--stationary', action="store_true", default=False, help='determine whether the environment is stationary')
    parser.add_argument('--collaborating_agent', type=str, choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding'], help='Collaborator agent to use.',)
    parser.add_argument('--collaborator_model', type=str, help='File containing a pre-trained agent model.',)
    parser.add_argument('--participant', type=str, default='test', help='Participant ID.',)
    parser.add_argument('--test_run', action="store_true", default=False, help='determine whether the environment is stationary')

    return parser.parse_args(args)


def create_snake_environment(level_filename, stationary, collaboration=None, test=False, participant=None, mode=None):

    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    if test: env_config["max_step_limit"] = 50

    return Environment(config=env_config, stationary=stationary, collaboration=collaboration, verbose=1, participant=participant, mode=mode)


def create_agent(name, model, dimension, env, reward_mapping=None):

    if name == 'human': return HumanAgent()
    if name == 'random': return RandomActionAgent()
    # if name == 'val-itr': return ValueIterationAgent(grid_size=dimension, env=env, reward_mapping=reward_mapping)
    # if name == 'mixed': return MixedActionAgent(grid_size=dimension, env=env)
    # if name == 'tile-coding': return TileCodingAgent(weights="tile_coding_weights_660.log")
    # if name == 'dqn':
    #     if model is None:
    #         raise ValueError('A model file is required for a DQN agent.')
    #     return DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=4)
    # if name == 'one-hot-dqn':
    #     if model is None:
    #         raise ValueError('A model file is required for an one-hot DQN agent.')
    #     return OneHotDQNAgent(model=model, memory_size=1000, channels=6)

    raise KeyError(f'Unknown agent type: "{name}"')


def play_gui(q2, q1, env, agent, agent_name, num_episodes, save_frames, field_size, participant, test=False, mode=None):
    global gui
    if mode == 'navigation':
        gui = PyGameNavigation(q2, q1, save_frames=save_frames, field_size=field_size, test=test, mode=mode)
    elif mode == 'surveillance':
        gui = PyGameSurveillance(q2, q1, save_frames=save_frames, field_size=field_size, test=test, mode=mode)
    else:
        raise ValueError('Specify the mode of the game: navigation or surveillance.')
    gui.load_environment(env)
    gui.load_agent(agent, agent_name)

    gui.run(num_episodes=num_episodes, participant=participant)


def start_gui(parsed_args, q2=None, q1=None):
    # parsed_args = parse_command_line_args(sys.argv[1:])

    if not os.path.exists('robotaxi_integration/csv'): os.makedirs('robotaxi_integration/csv')
    if not os.path.exists('robotaxi_integration/log'): os.makedirs('robotaxi_integration/log')

    env = create_snake_environment(parsed_args.level, parsed_args.stationary, test=parsed_args.test_run, participant=parsed_args.participant, mode=parsed_args.mode)
    # model = load_model(parsed_args.model) if parsed_args.model is not None else None
    model = None
    dimension = int(parsed_args.level.split('/')[-1].split('x')[0])

    agent = create_agent(parsed_args.agent, model, dimension, env)

    play_gui(q2, q1, env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes, save_frames=parsed_args.save_frames,
             field_size=dimension, participant=parsed_args.participant, test=parsed_args.test_run, mode=parsed_args.mode)


if __name__ == '__main__':

    start_gui(parse_command_line_args(sys.argv[1:]))
