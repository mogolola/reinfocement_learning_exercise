import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import operator
from agents import (ValueIteration, PolicyIteration, QLearning, SARSA)

parser = argparse.ArgumentParser(
    description='test bed for dynamic programming algorithms')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

parser_VI = subparsers.add_parser(
    'VI', description='Value Iteration agent')
parser_PI = subparsers.add_parser(
    'PI', description='Policy Iteration agent')
parser_QL = subparsers.add_parser(
    'QL', description='Q-Learning agent')
parser_SARSA = subparsers.add_parser(
    'SARSA', description='SARSA agent')

parsers = [parser_VI, parser_PI, parser_QL, parser_SARSA]

arg_dico = {'VI': ValueIteration,
            'PI': PolicyIteration,
            'QL': QLearning,
            'SARSA': SARSA}

for pr in parsers:
    pr.add_argument('--learning-rate', type=float, metavar='lr',
                    default=0.1, help='Learning Rate')
    pr.add_argument('--discount', type=float, metavar='d',
                    default=0.9, help='Discount factor')
    pr.add_argument('--grid', type=int, metavar='g',
                    default=0, help='Type of grid, 0 for standard, 1 for labyrinth, 2 for more complicated')

args = parser.parse_args()

agent = args.agent
agent_options = vars(args)
agent_options.pop('agent')


class GridMDP(object):
    def __init__(self, size=(5, 5), starting_point="fixed",
                 terminal_states=[((4, 4), 20)],
                 stochasticity=0.0, walls=[], penalty=-5):
        super(GridMDP, self).__init__()
        # zero for blank space. N for reward and -1 for wall
        self.grid = np.zeros(size)
        for t_state in terminal_states:
            self.grid[t_state[0]] = t_state[1]  # t_state: (position, reward)
        self.walls = walls
        self.starting_point = starting_point
        self.stochasticity = stochasticity
        self.penalty = penalty  # Penalty for going out of bounds or crashing into a wall
        self.reward = []
        self.history = []
        self.action_history = []
        self.size = size
        self.restart()
        self.starting_point = starting_point




    def restart(self):
        self.reward.append([0])
        if self.starting_point == "fixed":
            self.position = (0, 0)
        elif self.starting_point == "random":
            self.position = (np.random.randint(
                0, self.size[0]), np.random.randint(0, self.size[0]))
            while self.grid[self.position] != 0 and self.position in self.walls:
                self.position = (np.random.randint(
                    0, self.size[0]), np.random.randint(0, self.size[0]))
        else:
            raise ValueError
        self.history.append([self.position])


    def act(self, action):
        if self.grid[self.position] > 0:  # Episode has ended
            self.reward[-1].append(self.grid[self.position])
            self.restart()
            # One turn of terminal state
        try:
            if np.random.choice([False, True], p=[1 - self.stochasticity, self.stochasticity]):
                # Stochastic effect
                n_actions = [i for i in [
                    (1, 0), (-1, 0), (0, 1), (0, -1)] if i != tuple(map(operator.mul, action, (-1, -1)))]
                action = n_actions[np.random.choice(range(len(n_actions)))]
                # all actions are possible except for the opposite of the action
            n_position = tuple(map(operator.add, action, self.position))
            if any([i < 0 for i in n_position]) or n_position in self.walls:
                raise IndexError  # Out of bounds
            # assess the reward
            self.grid[n_position]  # Test out of bounds
            self.history[-1].append(n_position)
            self.reward[-1].append(-1)
            self.position = n_position
            return [np.mean(i) for i in self.reward]
        except IndexError:
            self.reward[-1].append(self.penalty)
            self.history[-1].append(self.position)
            return [np.mean(i) for i in self.reward]
            # not moving

    # Do not modify this class.


def plot_results(meanrewards, policy, walls, v=None):
    actions = ['up', 'down', 'left', 'right']
    plt.figure(1)
    plt.plot(meanrewards)
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    for i in range(policy.shape[0]):
        plt.figure(i + 1)
        for wall in walls:
            policy[i, wall[0], wall[1]] = -1
        plt.matshow(policy[i, :, :])
        plt.title('Policy : ' + actions[i])
    if v is not None:
        for wall in walls:
            v[wall[0], wall[1]] = -30
        plt.matshow(v)
    plt.show()


class AgentTester:
    def __init__(self, agentClass, iterations, params):
        self.iterations = iterations
        if params['grid'] == 1:
            self.mdp = GridMDP(size=(10, 10), starting_point="fixed",
                               terminal_states=[((9, 9), 20)],
                               stochasticity=0.0,
                               walls=[(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (0, 5), (5, 7), (5, 8), (5, 9), (5, 6), (5, 5), (5, 4)], penalty=-5)
        elif params['grid'] == 0:
            self.mdp = GridMDP()

        elif params['grid'] == 2:
            self.mdp = GridMDP(size=(10, 10), starting_point="fixed",
                               terminal_states=[((9, 9), 20)],
                               stochasticity=0.1,
                               walls=[(1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (0, 5), (5, 7), (5, 8), (5, 9), (5, 6), (5, 5), (5, 4), (6, 4), (7, 4), (8, 4)], penalty=-5)
        self.agentClass = agentClass(self.mdp, **params)

    def oneStep(self):
        action = self.agentClass.action()
        mean_reward = self.mdp.act(action)
        self.agentClass.update()
        #input("Press Enter to continue...")
        return mean_reward

    def test(self):
        meanvalues = np.zeros(self.iterations)
        try:
            for i in range(self.iterations):
                meanrewards = self.oneStep()
                if i % 100 == 0:
                    display = '\nepoch: {:5.0f} -- mean reward: {:2.2f}'
                    sys.stdout.write(display.format(
                        i, np.mean(meanrewards)))
                    sys.stdout.flush()

            policy = self.agentClass.policy
            return meanrewards, policy
        except KeyboardInterrupt:
            print('\n\nInterrupted')
            policy = self.agentClass.policy
            return meanrewards[:-1], policy
        # Modify only the agent class


if __name__ == '__main__':
    try:
        tester = AgentTester(arg_dico[agent], 10000,
                             agent_options)
        # Do not modify.
        meanrewards, policy = tester.test()
        if hasattr(tester.agentClass, 'V'):
            plot_results(meanrewards, policy,
                         tester.mdp.walls, tester.agentClass.V)
        else:
            plot_results(meanrewards, policy,
                         tester.mdp.walls)
    except NotImplementedError as e:
        print('Unimplemented agent: {}'.format(
              e.args[0]))
