import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import operator
from agents import VEvalTemporalDifferencing
from agents import VEvalMonteCarlo
from agents import RandomPolicy

parser = argparse.ArgumentParser(
    description='test bed for dynamic programming algorithms')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

parser_vTD = subparsers.add_parser(
    'vTD', description='V-function evaluation using Bellman equation, temporal differencing')
parser_vmonte = subparsers.add_parser(
    'vmonte', description='V - function evaluation using Monte - carlo approximations')


arg_dico = {'vTD': VEvalTemporalDifferencing,
            'vmonte': VEvalMonteCarlo}

parser_vTD.add_argument('--learning-rate', type=float, metavar='lr',
                        default=0.1, help='Learning Rate')
parser_vTD.add_argument('--discount', type=float, metavar='d',
                        default=0.6, help='Discount factor')
parser_vmonte.add_argument('--discount', type=float, metavar='d',
                           default=0.6, help='Discount factor')

args = parser.parse_args()

agent = args.agent
agent_options = vars(args)
agent_options.pop('agent')


class GridMDP(object):
    def __init__(self, size=(5, 5), starting_point="fixed", terminal_states=[((4, 4), 20)], stochasticity=0.0, walls=[], penalty=-5):
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
        self.restart()
        self.size = size
        self.starting_point = starting_point

    def restart(self):
        self.reward.append([0])
        if self.starting_point == "fixed":
            self.position = (0, 0)
        else:
            self.position = (np.random.randint(
                0, self.size[0]), np.random.randint(0, self.size[0]))
            while self.grid[self.position] != 0:
                self.position = (np.random.randint(
                    0, self.size[0]), np.random.randint(0, self.size[0]))
        self.history.append([self.position])

    def act(self, action):
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
            reward = self.grid[n_position]
            self.history[-1].append(n_position)

            if reward > 0:  # Episode has ended
                self.reward[-1].append(reward)
                self.restart()
            else:
                self.reward[-1].append(-1)
                self.position = n_position
            return np.sum([np.sum(i) for i in self.reward])
        except IndexError:
            self.reward[-1].append(self.penalty)
            self.history[-1].append(self.position)
            return np.sum([np.sum(i) for i in self.reward])
            # not moving

    # Do not modify this class.


def plot_results(meanrewards, meanvalues, statevalues):
    plt.figure(1)
    plt.plot(meanrewards)
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Average state value')
    plt.plot(meanvalues)
    plt.matshow(statevalues)
    plt.show()


class AgentTester:
    def __init__(self, agentClass, iterations, params):
        self.iterations = iterations
        self.mdp = GridMDP()
        self.agentClass = agentClass(self.mdp, RandomPolicy, **params)

    def oneStep(self):
        action = self.agentClass.action()
        mean_reward = self.mdp.act(action)
        self.agentClass.update()
        mean_value = np.mean(self.agentClass.values)
        return mean_reward, mean_value

    def test(self):
        meanrewards = np.zeros(self.iterations)
        meanvalues = np.zeros(self.iterations)
        try:
            for i in range(self.iterations):
                meanrewards[i], meanvalues[i] = self.oneStep()
                if i % 100 == 0:
                    display = '\nepoch: {:5.0f} -- mean reward: {:2.2f} -- mean value: {:2.2f} '
                    sys.stdout.write(display.format(
                        i, meanrewards[i], meanvalues[i]))
                    sys.stdout.flush()

            statevalues = self.agentClass.values
            return meanrewards, meanvalues, statevalues
        except KeyboardInterrupt:
            print('\n\nInterrupted')
            last_fill = np.argwhere(meanrewards == 0)[0][0]
            statevalues = self.agentClass.values
            return meanrewards[: last_fill], meanvalues[: last_fill], statevalues
# Modify only the agent class


if __name__ == '__main__':
    try:
        tester = AgentTester(arg_dico[agent], 10000,
                             agent_options)
        # Do not modify.
        meanrewards, meanvalues, statevalues = tester.test()
        plot_results(meanrewards, meanvalues, statevalues)
    except NotImplementedError as e:
        print('Unimplemented agent: {}'.format(
              e.args[0]))
