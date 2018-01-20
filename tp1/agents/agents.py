"""
File to complete. Contains the agents
VEvalTemporalDifferencing and VEvalMonteCarlo (TD0)
"""
import numpy as np
import math


class Policy(object):
    """ Base class for policies. Do not modify
    """

    def __init__(self):
        super(Policy, self).__init__()

    def action(self, mdp, state, values):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self):
        super(RandomPolicy, self).__init__()

    def action(self, *args, **kwargs):
        actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return actions[np.random.choice(range(len(actions)))]


class VEvalTemporalDifferencing(object):
    def __init__(self, mdp, policy, *args, **kwargs):
        super(VEvalTemporalDifferencing, self).__init__()
        self.mdp = mdp
        self.policy = policy()
        self.values = np.zeros(mdp.size)  # Store state values in this variable
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.discount = kwargs.get('discount', 0.6)

    def update(self):
        # TO IMPLEMENT
        # Ingredients : discount, values, learning_rate, old position, new position, reward,...
        if self.mdp.history[-1] == [(0, 0)]:  #if the the serach path restart
            state_t = self.mdp.history[-2][-2]
            state_tplus1 = self.mdp.history[-2][-1]
            r = self.mdp.reward[-2][-1]
        else: #if the search path continue
            state_t = self.mdp.history[-1][-2]
            state_tplus1 = self.mdp.history[-1][-1]
            r = self.mdp.reward[-1][-1]
        #the 1st belleman function V
        self.values[state_t] = self.values[state_t] + self.learning_rate * (r + self.discount * self.values[state_tplus1] - self.values[state_t])

    def action(self):
        self.last_position = self.mdp.position
        self.last_action = self.policy.action(
            self.mdp, self.last_position, self.values)
        return self.last_action


class VEvalMonteCarlo(object):
    def __init__(self, mdp, policy, *args, **kwargs):
        super(VEvalMonteCarlo, self).__init__()
        self.mdp = mdp
        self.policy = policy
        self.values = np.zeros(mdp.size)  # Store state values in this variable
        self.sum_values = np.zeros(mdp.size)
        self.n_transitions = np.zeros(mdp.size)
        self.discount = kwargs.get('discount', 0.6)

    def update(self):
        # TO IMPLEMENT
        # Ingredients: Reward, history of positions, values, discount,...
        if self.mdp.history[-1] == [(0,0)]: #if a new episode begin, then update
            for state in self.mdp.history[-2]: #update state values for every state passed by in the last episode
                first_occurence_idx = self.mdp.history[-2].index(state) #get the index of first occurence of the state in episode
                R = sum(x * (self.discount ** i) for i,x in enumerate(self.mdp.reward[-2][first_occurence_idx:])) #calculate for sum of discounted rewards starting from the state in episode
                self.n_transitions[state] += 1 #count for appearance times in all episodes
                self.sum_values[state] += R #count for sum of values
                self.values[state] = self.sum_values[state] / self.n_transitions[state] #update mean state value
        else:
            pass #do nothing



    def action(self):
        self.last_position = self.mdp.position
        self.last_action = self.policy.action(
            self.mdp, self.last_position, self.values)
        return self.last_action
