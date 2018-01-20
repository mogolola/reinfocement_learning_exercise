"""
File to complete. Contains the agents
VEvalTemporalDifferencing and VEvalMonteCarlo (TD0)
"""
import numpy as np
import operator
import math


class Agent(object):
    # DO NOT MODIFY
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(Agent, self).__init__()
        if initial_policy is not None:
            self.policy = initial_policy
        else:
            self.policy = np.zeros((4, mdp.size[0], mdp.size[1])) + 0.25
        # Init the random policy
        # dim[0] is the actions, in the order (up,down,left,right)
        self.mdp = mdp
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.discount = kwargs.get("discount", 0.95)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        # For some agents : V Values
        self.V = np.zeros(mdp.size)

        # For others: Q values
        self.Q = np.zeros((4, mdp.size[0], mdp.size[1]))

    def update(self):
        # DO NOT MODIFY
        raise NotImplementedError

    def action(self):
        self.last_position = self.mdp.position
        return self.actions[np.random.choice(range(len(self.actions)),
                                             p=self.policy[:, self.last_position[0],
                                                           self.last_position[1]])]

    def best_move(self, current_status):
        # this function is to evaluate each action and return the best action
        # who maximize value of action based on status. This function is used by Value iteration and Policy iteration
        best_value = -100
        best_action = None
        for action in self.actions:  # loop over all 4 actions to find best action
            V_sa = 0  # initilize value of action based on status
            n_actions = [i for i in [
                (1, 0), (-1, 0), (0, 1), (0, -1)] if i != tuple(
                map(operator.mul, action, (-1, -1)))]  # get list of possible real action based on command action

            P_random = self.mdp.stochasticity
            P_determinate = (1 - self.mdp.stochasticity)
            P_sas = {}  # get a dictionary to store the possibility of each real_action
            for real_action in n_actions:
                if real_action == action:
                    P_sas[real_action] = P_determinate + (P_random / 3)
                else:
                    P_sas[real_action] = (P_random / 3)
            for real_action in n_actions:  # loop over each possible real action and sum to get V_sa
                next_state = tuple(map(operator.add, real_action, current_status))
                (i, j) = next_state
                if i < 0 or i >= self.mdp.size[0] or j < 0 or j >= self.mdp.size[1] or (i, j) in self.mdp.walls:
                    next_state = current_status
                    R_sas = self.mdp.penalty + self.mdp.grid[next_state]
                    V_next_state = self.V[next_state]
                else:
                    R_sas = self.mdp.grid[next_state]
                    V_next_state = self.V[next_state]

                V_sa = V_sa + P_sas[real_action] * (R_sas + self.discount * V_next_state)

            if V_sa > best_value:  # compare to get best_value and best_action
                best_value = V_sa
                best_action = action
        return best_action, best_value


class ValueIteration(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(ValueIteration, self).__init__(
            mdp, initial_policy, *args, **kwargs)

        self.theta = 0.01 #threshold to judge convergence
        self.var = 0 #initilize var, which indicate the biggest change of status value
        self.convergence = False

    def update(self):
        # TO IMPLEMENT
        if self.convergence: # if convergence, no more need to update
            pass
        else:
            while not self.convergence: #loop until convergence
                V_grid_next = np.zeros(self.mdp.size) # initilize V_grid_next to store updated value
                Policy_etoile = np.zeros((4, self.mdp.size[0], self.mdp.size[1])) + 0.25 #initialize to store best policy for tempora


                it = np.nditer(self.mdp.grid, flags=['multi_index'])
                while not it.finished: # loop over all status
                    current_state = it.multi_index
                    if current_state not in self.mdp.walls:

                        best_action, best_value = self.best_move(current_state)

                        V_grid_next[current_state] = best_value  #store value for each status

                        best_action_index = self.actions.index(best_action) #store policy for each status
                        for index, _policy in enumerate(Policy_etoile):
                            if index == best_action_index:
                                Policy_etoile[index][current_state] = 1
                            else:
                                Policy_etoile[index][current_state] = 0
                    it.iternext()

                self.var = np.max(abs(self.V - V_grid_next)) #get biggest change of status value
                self.V = V_grid_next #update status value
                if self.var < self.theta:
                    self.convergence = True  #if var smaller than threshold theta, judge convergence
                    self.policy = Policy_etoile  # update policy, notice that this is the only update of policy!!!!!





    def action(self):
        # YOU CAN MODIFY
        return super(ValueIteration, self).action()

class PolicyIteration(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(PolicyIteration, self).__init__(
            mdp, initial_policy, *args, **kwargs)
        self.theta = 0.01  # threshold to judge convergence
        self.var = 0  # initilize var, which indicate the biggest change of status value
        self.policy_next = np.zeros((4, self.mdp.size[0], self.mdp.size[1])) + 0.25 # initilize improved policy
        self.convergence = False
        self.policy_stable = False



    def update(self):
        # TO IMPLEMENT

        while not self.policy_stable: # loop until the policy won't change


            self.convergence = False
            # get the real possibility moving to each direction
            P_up = self.policy[0] * (1 - 2 / 3 * self.mdp.stochasticity) + (self.policy[2] + self.policy[
                3]) * self.mdp.stochasticity / 3
            P_down = self.policy[1] * (1 - 2 / 3 * self.mdp.stochasticity) + (self.policy[2] + self.policy[
                3]) * self.mdp.stochasticity / 3
            P_left = self.policy[2] * (1 - 2 / 3 * self.mdp.stochasticity) + (self.policy[0] + self.policy[
                1]) * self.mdp.stochasticity / 3
            P_right = self.policy[3] * (1 - 2 / 3 * self.mdp.stochasticity) + (self.policy[0] + self.policy[
                1]) * self.mdp.stochasticity / 3

            P_action = [P_up, P_down, P_left, P_right]

            while not self.convergence:  # loop until convergence
                V_grid_next = np.zeros(self.mdp.size)
                it = np.nditer(self.mdp.grid, flags=['multi_index'])
                while not it.finished:  # loop over all status, similar to value iteration but based on given policy
                    current_state = it.multi_index
                    if current_state not in self.mdp.walls:
                        for index, real_action in enumerate(self.actions): # loop over each possible direction
                            next_state = tuple(map(operator.add, real_action, current_state))
                            (i, j) = next_state
                            if i < 0 or i >= self.mdp.size[0] or j < 0 or j >= self.mdp.size[1] or (i, j) in self.mdp.walls: # if out of bound or in walls
                                next_state = current_state
                                R_sas = self.mdp.penalty + self.mdp.grid[next_state]
                                V_next_state = self.V[next_state]
                            else:
                                R_sas = self.mdp.grid[next_state]
                                V_next_state = self.V[next_state]
                            V_grid_next[current_state] += P_action[index][current_state] * (R_sas + self.discount * V_next_state)

                    it.iternext()

                self.var = np.max(abs(self.V - V_grid_next))
                self.V = V_grid_next  # update status value
                if self.var < self.theta:
                    self.convergence = True  # if var smaller than threshold theta, judge convergence



            it = np.nditer(self.mdp.grid, flags=['multi_index'])
            while not it.finished:  # loop over all status, to improve policy
                current_state = it.multi_index
                if current_state not in self.mdp.walls:
                    best_action, best_value = self.best_move(current_state)
                    best_action_index = self.actions.index(best_action)  # store policy for each status
                    for index, _policy in enumerate(self.policy_next):
                        if index == best_action_index:
                            self.policy_next[index][current_state] = 1
                        else:
                            self.policy_next[index][current_state] = 0
                it.iternext()


            if np.array_equal(self.policy, self.policy_next):
                self.policy_stable = True
            self.policy = self.policy_next





    def action(self):
        # YOU CAN MODIFY
        return super(PolicyIteration, self).action()


class QLearning(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(QLearning, self).__init__(mdp, initial_policy, *args, **kwargs)
        self.epsilon = 0.8 # hyper parameter for trade_off
        self.beta = 0.99 # hyper parameter to decay epsilon after each episode
        self.alpha = 0.9 # hyper parameter for learning rate, value between (0,1)
        self.current_action = None

    def update(self):
        # TO IMPLEMENT
        if len(self.mdp.history) == 1: #if this is the begining
            pass

        else:
            if len(self.mdp.history[-1]) == 2:  #if the the serach path restart
                self.epsilon *= self.beta #update epsilon

            state_t = self.mdp.history[-1][-2] #current state to evaluate Q(s,a)
            state_tplus1 = self.mdp.history[-1][-1] #next state
            r = self.mdp.reward[-1][-1] #reward of action taken

            action = self.current_action
            action_index = self.actions.index(action)
            self.Q[action_index][state_t] += self.alpha * (r +
                                                self.discount * max(self.Q[i][state_tplus1] for i in range(4)) -
                                                       self.Q[action_index][state_t]) #Q learning function
            if np.random.rand() > self.epsilon: #the possibility for updating Policy
                it = np.nditer(self.mdp.grid, flags=['multi_index'])
                while not it.finished:  # loop over all status to update policy
                    current_state = it.multi_index
                    Q_max = -100
                    best_action_index = None
                    for i in range(4):
                        if self.Q[i][current_state] > Q_max:
                            Q_max = self.Q[i][current_state]
                            best_action_index = i
                    for index, _policy in enumerate(self.policy):
                        if index == best_action_index:
                            self.policy[index][current_state] = 1
                        else:
                            self.policy[index][current_state] = 0
                    it.iternext()




    def action(self):
        # YOU CAN MODIFY
        self.current_action = super(QLearning, self).action() # register the last action taken
        return self.current_action


class SARSA(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(SARSA, self).__init__(mdp, initial_policy, *args, **kwargs)
        self.epsilon = 0.8
        self.beta = 0.95
        self.alpha = 0.9
        self.action_t = None  # the previous action
        self.action_tplus1 = None # the next action

    def update(self):
        # TO IMPLEMENT
        if len(self.mdp.history) == 1:
            pass

        else:
            if len(self.mdp.history[-1]) == 2:  #if the the serach path restart
                state_t = self.mdp.history[-2][-2] #register the last status of the previous episode
                state_tplus1 = self.mdp.history[-1][-2] #register the second status of current episode, the first is actually not very important
                r_t = self.mdp.reward[-2][-1] #register the last reward of previous episode, which is for sure, 20
                self.epsilon *= self.beta
            else: #if the search path continue
                state_t = self.mdp.history[-1][-3] #register the status to be evaluated
                state_tplus1 = self.mdp.history[-1][-2] #registere the next status,
                # the two actions and status and reward must respect to following sequence: SARSA
                r_t = self.mdp.reward[-1][-2]

            action_t_index = self.actions.index(self.action_t)
            action_tplus1_index = self.actions.index(self.action_tplus1)

            self.Q[action_t_index][state_t] += self.alpha * (r_t + self.discount * self.Q[action_tplus1_index][state_tplus1]
                                                             - self.Q[action_t_index][state_t]) # SARSA algo function


            if np.random.rand() > self.epsilon: # update policy under controlled posiibility
                it = np.nditer(self.mdp.grid, flags=['multi_index'])
                while not it.finished:  # loop over all status
                    current_state = it.multi_index
                    Q_max = -100
                    best_action_index = None
                    for i in range(4):
                        if self.Q[i][current_state] > Q_max:
                            Q_max = self.Q[i][current_state]
                            best_action_index = i
                    for index, _policy in enumerate(self.policy):
                        if index == best_action_index:
                            self.policy[index][current_state] = 1
                        else:
                            self.policy[index][current_state] = 0
                    it.iternext()

    def action(self):
        # YOU CAN MODIFY
        self.action_t = self.action_tplus1
        self.action_tplus1 = super(SARSA, self).action()
        return self.action_tplus1
