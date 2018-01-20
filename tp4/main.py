import sys

import pylab as plb
import numpy as np
import mountain_car


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass

# implement your own agent here

class QLearningAgent():
    def __init__(self):

        self.x_p = 400  # Number of intervals in horizental axis
        self.v_k = 50  # Number of intervals in horizental velocity


        #initialize state space
        self.s_1 = np.zeros([self.x_p + 1, self.v_k + 1])
        self.s_2 = np.zeros([self.x_p + 1, self.v_k + 1])
        for i in range(self.x_p + 1):
            self.s_1[i, :] = -150.0 + i * 300.0 / float(self.x_p)
        for i in range(self.v_k + 1):
            self.s_2[:, i] = -20.0 + i * 40.0 / float(self.v_k)

        # weight initialization
        self.W = np.zeros([(self.x_p + 1), (self.v_k + 1),3])


        self.E = np.zeros(self.W.shape)

        # Q(s,a) when a = -1,0,1
        self.q = np.zeros(3)
        self.q_tminus1 = np.zeros(3)

        # correction
        self.cor = np.zeros(self.W.shape)

        # feature
        self.feature = np.zeros([self.x_p + 1, self.v_k + 1])
        self.feature_tminus1 = np.zeros([self.x_p + 1, self.v_k + 1])

        #judge if it is the first action
        self.is_first_action = True

        # Hyper parameters
        self.epsilon = 0.6  # epsilon greedy
        self.decay = 0.8    #decay of epsilon after each episode
        self.gamma = 0.6    #discount
        self.alpha = 0.2    #learning rate
        self.lamda = 0.9    #TD lambda


    # compute the feature vector based on the current state(x, vx)
    def compute_feature(self, x, v):
        return np.exp(-(x - self.s_1) * (x - self.s_1) - (v - self.s_2) * (v - self.s_2))

    def get_q(self,feature):
        # Q(s,a) when a = -1,0,1
        # Q(s,a) = SUM_i,j[W(a,i,j) * f(i,j)]
        q = np.zeros(3)
        for i in range(3):
            q[i] = np.sum( np.multiply(feature, self.W[:,:,i]))
        return q


    def act(self):  # epislon greedy policy

        rand = np.random.uniform()
        if rand < self.epsilon:
            self.action = np.random.randint(-1, 2)
        else:
            self.action = np.argmax(self.q) - 1
        #print (self.action)
        return self.action

    # update w using TD(lambda) back-views
    def update(self, nextState, reward):


        if reward > 0:
            self.epsilon *= self.decay      #decay epsilon at the end of each episode
            print (reward)

        if self.is_first_action == False :  #for the first action, do not update parameter
            #update W for the last state
            cor = reward + self.gamma * np.max(self.q) - np.max(self.q_tminus1)    #difference of expected Q and actual Q
            self.E *= self.gamma*self.lamda
            self.E[:,:,self.action_tminus1+1] += self.feature_tminus1   #update E
            self.W[:,:,self.action_tminus1+1] += self.alpha * cor * self.E[:,:,self.action_tminus1+1]   #update W

        self.q_tminus1 = self.q     #register q
        self.feature_tminus1 = self.feature     #register feature
        self.action_tminus1 = self.action       #register action

        #update Features and Q for current state
        x = nextState[0]
        v = nextState[1]
        self.feature = self.compute_feature(x, v)
        self.q = self.get_q(self.feature)

        self.is_first_action = False



# test class, you do not need to modify this class
class Tester:

    def __init__(self, agent):
        self.mountain_car = mountain_car.MountainCar(T=-50)
        self.agent = agent

    def visualize_trial(self, n_steps=100):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountain_car.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            print("Enter to continue...")
            input()

            sys.stdout.flush()

            reward = self.mountain_car.act(self.agent.act())
            self.agent.state = [self.mountain_car.x, self.mountain_car.vx]

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """

        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            self.mountain_car.reset()
            step = 1
            while step <= max_episode or max_episode <= 0:
                reward = self.mountain_car.act(self.agent.act())
                self.agent.update([self.mountain_car.x, self.mountain_car.vx],
                                  reward)
                rewards[c_episodes] += reward
                if reward > 0.:
                    break
                step += 1
            #self.agent.epsilon = self.agent.epsilon * 0.9
            print (self.agent.epsilon)
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    agent = QLearningAgent()
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(20, 3000)

    print("End of learning, press Enter to visualize...")
    input()
    test.visualize_trial()
    plb.show()
