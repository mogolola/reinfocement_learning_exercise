import numpy as np
import math


class epsGreedyAgent: #the greedy algorithms
    def __init__(self, A, epsilon):
        self.epsilon = epsilon
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon:
            a = np.random.randint(0, self.A)
        else:
            a = np.argmax(self.Q)
        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class optimistEpsGreedyAgent(epsGreedyAgent):
    def __init__(self, A, epsilon, optimism):
        self.epsilon = epsilon
        self.A = A
        self.Q = np.ones(A) * optimism #Q is initilized to be the optimum expected. this approach is Approximating from upperbound
        self.numpick = np.zeros(A)

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon:
            a = np.random.randint(0, self.A)
        else:
            a = np.argmax(self.Q)
        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class softmaxAgent:

    def __init__(self, A, temperature):
        self.temperature = temperature
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self): #choose action according to sotfmax probability, reason why not choose the biggest probability is that
                        #it will be easily stuck into local optimum
        prob = np.exp(self.Q / self.temperature) / sum(np.exp(self.Q / self.temperature))
        a = np.random.choice(
            self.A,
            1,
            p=prob
        )
        return a


    '''
    def interact(self):
        prob = np.exp(self.Q / self.temperature) / sum(np.exp(self.Q / self.temperature))
        a = np.argmax(prob)
        return a
    '''

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class UCBAgent:
    def __init__(self, A):
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self):
        E = self.Q + np.sqrt(np.log(np.full(self.numpick.shape,self.numpick.sum())) / self.numpick) #E is the expectation of
                                                                        #value. This approach comprehensively consider
                                                                        #Confidence interval and confidence. The item less chosen is
                                                                        #thus considered to have smaller Confidence interval and will have
                                                                        #bigger E, so that more likely to be chosen
        a = np.argmax(E)

        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class ThompsonAgent:
    '''
    this approach is inpsired by the paper: http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf
    the idea is to use beta distribution to approximate the optimum mean reward and its action
    '''
    def __init__(self, A, mu_0, var_0):
        self.S = np.zeros(A) #number of success, used as parameter of beta distribution
        self.F = np.zeros(A) #number of failure, used as parameter of beta distribution
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)
        self.r_tminus1 = 0.0 #initialze the previous reward to be zero
        self.a_tminus1 = 0 #initialze the previous action to be zero
        self.r_upperbd = 2.0 #resume the upperbound, resaon to give a upperbound is that the reward is originally given by
                             # gaussien distribution, so extream value exist. But we want to avoid beta distribution
                             # of bernoulli trail result (p_s below) to be sparse
        self.r_lowerbd = -2.0 #same reason to give a lowerbound
        self.r_max = 1.0 #initilize preliminary maximum value
        self.r_min = -1.0#initilize preliminary minimum value

    def interact(self):
        if self.r_tminus1 < 0.5:  #if reward is less than a threshold, we will mark it as failure
            self.F[self.a_tminus1] += 1
        else:
            p_s = (self.r_tminus1 - self.r_min) / (self.r_max - self.r_min) #we approximate the probability of the reward to be winner or failure
            r_bernoulli = np.random.binomial(1, p_s) #use bernoulli trail with the probability above to determinate the action to be a winner or failure, and update the parameter of bernoulli distribution
            #update parameter of bernoulli distribution
            if r_bernoulli == 1:
                self.S[self.a_tminus1] += 1
            else:
                self.F[self.a_tminus1] += 1


        Bernoulli = [np.random.beta(self.S[i]+1,self.F[i]+1) for i in range(self.A)] #based on cuurent bernoulli distribution parameter, each distribution will generate a random number,
        a = np.argmax(Bernoulli) #return the action who give the higgest number
        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]
        self.r_tminus1 = r
        self.a_tminus1 = a
        if self.r_max < r:
            self.r_max = r
        if self.r_max > self.r_upperbd:
            self.rmax = self.r_upperbd

        if self.r_min > r:
            self.r_min = r
        if self.r_min < self.r_lowerbd:
            self.r_min = self.r_lowerbd

