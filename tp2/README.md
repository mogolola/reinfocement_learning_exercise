# Reinforcement Learning, Practical 2

# Remiders

As the requirements of the second practical are the same as the first one ;
you can use the same environment. As a reminder, the command to install all
the necessary requirements is `pip install -r requirements.txt`, to execute
from in your virtual environment.

# Practical 2
In this practical, you are going to implement various algorithms that probe the
environment they are in and learn to maximize their reward. On one hand, the
algorithms that learn only the V-function of the environment, and on the another
hand the algorithms that learn the Q values, i.e. the state-action values.

The algorithms are :

1. Value Iteration
2. Policy Iteration
3. Q-Learning
4. SARSA

To throughtly evaluate the difference in performance between these algorithms,
the default MDPs given will be harder to solve than in practical1. 

You are provided with the `main.py` file, a MDP test bed. Use `python main.py -h`
to check how you are supposed to use this file. You will quickly notice that all
subcommands return error messages.

You can try out new policies by implementing them in the `agents.py` file,
and try out different grids of MDPs by changing its parameters (test the 
stochasticity parameter).

You will be noted on the implementation of the 4 agents in the `agents.py` file. 
Bonus points will be given to clean and scalable code. 
(Think of your code complexity)


## How do I complete these files ?

Fill in the `# TO IMPLEMENT` part of the
code. Remove the expection raising part (`raise NotImplementedError` ), and 
complete the three blank methods for each Agent.

