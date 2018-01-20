# Master_AIC_reinforcement_learning_exercise


# Introduction and set-up
For all practicals, we will make heavy use of python and many of its
libraries. To have access to the full power of python, virtual environments
and the command line, it is recommended to have access to a Unix system, an
emulated terminal (Windows 10 gives you access to such a terminal, or even anaconda), or a
Virtual Machine.

Besides, we advice you to use conda to manage the various requirements that may
change from one practical to the other. Conda can be installed following [this
link](https://conda.io/docs/install/quick.html). Install the latest python 3
version of conda. Once conda is installed, before working on a practical, you
should create a new conda environment. For example for the second practical,
execute `conda create -n practical1`. You can then access your new environment
using `source activate practical1`. Each of the practical provides you with a
`requirements.txt` file. You can install all the necessary requirements by
running `pip install -r requirements.txt` in your virtual environment.

# Practical 1
Markov Decision Processes (MDPs), Policy evaluation & V-function evaluation
through Bellman's equation and Monte-Carlo.
In this practical, you are going to be introduced to MDPs as a grid world. In
this setting, you are going to evaluate different policies through V-functions,
using first Bellman's equation and then Monte-Carlo approximations. The various
classes to fill in will be :

1. V-function evaluation (Monte Carlo approximation)
2. V-function evaluation (Bellman's equation)

You are provided with the `main.py` file, a MDP test bed. Use `python main.py -h`
to check how you are supposed to use this file. You will quickly notice that all
subcommands return error messages.

You can try out new policies by implementing them in the `agents.py` file,
and try out different grids of MDPs by changing its parameters (test the 
stochasticity parameter).

You will be noted on the implementation of the 2 agents in the `agents.py` file. 
Bonus points will be given to clean and scalable code. 
(Think of your code complexity)


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

# Practical 3
In this first practical, you are asked to put what you just learnt
about bandits to good use. You are provided with the `main.py` file,
a bandits test bed. Use `python main.py -h` to check how you are
supposed to use this file. You will quickly notice that all but the
`eps` subcommand return error messages. Your job is to fix this behavior
by implementing optimistic, softmax and UCB agents in the `agents/agents.py`
file. 


# Practical 4
This practical corresponds to the one given at Centrale last year, made by Guillaume Charpiat and Corentin Tallec -as the practical are shared between Centrale and M2 AIC-. You will find the subject in the .pdf file in the git, and two files : one is the definition of the famous mountain car problem and the other is the file used for testing and implementing your agents. 