# Reinforcement Learning, Practical 1

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


## How do I complete these files ?
Just fill in the `# TO IMPLEMENT` part of the
code. Remove the expection raising part (`raise NotImplementedError` ), and 
complete the two blank methods for each Agent.

