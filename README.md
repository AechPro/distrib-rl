# Distributed Reinforcement Learning
A platform to perform on-policy Reinforcement Learning with multiple asynchronous computers. The platform's structure is such that all data 
from connected workers is routed to a Redis server, where a learner consumes the available data and uses it to compute updates to a policy 
and Value Function estimator. These models are then passed to Redis so workers can update their local copies.

# Features
Distrib-RL offers a variety of features to make it easier to scientifically test algorithms and environments within the framework. 

### Weights & Biases integration
Distrib-RL natively integrates logging to Weights & Biases so users can view graphs from training runs in real-time in a browser with their Weights & Biases account.

### Fully automated experiments 
All Distrib-RL runs operate on a single configuration file. Modifications of this file can be specified in the form of an `Experiment`. 
Experiments modify a config file and test that modification across a range of random seeds before moving on to the subsequent modification. 
Experiments automatically create a Weights & Biases project and results from every seeded run from every modification of a config in 
an experiment are automatically logged to groups under that project where users can view the results of their experiments.

### Determinism
To enable reproducibility in Distrib-RL, every source of randomness is seeded. The seed must be specified in a config file and experiments will modify the seed automatically 
by incrementing it while testing modifications of the config. Because of the distributed nature of the data production and consumption processes, there will always be some 
uncontrollable randomness in the system. Connected CPU cores will slightly slow down or speed up at random depending on what other applications are running on the machine, 
how the  operating system schedules the worker process, and many other unpredictable factors. These factors mean that running the same experiment twice with Distrib-RL will 
not result in an identical set of runs, but they will be as close to identical as possible.

### Multi-Agent Reinforcement Learning
Distrib-RL is designed with Multi-Agent Reinforcement Learning (MARL) in mind. The Rocket League Gym (RLGym) is natively supported, so users can train Rocket League agents
with Reinforcement Learning while making use of multiple asynchronous machines.

# Installation
Distrib-RL is still in the early stages of development and will undergo many breaking changes before it is released as a PIP package. Right now the only way to install
and use Distrib-RL is to download the repository and run the code locally.
