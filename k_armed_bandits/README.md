# K-Armed bandit


## Overview
Implemented stationary stochastic K-bandits using Jax library. Explored  exploration, exploitation dynamics, and how fast reinforcement learning agents learn

### Goal
action - value estimation

exploration vs. exploitation

stochastic reward environments

functional state updates in jax + vmap

---
## Environment
Implemented a Gaussian K-Armed bandit
Each had a true reward mean \mu(a)


## Agent
Agent had 2 functions pick_every_lever and UCB Agent
State representation was (t, N(a), Q(a), key)
-t = timestamp
- N(a) = number of times each arm was selected
- Q(a) = Estimated reward of every arm
- key = key for jax rng 


### pick_every_lever
Rather than hard coding levers, I chose to go the textbook route 
Creates mask of every letter that has not been picked yet then returns index of first value 
Returned single integer as choice

### UCB Agent 
Implemented an UCB agent
Predicted optimal path with optional amount of exploration, through c 
Returned single integer as choice 



