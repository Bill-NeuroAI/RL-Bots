
# # Setup

import jax 
import jax.numpy as jnp

# # Environment 1
# We will first implement 10 slot machines.
# 
# 
# Environment owns 3 things. True rewards, probabilities, and step function

#state is (t,N,Q,key)

def k_slots(fn,state, t_r):
    t = state[0]
    N = state[1]
    Q = state[2]
    #Selecting k-arm
    s_m = fn(state)

    #Reward 
    key, k1 = jax.random.split(state[3])
    r = t_r[s_m] + jax.random.normal(k1)
    N = N.at[s_m].add(1)

    n_Q = Q[s_m] + ((1/N[s_m]) * (r - Q[s_m]))

    #updating state
    Q = Q.at[s_m].set(n_Q)
    t = t+1


    return (t, N, Q, key),N







