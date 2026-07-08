import jax 
import jax.numpy as jnp

def agent_updater(old_state, policy, reward, new_state,Q_table):
    #this is an inplace replacement so i would need to use at[] and set()
    n_q = ((1-alpha)*Q_table[old_state][policy]) + (alpha* (reward + (gamma*jnp.max(Q_table[new_state]))))
    Q_table = Q_table.at[old_state,policy].set(n_q)
    return Q_table

def epsilon_greedy(state, Q_table, epsilon, key):
    if(jax.random.bernoulli(key) > epsilon):
        return Q_table[state].argmax()
    else:
        return jax.random.randint(key, (1), 0, 3)



