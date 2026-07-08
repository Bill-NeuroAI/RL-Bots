import jax 
import jax.numpy as jnp

def agent_updater(old_state, policy, reward, new_state,Q_table, gamma, alpha):
    #this is an inplace replacement so i would need to use at[] and set()
    n_q = ((1-alpha)*Q_table[old_state,policy]) + (alpha* (reward + (gamma*jnp.max(Q_table[new_state]))))
    Q_table = Q_table.at[old_state,policy].set(n_q)
    return Q_table

def epsilon_greedy(state, Q_table, epsilon, key):
    k1, k2 = jax.random.split(key)
    pred = jax.random.bernoulli(k1,(1-epsilon))
    true_fun = lambda :  Q_table[state].argmax()
    false_fun = lambda : jax.random.randint(k2, (), 0, 4)
    return jax.lax.cond(pred, true_fun=true_fun, false_fun=false_fun)
