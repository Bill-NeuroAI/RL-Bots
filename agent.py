# %% [markdown]
# # Setup

# %%
import jax
import jax.numpy as jnp
import env

# %%
#Initialize N_t, N_t(a)
t = 1
n_steps_a = jnp.zeros(10)
key = jax.random.key(0)

#I should vmap through environment to create baseline. But then what? hmmmm. what function would i even pass into my environment
#This is (t, N, Q)
state = (  t,  jnp.zeros(10), jnp.zeros(10),key)
c = 0.5
mu = jnp.array([-1.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.0, 0.4, -0.8])

def pick_every_lever(state):
        mask = (state[1] ==0)
        f_index = mask.argmax()
        return f_index


#(t,N,Q,key)
def UCB_agent(state):
    #right side ucb
    r_s =  c * ((jnp.log(state[0]) / state[1]) ** 0.5)
    return (state[2] + r_s).argmax()




