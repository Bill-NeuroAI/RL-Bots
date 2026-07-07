

# %%
import jax
import jax.numpy as jnp
import k_armed_bandits.env as env

#state is a tuple with (t,N,Q,key)

def pick_every_lever(state):
        mask = (state[1] ==0)
        f_index = mask.argmax()
        return f_index


#(t,N,Q,key)
def UCB_agent(state,c):
    #right side ucb
    r_s =  c * ((jnp.log(state[0]) / state[1]) ** 0.5)
    return (state[2] + r_s).argmax()

      


