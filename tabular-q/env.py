import jax 
import jax.numpy as jnp

#policy is column index of choice, state integer representation of position: 0 - 8.
# policy is 0 -3. up, down, left, right

#Responsible for taking steps. If out of bounds, sends back 
def gridworld(state, policy):
    #Converting int state to (x,y) state
    y = state//3
    x = state%3
    dx = jnp.array([0,0,1,-1])
    dy = jnp.array([1,-1,0,0])

    x = x+dx[policy]
    y = y+dy[policy]

    x = jnp.clip(x,0,2)
    y = jnp.clip(y,0,2)

    state = (x + (y*3))

    conditions = [state == 8, state ==4]
    rewards = [100, -100]
    reward = jnp.select(conditions, rewards, default=-1)
    return state, reward
