import jax 
import jax.numpy as jnp

#policy is column index of choice, state integer representation of position: 0 - 8.
# policy is 0 -3. up, down, left, right

#Responsible for taking steps. If out of bounds, sends back 
def gridworld(state, policy):
    #Converting int state to (x,y) state
    y = state//15
    x = state%15
    dx = jnp.array([0,0,1,-1])
    dy = jnp.array([1,-1,0,0])

    x = x+dx[policy]
    y = y+dy[policy]

    x = jnp.clip(x,0,14)
    y = jnp.clip(y,0,14)

    state = (x + (y*15))

    conditions = [state == 224, 
                  jnp.isin(state,jnp.array([31,107,63, 154, 20, 186, 112, 32, 144, 55, 176, 87,208]))]
    rewards = [100, -100]
    reward = jnp.select(conditions, rewards, default=-1)
    return state, reward
