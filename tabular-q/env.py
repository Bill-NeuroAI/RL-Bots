# %%
import jax 
import jax.numpy as jnp

# %%
#policy is column index of choice, state integer representation of position: 0 - 8.
# policy is 0 -3. up, down, left, right
def grid_world(state, policy):
    y = jnp.floor(state/3)
    x = state%3
    #reward remains normal, even in out of bounds cases
    reward = -1
    match policy:
        #up
        case 0:
            y = y + 1
        #down
        case 1:
            y = y - 1
        #left
        case 2:
            x = x - 1
        #right
        case 3:
            x = x + 1
    
    #checking for out of bounds
    if (x > 2):
        x = 2
    if (y > 2):
        y = 2
    if(x < 0):
        x = 0
    if(y < 0):
        y = 0

    if ((x==2) and (y == 1)):
        done = True
        reward = -10
    if ((x == 2) and (y==2)):
        done = True
        reward = 10

    state = (x + (y*3))


    return state, reward




