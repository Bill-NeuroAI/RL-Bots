import jax 
import jax.numpy as jnp

#policy is column index of choice, state integer representation of position: 0 - 8.
# policy is 0 -3. up, down, left, right

#Responsible for taking steps. If out of bounds 
def state_step(state, policy):
    #Converting int state to (x,y) state
    y = jnp.floor(state/3)
    x = state%3

    #checks policy and acts accordingly
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

    #returns new state
    return  (x + (y*3))



def reward_function(state):
    reward = -1
    if (state == 8):
        return 100, True
    elif (state == 4):
        return -100, True
    else:
        return -1, False




