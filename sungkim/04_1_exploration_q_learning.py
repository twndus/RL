# inspired by lab 3
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# define argmax function by policy: greedy but random for ties
def randargmax(vector: np.array):
    return np.random.choice(np.where(vector == vector.max())[0])

def e_greedy(vector, epsilon=0.1):
    '''
        1. set epsilon
        2. choose random rate p
        3. p <= e: random action regardless of Q
           p >  e: greedy action
    '''
    p = np.random.rand()
    if p <= epsilon:
        action = np.random.randint(4)
    else:
        action = randargmax(vector)
    return action

def decaying_e_greedy(vector, iteration, epsilon=0.1):
    '''
        1. set epsilon, but epsilon will be decayed by iteration
        2. choose random rate p
        3. p <= e: random action regardless of Q
           p >  e: greedy action
    '''
    p = np.random.rand()
    if p <= epsilon/(1+iteration):
        action = np.random.randint(4)
    else:
        action = randargmax(vector)
    return action

def add_random_noise(vector):
    '''
        1. make random noise
        2. add random noise to current Q
        3. greedy
    '''
    noise = np.random.rand(vector.shape[0])
    action = randargmax(vector + noise)
    return action

def add_decaying_random_noise(vector, iteration):
    '''
        1. make random noise, but they will be decayed by iteration
        2. add random noise to current Q
        3. greedy
    '''
    noise = np.random.rand(vector.shape[0])/(1+iteration)
    action = randargmax(vector + noise)
    return action

def main():
    # create environment
    env = gym.make(
        'FrozenLake-v1',
        # render_mode="human", 
        is_slippery=False)#, render_mode="human", is_slippery=False)

    # exploration method
    exp_method = 'decaying_e_greedy' # plain, e_greedy, decaying_e_greedy, add_random_noise, add_decaying_random_noise

    # Q table (obs, actions: left, down, right, up)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # execute episodes
    num_episodes = 2000
    reward_list = []

    for i in tqdm(range(num_episodes)):

        # reset and observe current state
        state, _ = env.reset()
        done = False
        
        while not done:
            # choose actions by q table
            if exp_method == 'e_greedy':
                action = e_greedy(Q[state,:])
            elif exp_method == 'decaying_e_greedy':
                action = decaying_e_greedy(Q[state,:], i)
            elif exp_method == 'add_random_noise':
                action = add_random_noise(Q[state,:])
            elif exp_method == 'add_decaying_random_noise':
                action = add_decaying_random_noise(Q[state,:], i)
            else:
                action = randargmax(Q[state,:])

            # transition to next state
            next_state, reward, done, _, _ = env.step(action)

            # update Q 
            Q[state, action] = reward + np.max(Q[next_state, :])

            # transit to next state
            state = next_state
            
        reward_list.append(reward)

    print(f'success rate: {sum(reward_list)/num_episodes*100}')
    print(f'Q table: {Q}')

    plt.bar(range(num_episodes), reward_list) 
    plt.show()

if __name__ == '__main__':
    main()
