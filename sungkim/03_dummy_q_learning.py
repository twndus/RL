# inspired by lab 3
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

def main():
    # create environment
    env = gym.make('FrozenLake-v1', is_slippery=False)#, render_mode="human", is_slippery=False)

    # Q table (obs, actions: left, down, right, up)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # define argmax function by policy: greedy but random for ties
    def randargmax(vector: np.array):
        return np.random.choice(np.where(vector == vector.max())[0])

    # execute episodes
    num_episodes = 2000
    reward_list = []

    for i in tqdm(range(num_episodes)):

        # reset and observe current state
        state, _ = env.reset()
        done = False
        
        while not done:
            # choose actions by q table
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
