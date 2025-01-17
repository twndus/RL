# inspired by lab 3
import numpy as np
import gymnasium as gym

def main():
    # create environment
    env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)

    # reset and observe current state
    state, _ = env.reset()

#    # print action_space
#    print(env.action_space.n)
#
#    # print observation_space
#    print(env.observation_space)

    # Q table (obs, actions: left, down, right, up)
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    print((env.observation_space.n, env.action_space.n))

    # define argmax function by policy: greedy but random for ties
    def randargmax(state: int):
        return np.random.choice(np.where(qtable[state,:] == qtable[state,:].max())[0])

    # execute episode
    for _ in range(1000):
        # env.render()
        
#        # choose actions randomly
#        action = env.action_space.sample()

        # choose actions by q table
        action = randargmax(state)

        # transition to next state
        next_state, reward, done, _, _ = env.step(action)

        # update qtable
        qtable[state, action] = reward + np.max(qtable[next_state, :])

        # transit to next state
        state = next_state

        if done:
            state, _ = env.reset()

    print(qtable)

if __name__ == '__main__':
    main()
