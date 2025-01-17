# inspired by lecture 2: playing OpenAI Gym Games
import gymnasium as gym

def main():
#    env = gym.make("FrozenLake-v1", render_mode="human")
#    observation, info = env.reset(seed=42)
#    env.render()
    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()

if __name__ == '__main__':
    main()
