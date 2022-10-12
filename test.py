import gym
import d4rl # Import required to register environments
import numpy as np

def create_init_action(sample):
    dims = len(sample)
    return np.random.uniform(-1, 1, size=(dims,))


# Create the environment
env = gym.make('kitchen-complete-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
print("Actions space: ",env.action_space)
init_action = create_init_action(env.action_space.sample())
action = init_action
for i in range(120):
    action[7] += 0.01
    #print("Actions space: ",env.action_space.sample())
    env.step(action=action)
    env.render('human')

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
'''dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)'''