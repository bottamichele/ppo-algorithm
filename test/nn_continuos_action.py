import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers import  NumpyToTorch

from ppo_algorithm import PPOTrainAgent, ActionSpace, NN

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

EPISODES = 1000
CLIP_RANGE = 0.2
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 10**-3
NORMALIZE_ADVANTAGE = True
N_EPOCHS_PER_EPISODE = 8
ENTROPY_COEFFICIENT = 0.001
KL_COEFFICIENT = 0.0
KL_TARGET = 0.0
DEVICE = tc.device("cpu")

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    #Create enviroment.
    env = gym.make("LunarLander-v3", continuous=True)
    env = NumpyToTorch(env, device=DEVICE)
    OBSERVATION_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.shape[0]

    #Create agent.
    actor = NN(OBSERVATION_SIZE, ACTION_SIZE, 256, 2)
    critic = NN(OBSERVATION_SIZE, 1, 256, 2)
    agent = PPOTrainAgent(actor, 
                          critic, 
                          ActionSpace.CONTINUOUS, 
                          lr=LEARNING_RATE, 
                          gamma=GAMMA, 
                          batch_size=BATCH_SIZE, 
                          n_epochs=N_EPOCHS_PER_EPISODE, 
                          norm_adv=NORMALIZE_ADVANTAGE, 
                          entropy_coeff=ENTROPY_COEFFICIENT,
                          kl_coeff=KL_COEFFICIENT,
                          kl_target=KL_TARGET, 
                          device=DEVICE)

    #Training phase.
    scores = []
    total_states = 0

    for episode in range(1, EPISODES+1):
        #Episode.
        obs, _ = env.reset()
        epis_done = False
        scores.append(0)
        
        while not epis_done:
            #Choose action.
            action, action_dist = agent.choose_action(obs)

            #Perform action choosen.
            next_obs, reward, terminated, truncated, _ = env.step(action)   
            epis_done = terminated or truncated

            #Store a trajectory's item.
            agent.trajectory.store(obs, action, action_dist, reward)
                    
            #Updates.
            obs = next_obs
            scores[-1] += reward
            total_states += 1

        #Train step.
        agent.train()

        #Print training stats of current epidode ended.
        print("- Episode {:3d}: score = {:3.1f}; avg score = {:3.2f}; total states = {:>5d}".format(episode, scores[-1], np.mean(scores[-100:]), total_states))

    env.close()