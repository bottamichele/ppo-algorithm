import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers.vector import RecordEpisodeStatistics

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net.nn import NNActorCriticDiscrete
from ppo_algorithm.agent import PPOAgent

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 20000
N_ACTORS = 6
N_STEPS = 256
BATCH_SIZE = 64
LEARNING_RATE = 10**-3
N_EPOCHS = 6
DEVICE = tc.device("cpu")

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    #Create enviroment.
    envs = gym.make_vec("CartPole-v1", num_envs=N_ACTORS, vectorization_mode="async")
    envs = RecordEpisodeStatistics(envs, buffer_length=1)
    OBSERVATION_SIZE = envs.single_observation_space.shape[0]
    ACTION_SIZE = envs.single_action_space.n

    #Create agent.
    agent = PPOAgent(NNActorCriticDiscrete(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1).to(device=DEVICE),
                     Rollout(N_STEPS, N_ACTORS, (OBSERVATION_SIZE,), (), act_dtype=tc.int32, device=DEVICE),
                     LEARNING_RATE,
                     BATCH_SIZE,
                     N_EPOCHS,
                     DEVICE)

    #Training phase.
    total_states = 0
    obs, infos = envs.reset()
    done = np.zeros(N_ACTORS, dtype=np.int32)

    print("==================================================")

    while total_states <= TARGET_TOTAL_STEPS:
        for _ in range(N_STEPS):
            #Choose action.
            action, value, log_prob = agent.choose_action(tc.Tensor(obs).to(device=DEVICE))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, infos = envs.step(action.cpu().numpy())

            #Store one step infos into rollout.
            agent.remember(tc.Tensor(obs).to(device=DEVICE), 
                          action, 
                          log_prob, 
                          tc.Tensor(reward).to(device=DEVICE), 
                          tc.Tensor(done).to(device=DEVICE), 
                          value.reshape(-1))

            #Next observation.
            obs = next_obs
            done = np.logical_or(terminated, truncation)
            total_states += 1

            if "episode" in infos:
                print("- state = {:>6d}; cum. reward = {}".format(total_states, infos["episode"]["r"][infos["_episode"]]))

        #Train step.
        agent.train(tc.Tensor(obs).to(device=DEVICE), tc.Tensor(done).to(device=DEVICE))

    envs.close()