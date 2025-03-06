import gymnasium as gym
import numpy as np
import torch as tc

from torch.optim import Adam

from gymnasium.vector import SyncVectorEnv

from ppo_algorithm import ActorCritic, Buffer
from ppo_algorithm.train import train_policy_da, ppo_train_step

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 250000
N_ACTORS = 1
STEPS = 500
GAMMA = 0.99
GAE_COEFFICIENT = 0.95
NORMALIZE_ADVANTAGE = True
BATCH_SIZE = 64
MAX_GRADIENT_NORM = 0.5
LEARNING_RATE = 10**-3
N_EPOCHS_PER_EPISODE = 6
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
KL_TARGET = None
DEVICE = tc.device("cpu")

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    #Create enviroment.
    envs = gym.make_vec("LunarLander", num_envs=N_ACTORS, vectorization_mode="sync")
    OBSERVATION_SIZE = envs.single_observation_space.shape[0]
    ACTION_SIZE = envs.single_action_space.n

    #Create ActorCritic net.
    model = ActorCritic(OBSERVATION_SIZE, ACTION_SIZE, 256, 1, 1, 1)

    #Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create buffer.
    buffer = Buffer(STEPS, N_ACTORS, OBSERVATION_SIZE, 1, act_dtype=tc.int32, device=DEVICE)

    #Training phase.
    scores = []
    total_states = 0
    obs, _ = envs.reset()

    while total_states <= TARGET_TOTAL_STEPS:
        for _ in range(STEPS):
            obs = tc.from_numpy(obs).to(device=DEVICE)

            #Choose action and compute log probability
            with tc.no_grad():
                action, value, action_dist = train_policy_da(model, obs)
                log_prob = action_dist.log_prob(action)

                #Reshape dimensions
                action = action.reshape(-1)
                value = value.reshape(-1)

            #Perform action chosen.
            next_obs, r, terminated, truncation, _ = envs.step(action.cpu().to(dtype=tc.int32).numpy())

            done = tc.from_numpy(np.logical_or(terminated, truncation)).to(dtype=np.int32, device=DEVICE)
            reward = tc.from_numpy(r).to(device=DEVICE)

            #Store one step infos into buffer.
            buffer.store(obs, action, log_prob, reward, done, value)

            #Next observation.
            obs = next_obs
            total_states += 1

        ppo_train_step(model, 
                       train_policy_da, 
                       buffer, 
                       optimizer, 
                       norm_adv=NORMALIZE_ADVANTAGE, 
                       n_epochs=N_EPOCHS_PER_EPISODE, 
                       batch_size=BATCH_SIZE, 
                       max_grad_norm=MAX_GRADIENT_NORM,
                       clip_range=CLIP_RANGE,
                       kl_target=KL_TARGET,
                       value_coeff=VALUE_COEFFICIENT,
                       entr_coeff=ENTROPY_COEFFICIENT)

    envs.close()