import gymnasium as gym
import numpy as np
import torch as tc

from torch.optim import Adam

from gymnasium.wrappers.vector import RecordEpisodeStatistics

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net.nn import NNActorCriticContinuous
from ppo_algorithm.training import  ppo_train_step

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 1_000_000
N_ACTORS = 1
N_STEPS = 2048
GAMMA = 0.99
GAE_COEFFICIENT = 0.95
NORMALIZE_ADVANTAGE = True
BATCH_SIZE = 64
MAX_GRADIENT_NORM = 0.5
LEARNING_RATE = 3*10**-4
N_EPOCHS = 10
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
KL_TARGET = None
SHOW_ACTION_STD = True
DEVICE = tc.device("cpu")

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    #Create enviroment.
    envs = gym.make_vec("Ant-v4", num_envs=N_ACTORS, vectorization_mode="async")
    envs = RecordEpisodeStatistics(envs, 1)
    OBSERVATION_SIZE = envs.single_observation_space.shape[0]
    ACTION_SIZE = envs.single_action_space.shape[0]

    #Create ActorCritic net.
    model = NNActorCriticContinuous(OBSERVATION_SIZE, ACTION_SIZE, 64, 1, 2, 2).to(device=DEVICE)

    #Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create rollout.
    rollout = Rollout(N_STEPS, N_ACTORS, (OBSERVATION_SIZE,), (ACTION_SIZE,), device=DEVICE)

    #Training phase.
    total_states = 0
    obs, infos = envs.reset()
    done = np.zeros(N_ACTORS, dtype=np.int32)

    print("==================================================")

    while total_states <= TARGET_TOTAL_STEPS:
        for _ in range(N_STEPS):
            #Choose action.
            with tc.no_grad():
                action, value, log_prob, _ = model.action_and_value(tc.Tensor(obs).to(device=DEVICE))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, infos = envs.step(action.cpu().numpy())

            #Store one step infos into rollout.
            rollout.store(tc.Tensor(obs).to(device=DEVICE), 
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

        if SHOW_ACTION_STD:
            print("ACTION STD = {}".format(model._action_logstd.detach().exp()))        

        #Compute advantages and returns.
        with tc.no_grad():
            last_value = model.value(tc.Tensor(obs).to(device=DEVICE))
            rollout.compute_advantages_and_returns(last_value.reshape(-1), tc.Tensor(done).to(device=DEVICE))

        #Train step.
        ppo_train_step(model,
                       rollout, 
                       optimizer, 
                       norm_adv=NORMALIZE_ADVANTAGE, 
                       n_epochs=N_EPOCHS, 
                       batch_size=BATCH_SIZE, 
                       max_grad_norm=MAX_GRADIENT_NORM,
                       clip_range=CLIP_RANGE,
                       kl_target=KL_TARGET,
                       value_coeff=VALUE_COEFFICIENT,
                       entr_coeff=ENTROPY_COEFFICIENT)

    envs.close()