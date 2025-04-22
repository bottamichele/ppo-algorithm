import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv

from torch.optim import Adam

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net.cnn import CnnActorCriticDiscrete
from ppo_algorithm.training import  ppo_train_step

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_FRAMES = 200000
FRAME_STACK = 4
N_ACTORS = 8
N_STEPS = 256
GAMMA = 0.99
GAE_COEFFICIENT = 0.95
NORMALIZE_ADVANTAGE = True
BATCH_SIZE = 64
MAX_GRADIENT_NORM = 0.5
LEARNING_RATE = 10**-4
N_EPOCHS = 4
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
KL_TARGET = None
DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")

# ========================================
# ================= MAIN =================
# ========================================

def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = TransformObservation(env, lambda x:env.render(), Box(low=0, high=255, shape=(400, 600, 3), dtype=np.uint8))
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=FRAME_STACK, padding_type="zero")
    
    return env

if __name__ == "__main__":
    #Create enviroment.
    envs = AsyncVectorEnv([lambda:make_env("CartPole-v1") for _ in range(N_ACTORS)])
    envs = RecordEpisodeStatistics(envs, buffer_length=1)
    OBSERVATION_SIZE = envs.single_observation_space.shape
    ACTION_SIZE = envs.single_action_space.n

    #Create ActorCritic net.
    model = CnnActorCriticDiscrete(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(device=DEVICE)

    #Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create rollout.
    rollout = Rollout(N_STEPS, N_ACTORS, OBSERVATION_SIZE, (), obs_dtype=tc.uint8, act_dtype=tc.int32, device=DEVICE)

    #Training phase.
    total_frames = 0
    obs, infos = envs.reset()
    done = np.zeros(N_ACTORS, dtype=np.int32)

    print("==================================================")

    while total_frames <= TARGET_TOTAL_FRAMES:
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
            total_frames += 1

            if "episode" in infos:
                print("- frame = {:>6d}; cum. reward = {}".format(total_frames, infos["episode"]["r"][infos["_episode"]]))

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