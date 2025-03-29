import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv

from torch.optim import Adam

from ppo_algorithm import CnnActorCritic, Buffer
from ppo_algorithm.train import train_policy_da, ppo_train_step

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_FRAMES = 200000
FRAME_STACK = 4
N_ACTORS = 8
N_STEPS = 250
GAMMA = 0.99
GAE_COEFFICIENT = 0.95
NORMALIZE_ADVANTAGE = False
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
    model = CnnActorCritic(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(device=DEVICE)

    #Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create buffer.
    buffer = Buffer(N_STEPS, N_ACTORS, OBSERVATION_SIZE, 1, obs_dtype=tc.uint8, act_dtype=tc.int32, device=DEVICE)

    #Training phase.
    total_frames = 0
    obs, infos = envs.reset()
    done = np.zeros(N_ACTORS, dtype=np.int32)

    while total_frames <= TARGET_TOTAL_FRAMES:
        for _ in range(N_STEPS):
            obs = tc.from_numpy(obs).to(device=DEVICE)
            done = tc.from_numpy(done).to(dtype=tc.int32, device=DEVICE)

            #Choose action and compute log probability
            with tc.no_grad():
                action, value, action_dist = train_policy_da(model, obs)
                log_prob = action_dist.log_prob(action)

            #Perform action chosen.
            next_obs, r, terminated, truncation, infos = envs.step(action.reshape(-1).cpu().to(dtype=tc.int32).numpy())
            reward = tc.from_numpy(r).to(device=DEVICE)

            #Store one step infos into buffer.
            buffer.store(obs, action, log_prob, reward, done, value.reshape(-1))

            #Next observation.
            obs = next_obs
            done = np.logical_or(terminated, truncation)
            total_frames += 1

            if "episode" in infos:
                print("- frame = {:>6d}; cum. reward = {}".format(total_frames, infos["episode"]["r"][infos["_episode"]]))

        #Compute advantages and returns.
        with tc.no_grad():
            _, value = model(tc.from_numpy(obs).to(device=DEVICE))
        buffer.compute_advantage_and_return(value.reshape(-1), tc.from_numpy(done).to(dtype=tc.int32, device=DEVICE))

        #Train step.
        ppo_train_step(model, 
                       train_policy_da, 
                       buffer, 
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