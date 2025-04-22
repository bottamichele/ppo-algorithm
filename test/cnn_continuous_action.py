import gymnasium as gym
import numpy as np
import torch as tc

from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv

from torch.optim import Adam

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net.cnn import CnnActorCriticContinuous
from ppo_algorithm.training import ppo_train_step

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_FRAMES = 100000
FRAME_STACK = 1
N_ACTORS = 6
N_STEPS = 512
GAMMA = 0.99
GAE_COEFFICIENT = 0.95
NORMALIZE_ADVANTAGE = True
BATCH_SIZE = 64
MAX_GRADIENT_NORM = 0.5
LEARNING_RATE = 10**-4
N_EPOCHS = 6
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
KL_TARGET = None
SHOW_ACTION_STD = True
DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")

# ========================================
# ================= MAIN =================
# ========================================

def make_env(env_name):
    env = gym.make(env_name)
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=FRAME_STACK, padding_type="zero")
    return env

if __name__ == "__main__":
    #Create enviroment.
    envs = AsyncVectorEnv([lambda:make_env("CarRacing-v3") for _ in range(N_ACTORS)])
    envs = RecordEpisodeStatistics(envs, buffer_length=1)
    OBSERVATION_SIZE = envs.single_observation_space.shape
    ACTION_SIZE = envs.single_action_space.shape[0]

    #Create ActorCritic net.
    model = CnnActorCriticContinuous(obs_size=OBSERVATION_SIZE, action_size=ACTION_SIZE).to(device=DEVICE)

    #Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #Create rollout.
    rollout = Rollout(N_STEPS, N_ACTORS, OBSERVATION_SIZE, (ACTION_SIZE,), obs_dtype=tc.uint8, device=DEVICE)

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


    env = gym.make("CarRacing-v3", render_mode="human")
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=FRAME_STACK, padding_type="zero")

    for e in range(500):
        obs, _ = env.reset()
        done = False
        r = 0

        while not done:
            action, _, _, _ = model.action_and_value(tc.Tensor(obs).unsqueeze(0).to(device=DEVICE))

            next_obs, reward, terminated, truncation, _ = env.step(action.squeeze(0).cpu().numpy())

            obs = next_obs
            done = terminated or truncation
            r += reward

        print(f"- Episode {e+1}: score = {r}")