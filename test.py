from pettingzoo.test import api_test
from pettingzoo.test import parallel_test
from pettingzoo.test import render_test
import rl_price_competition_multi_agent_env
import numpy as np
import warnings
from collections import defaultdict
import random
import gym
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO


env = rl_price_competition_multi_agent_env.env()

parallel_test.parallel_api_test(env)
# api_test(env, num_cycles=1000, verbose_progress=False)
# render_test(env)

# env.reset()
# agents = env.agents
# print(agents)
# env.reset()
# select = env.agent_selection
# print(select)
# print(env.last())
# env.step(action=[[1,2],[2,2]])
# print(env.last())
# print(new_obs)
# select = env.agent_selection
# print(select)

# model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
# model.learn(total_timesteps=2000000)
# model.save("policy")

# model = PPO.load("policy")

# env.reset()
# for agent in env.agent_iter():
#    obs, reward, done, info = env.last()
#    act = model.predict(obs, deterministic=True)[0] if not done else None
#    env.step(act)
#    env.render()

def test_observation(observation, observation_0):
    if isinstance(observation, np.ndarray):
        if np.isinf(observation).any():
            warnings.warn(
                "Observation contains infinity (np.inf) or negative infinity (-np.inf)"
            )
        if np.isnan(observation).any():
            warnings.warn("Observation contains NaNs")
        if len(observation.shape) > 3:
            warnings.warn("Observation has more than 3 dimensions")
        if observation.shape == (0,):
            assert False, "Observation can not be an empty array"
        if observation.shape == (1,):
            warnings.warn("Observation is a single number")
        if not isinstance(observation, observation_0.__class__):
            warnings.warn("Observations between agents are different classes")
        if (observation.shape != observation_0.shape) and (
            len(observation.shape) == len(observation_0.shape)
        ):
            warnings.warn("Observations are different shapes")
        if len(observation.shape) != len(observation_0.shape):
            warnings.warn("Observations have different number of dimensions")
        if not np.can_cast(observation.dtype, np.dtype("float64")):
            warnings.warn("Observation numpy array is not a numeric dtype")
        if np.array_equal(observation, np.zeros(observation.shape)):
            warnings.warn("Observation numpy array is all zeros.")
        if not np.all(observation >= 0) and (
            (len(observation.shape) == 2)
            or (len(observation.shape) == 3 and observation.shape[2] == 1)
            or (len(observation.shape) == 3 and observation.shape[2] == 3)
        ):
            warnings.warn(
                "The observation contains negative numbers and is in the shape of a graphical observation. This might be a bad thing."
            )
    else:
        warnings.warn("Observation is not NumPy array")

def test_reward(reward):
    if (
        not (isinstance(reward, int) or isinstance(reward, float))
        and not isinstance(np.dtype(reward), np.dtype)
        and not isinstance(reward, np.ndarray)
    ):
        warnings.warn("Reward should be int, float, NumPy dtype or NumPy array")
    if isinstance(reward, np.ndarray):
        if isinstance(reward, np.ndarray) and not reward.shape == (1,):
            assert False, "Rewards can only be one number"
        if np.isinf(reward):
            assert False, "Reward must be finite"
        if np.isnan(reward):
            assert False, "Rewards cannot be NaN"
        if not np.can_cast(reward.dtype, np.dtype("float64")):
            assert False, "Reward NumPy array is not a numeric dtype"

def play_test(env, observation_0, num_cycles):
    """
    plays through environment and does dynamic checks to make
    sure the state returned by the environment is
    consistent. In particular it checks:
    * Whether the reward returned by last is the accumulated reward
    * Whether the agents list shrinks when agents are done
    * Whether the keys of the rewards, dones, infos are equal to the agents list
    * tests that the observation is in bounds.
    """
    env.reset()

    done = {agent: False for agent in env.agents}
    live_agents = set(env.agents[:])
    has_finished = set()
    generated_agents = set()
    accumulated_rewards = defaultdict(int)
    for agent in env.agent_iter(env.num_agents * num_cycles):
        generated_agents.add(agent)
        assert (
            agent not in has_finished
        ), "agents cannot resurect! Generate a new agent with a new name."
        assert isinstance(
            env.infos[agent], dict
        ), "an environment agent's info must be a dictionary"
        prev_observe, reward, done, info = env.last()
        if done:
            action = None
        elif isinstance(prev_observe, dict) and "action_mask" in prev_observe:
            action = random.choice(np.flatnonzero(prev_observe["action_mask"]))
        else:
            action = env.action_space(agent).sample()
        
        if agent not in live_agents:
            live_agents.add(agent)

        assert live_agents.issubset(
            set(env.agents)
        ), "environment must delete agents as the game continues"

        if done:
            live_agents.remove(agent)
            has_finished.add(agent)

        assert (
            accumulated_rewards[agent] == reward
        ), "reward returned by last is not the accumulated rewards in its rewards dict"
        accumulated_rewards[agent] = 0

        env.step(action)

        for a, rew in env.rewards.items():
            accumulated_rewards[a] += rew

        assert env.num_agents == len(
            env.agents
        ), "env.num_agents is not equal to len(env.agents)"
        assert set(env.rewards.keys()) == (
            set(env.agents)
        ), "agents should not be given a reward if they were done last turn"
        assert set(env.dones.keys()) == (
            set(env.agents)
        ), "agents should not be given a done if they were done last turn"
        assert set(env.infos.keys()) == (
            set(env.agents)
        ), "agents should not be given an info if they were done last turn"
        if hasattr(env, "possible_agents"):
            assert set(env.agents).issubset(
                set(env.possible_agents)
            ), "possible agents should always include all agents, if it exists"

        if not env.agents:
            break

        if isinstance(env.observation_space(agent), gym.spaces.Box):
            assert env.observation_space(agent).dtype == prev_observe.dtype
        assert env.observation_space(agent).contains(
            prev_observe
        ), "Out of bounds observation: " + str(prev_observe)

        assert env.observation_space(agent).contains(
            prev_observe
        ), "Agent's observation is outside of it's observation space"
        test_observation(prev_observe, observation_0)
        if not isinstance(env.infos[env.agent_selection], dict):
            warnings.warn(
                "The info of each agent should be a dict, use {} if you aren't using info"
            )

    if not env.agents:
        assert (
            has_finished == generated_agents
        ), "not all agents finished, some were skipped over"

    env.reset()
    for agent in env.agent_iter(env.num_agents * 2):
        obs, reward, done, info = env.last()
        if done:
            action = None
        elif isinstance(obs, dict) and "action_mask" in obs:
            action = random.choice(np.flatnonzero(obs["action_mask"]))
        else:
            action = env.action_space(agent).sample()
        assert isinstance(done, bool), "Done from last is not True or False"
        assert (
            done == env.dones[agent]
        ), "Done from last() and dones[agent] do not match"
        assert (
            info == env.infos[agent]
        ), "Info from last() and infos[agent] do not match"
        float(
            env.rewards[agent]
        )  # "Rewards for each agent must be convertible to float
        test_reward(reward)
        observation = env.step(action)
        assert observation is None, "step() must not return anything"

# env.reset()
# observation_0, _, _, _ = env.last()
# test_observation(observation_0, observation_0)

# non_observe, _, _, _ = env.last(observe=False)
# assert non_observe is None, "last must return a None when observe=False"

# print("Finished test_observation")

# play_test(env, observation_0, 10)

# print("Finished play test")