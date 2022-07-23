"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two algorithms (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse
import gym
import os

import ray
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
import multi_agent_pricing_env
from multi_agent_pricing_env import MultiPriceCompetitionEnv
import fixed_price_policy
from fixed_price_policy import FixedPricePolicy
from reporter import CustomReporter
from ray import tune
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

parser = argparse.ArgumentParser()
# Use torch for both policies.
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=10, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=50.0, help="Reward at which we stop training."
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    register_env(
        "multi_agent_pricing_competition", lambda _: MultiPriceCompetitionEnv()
    )
    env = multi_agent_pricing_env.env()
    agent = env.sellers[0]
    obs_space = agent.observation_space
    act_space = agent.action_space


    def seelct_policy(algorithm, framework=None):
        if algorithm == "PPO":
            if framework == "torch":
                return PPOTorchPolicy
            elif framework == "tf":
                return PPOTF1Policy
            else:
                return PPOTF2Policy
        elif algorithm == "FIX":
            return FixedPricePolicy
        else:
            raise ValueError("Unknown algorithm: ", algorithm)

    # You can also have multiple policies per algorithm.
    policies = {
        "ppo_policy": (
            seelct_policy("PPO", args.framework),
            obs_space,
            act_space,
            {},
        ),
        "fixed_price": (
            seelct_policy("FIX"),
            obs_space,
            act_space,
            {},
        )
    }

    #  Now only using ppo
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "fixed_price"

    def on_postprocess_traj(info):
        """
        arg: {"agent_id": ..., "episode": ...,
            "pre_batch": (before processing),
            "post_batch": (after processing),
            "all_pre_batches": (other agent ids),
            }

        # https://github.com/ray-project/ray/blob/ee8c9ff7320ec6a2d7d097cd5532005c6aeb216e/rllib/policy/sample_batch.py
        Dictionaries in a sample_obj, k:
            t
            eps_id
            agent_index
            obs
            actions
            rewards
            prev_actions
            prev_rewards
            dones
            infos
            new_obs
            action_prob
            action_logp
            vf_preds
            behaviour_logits
            unroll_id       
        """
        agt_id = info["agent_id"]
        eps_id = info["episode"].episode_id
        policy_obj = info["pre_batch"][0]
        sample_obj = info["pre_batch"][1]

        print('agent_id = {}'.format(agt_id))
        print('episode = {}'.format(eps_id))
        print('policy = {}'.format(policy_obj))
        print('actions = {}'.format(sample_obj.columns(["actions"])))
        print('reward= {}'.format(sample_obj.columns(["rewards"])))

        return


    # ppo = PPO(
    #     env="multi_agent_pricing_competition",
    #     config={
    #         "multiagent": {
    #             "policies": policies,
    #             "policy_mapping_fn": policy_mapping_fn,
    #             "policies_to_train": ["ppo_policy"],
    #         },
    #         "model": {
    #             "vf_share_layers": True,
    #         },
    #         "num_sgd_iter": 6,
    #         "vf_loss_coeff": 0.01,
    #         # disable filters, otherwise we would need to synchronize those
    #         # as well to the DQN agent
    #         "observation_filter": "MeanStdFilter",
    #         # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #         "framework": args.framework,
    #     },
    # )


    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     ppo_policy: Y
    # for i in range(args.stop_iters):
    #     print("== Iteration", i, "==")

    #     # improve the PPO policy
    #     print("-- PPO --")
    #     result_ppo = ppo.train()
    #     print(pretty_print(result_ppo))

    #     # Test passed gracefully.
    #     if (
    #         args.as_test
    #         # and result_dqn["episode_reward_mean"] > args.stop_reward
    #         and result_ppo["episode_reward_mean"] > args.stop_reward
    #     ):
    #         print("test passed (both agents above requested reward)")
    #         quit(0)

        # swap weights to synchronize
        # dqn.set_weights(ppo.get_weights(["ppo_policy"]))

    tune.run("PPO",
            config={"env":"multi_agent_pricing_competition",
                    "framework":"torch",
                    "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": policy_mapping_fn,
                    "policies_to_train": ["ppo_policy"],
                    },
                    "model": {
                        "vf_share_layers": True,
                    },
                    "num_sgd_iter": 50,
                    "vf_loss_coeff": 0.01,
                    "observation_filter": "MeanStdFilter",
                    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "framework": args.framework,
                    "callbacks": {
                        "on_postprocess_traj": on_postprocess_traj,
                    }
                        },
            stop={"training_iteration": 20},
            local_dir="multi_agent_pricing_competition",
            progress_reporter=CustomReporter()
            )

    # Desired reward not reached.
    if args.as_test:
        raise ValueError("Desired reward ({}) not reached!".format(args.stop_reward))