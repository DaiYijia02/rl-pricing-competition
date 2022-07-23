import argparse
import os
import numpy as np

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
# Use torch.
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
    # use an agent to get the observation space and the action space
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

    # Only two agents, first is ppo, second is fixed price.
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "fixed_price"

    # Defining the printout in the console after each step.
    def on_postprocess_traj(info):
        """
        arg: {"agent_id": ..., "episode": ...,
            "pre_batch": (before processing),
            "post_batch": (after processing),
            "all_pre_batches": (other agent ids),
            }

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
        print('actions = {}'.format(np.multiply(
            sample_obj.columns(["actions"]), 5)+5))
        print('reward= {}'.format(sample_obj.columns(["rewards"])))
        return

    # The call to train the agent.
    tune.run("PPO",
             config={"env": "multi_agent_pricing_competition",
                     "framework": "torch",
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
        raise ValueError(
            "Desired reward ({}) not reached!".format(args.stop_reward))
