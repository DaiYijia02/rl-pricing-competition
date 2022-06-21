"""
From here mostly: https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py
"""
from settings import *


import algopricing.MultiAgentEnv_algopricing as MultiAgentEnv_algopricing
from algopricing.MultiAgentEnv_algopricing import MultiAgentEnv_algopricing


def make_env_agents(agentnames, params=default_params, first_file=None, second_file=None, third_file=None):
    import agents
    agentslist = []

    for en, name in enumerate(agentnames):
        ag = agents.load(name + ".py").Agent(en, params)
        agentslist.append(ag)
        try:
            ag = agents.load(name + ".py").Agent(en, params)
            agentslist.append(ag)
        except Exception as e:
            print('competitioncodeprefixprinting: ',
                  'error creating agent: ', name, e)
    assert len(
        agentslist) == 2, "competitioncodeprefixprinting: Skipping game, at least one agent errored out"

    env = MultiAgentEnv_algopricing(
        params, agentnames, len(
            agentnames), first_file, second_file, third_file
    )
    # customer_covariates_file=None, customer_noisyembeddings_file=None, customer_valuations_file=None
    return env, agentslist