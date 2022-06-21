# import gym
# from gym import spaces
# from gym.envs.registration import EnvSpec
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# Code adapted from here: https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py
# also useful: https://www.ai-articles.net/creating-a-custom-gym-openai-environment-for-algorithmic-trading/

def read_file(file_name_str):
    df = pd.read_csv(file_name_str)
    df.index = df['Unnamed: 0'].values
    del df['Unnamed: 0']
    return df

class MultiAgentEnv_algopricing(object):  # gym.Env
    def __init__(self, params, agent_names, n_agents=3, customer_covariates_file=None, customer_noisyembeddings_file=None, customer_valuations_file=None):
        self.time = 0
        self.cumulative_buyer_utility = 0
        self.n_items = params["n_items"]
        self.n_agents = n_agents
        self.agent_names = agent_names
        self.agent_profits = [0 for _ in range(self.n_agents)]
        self.agent_profits_timeseries = [[] for _ in range(self.n_agents)]
        self.customers = []
        self.customer_covariates_file = customer_covariates_file
        self.customer_noisyembeddings_file = customer_noisyembeddings_file
        self.customer_truevaluations_file = customer_valuations_file

        self.customer_covariates = None
        self.customer_noisyembeddings = None
        self.customer_truevaluations = None

        self._init_data_files()

    def _init_data_files(self):
        if self.customer_covariates_file is None:
            return
        else:
            self.customer_covariates = read_file(
                self.customer_covariates_file)
            self.customer_noisyembeddings = read_file(
                self.customer_noisyembeddings_file)
            self.customer_truevaluations = read_file(
                self.customer_truevaluations_file)

    def get_current_customer(self):
        assert self.time <= len(self.customers)
        if len(self.customers) == self.time:  # create new customer
            if self.customer_covariates is None:
                covariates = [0, 0, 0]
                valuations = [random.random() * 2 for _ in range(self.n_items)]
                noisyembedding = None
            else:
                customerindex = random.choice(
                    self.customer_covariates.index.values)
                covariates = self.customer_covariates.loc[customerindex].values
                if customerindex in self.customer_noisyembeddings.index.values:
                    noisyembedding = self.customer_noisyembeddings.loc[customerindex].values
                else:
                    noisyembedding = None
                valuations = [
                    self.customer_truevaluations.loc[customerindex, 'item{}valuations'.format(itemnum)] for itemnum in range(self.n_items)
                ]
            self.customers.append((covariates, noisyembedding, valuations))
        else:  # have alredy created the customer, just retreiving it
            covariates, noisyembedding, valuations = self.customers[self.time]
        return covariates, noisyembedding, valuations

    def get_current_state_customer_to_send_agents(self, sale=(np.nan, np.nan, [[np.nan, np.nan], [np.nan, np.nan]])):
        customer_covariates, customer_embedding, customer_valuations = self.get_current_customer()
        state = self.agent_profits
        return customer_covariates, customer_embedding, sale, state

    def step(self, all_agent_prices):
        eps = 1e-7  # for random tie-breaking

        # process current actions
        _, _, valuations = self.get_current_customer()
        max_utility = 0
        max_utility_item = -1
        max_utility_agent = -1
        # for each item, loop through agents. get lowest price for agent that has capacity left, keep track of which item if any the customer is buying
        for item in range(self.n_items):
            value = valuations[item]
            for agent in range(self.n_agents):
                util = value - all_agent_prices[agent][item]
                if util >= 0 and util + (random.random() - 0.5) * eps > max_utility:
                    max_utility = util
                    max_utility_item = item
                    max_utility_agent = agent

        if max_utility_agent >= 0:
            # print(type(all_agent_prices))
            # print(self.agent_profits[max_utility_agent])
            # print(all_agent_prices)
            # print(all_agent_prices[max_utility_agent])
            # print(type(all_agent_prices[max_utility_agent]))
            # print(all_agent_prices[
            #     max_utility_agent
            # ][max_utility_item])
            self.agent_profits[max_utility_agent] += all_agent_prices[
                max_utility_agent
            ][max_utility_item]
            self.cumulative_buyer_utility += max_utility
            sale = (
                max_utility_item,
                max_utility_agent,
                all_agent_prices,
            )
        else:
            sale = (np.nan, np.nan, all_agent_prices)
        for agent in range(self.n_agents):
            self.agent_profits_timeseries[agent].append(
                self.agent_profits[agent])

        # increase time step
        self.time += 1

        # create and send new customer for next time step
        return self.get_current_state_customer_to_send_agents(sale)

    def reset(self):
        self.time = 0
        self.cumulative_buyer_utility = 0
        self.agent_profits = [0 for _ in range(self.n_agents)]
        self.agent_profits_timeseries = [[] for _ in range(self.n_agents)]

        self.customers = []
        self._init_data_files()

    def render(self, close_before=False, mode="human", close=False, time_update=10):
        if self.time % time_update == 0:
            if close_before:
                plt.close()
            for agent in range(self.n_agents):
                name = "Agent {}: {}".format(agent, self.agent_names[agent])
                plt.plot(
                    list(range(self.time)),
                    self.agent_profits_timeseries[agent],
                    label=name,
                )
            plt.legend(frameon=False)
            plt.xlabel("Time")
            plt.ylabel("Profit")
            sns.despine()
            return True
        return False
        # print("Cumulative buyer utility: {}".format(self.cumulative_buyer_utility))