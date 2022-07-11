import gym
import numpy as np
import random
import matplotlib as plt
import sys
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class Seller:
    def __init__(
        self,
        seller_id,  # needed when identifying which seller, should be 0,1,2...
        n_sellers=2,
        n_items=2,
        max_step=10000,
    ):
        self.n_sellers = n_sellers
        self.n_items = n_items
        self.seller_id = seller_id
        self.timestep = 0
        self.max_step = max_step

    @property
    def observation_space(self):
        spaces = {'customer_covariates': gym.spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32),
            'customer_embedding': gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32),
            'sale': gym.spaces.Box(
            low=-1, high=11, shape=(6,), dtype=np.float32),
            'seller_profits': gym.spaces.Box(
            low=-10000, high=10000, shape=(2,), dtype=np.float32),}
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    @property
    def action_space(self):
        return gym.spaces.Box(low=0, high=float('10'), shape=(self.n_items,),
                          dtype=np.float32)

    def step(self, action, prices, state):
        self.timestep += 1
        prices[self.seller_id] = action
        return state, 0.0, self.timestep >= self.max_step, {self.seller_id: action}
    
    def get_observation(self, state):
        return state

def env(**kwargs):
    env = MultiPriceCompetitionEnv(**kwargs)
    return env

class MultiPriceCompetitionEnv(MultiAgentEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, n_sellers=2, n_items=2, customer_covariates=None, customer_embedding=None, max_step=10000):
        """
        n_agents: number of sellers in the pricing competition
        n_items: number of items available, same for each seller
        customer_covariates: TODO the demographic covariates distribution of customers
        maxstep: after max_step steps all agents will return done
        """

        super().__init__()
        self.n_sellers = n_sellers
        self.n_items = n_items
        self.max_step = max_step
        if customer_covariates==None:
            self.customer_covariates = [1, 1, 1]
        if customer_embedding==None:
            self.customer_embedding = [0]

        self.timestep = 0
        self.sale_0 = [-1, -1, 0, 0, 0, 0]

        self.seller_ids = list(range(self.n_sellers))
        self.sellers = [
            Seller(id, n_sellers, n_items)
            for id in self.seller_ids
        ]
        self.last_rewards = [0 for _ in range(self.n_sellers)]
        self.seller_profits = [0 for _ in range(self.n_sellers)]
        self.seller_profits_timeseries = [[] for _ in range(self.n_sellers)]

        self.dones = set()

        self.customers = []

        self.state = {'customer_covariates': self.customer_covariates,
            'customer_embedding': self.customer_embedding,
            'sale': self.sale_0,
            'seller_profits': self.seller_profits,}

        self.observation_space = [
            agent.observation_space for agent in self.sellers]
        self.action_space = [agent.action_space for agent in self.sellers]

    def reset(self):
        """Reinitializes variables."""

        self.timestep = 0

        self.last_rewards = [0 for _ in range(self.n_sellers)]
        self.seller_profits = [0 for _ in range(self.n_sellers)]
        self.seller_profits_timeseries = [[] for _ in range(self.n_sellers)]

        self.dones = set()

        self.customers = []

        self.state = {'customer_covariates': self.customer_covariates,
            'customer_embedding': self.customer_embedding,
            'sale': self.sale_0,
            'seller_profits': self.seller_profits,}

        return {i: a.get_observation(self.state) for i, a in enumerate(self.sellers)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        prices = [[0 for _ in range(self.n_sellers)] for _ in range(self.n_items)]
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action, prices, self.state)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)

        #Start true stepping
        eps = 1e-7  # for random tie-breaking

        valuations = [5, 5] # naive customer valuation, later can be calculated with covariate distribution

        max_utility = 0
        max_utility_item = -1
        max_utility_agent = -1

        # for each item, loop through agents. get lowest price for agent that has capacity left, keep track of which item if any the customer is buying
        for item in range(self.n_items):
            value = valuations[item]
            for agent in range(self.n_sellers):
                util = value - action[agent][item]
                if util >= 0 and util + (random.random() - 0.5) * eps > max_utility:
                    max_utility = util
                    max_utility_item = item
                    max_utility_agent = agent

        if max_utility_agent >= 0:
            self.seller_profits[max_utility_agent] += action[
                max_utility_agent
            ][max_utility_item]
            sale = np.concatenate(
                ([max_utility_item, max_utility_agent], action.flatten()))
        else:
            sale = np.concatenate(([-1, -1], action.flatten()))
        for agent in range(self.n_sellers):
            self.seller_profits_timeseries[agent].append(
                self.seller_profits[agent])

        # set rewards
        for agent in self.seller_list:
            if max_utility_agent == agent:
                rew[agent] = action[agent][max_utility_item]
            else:
                rew[agent] = 0

        # flatten and define state
        self.state = {'customer_covariates': self.customer_covariates,
            'customer_embedding': self.customer_embedding,
            'sale': sale.tolist(),
            'seller_profits': self.seller_profits,}

        return obs, rew, done, info

    def render(self, close_before=False, mode="human", close=False, time_update=10):
        if self.time % time_update == 0:
            if close_before:
                plt.close()
            for agent in range(self.n_sellers):
                name = "Agent {}: {}".format(agent, self.seller_list[agent])
                plt.plot(
                    list(range(self.time)),
                    self.seller_profits_timeseries[agent],
                    label=name,
                )
            plt.legend(frameon=False)
            plt.xlabel("Time")
            plt.ylabel("Profit")
            sys.despine()
            return True
        return False

