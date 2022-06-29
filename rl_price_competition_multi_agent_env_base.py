"""
A price competition environment over
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
import random
import os
import sys
import matplotlib.pyplot as plt
from utils import Agent
currentdir = os.path.dirname(os.path.realpath(__file__))
renderdir = os.path.dirname(currentdir)
sys.path.append(renderdir)
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

# ------------------------------------------------------------------------------

class Seller:
    def __init__(
        self,
        seller_id, # needed when identifying which seller, should be 0,1,2...
        init=False, # identifier whether this seller has initiated
        n_sellers=2,
        n_items=2
    ):
        self.n_sellers = n_sellers
        self.n_items = n_items
        self.seller_id = seller_id
        self.init = init

    @property
    def observation_space(self):
        return spaces.Box(low=float('-inf'), high=float('inf'), shape=(13,),
                                    dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Box(low=0.0, high=float('inf'), shape=(self.n_sellers,self.n_items),
                                    dtype=np.float32)
    
    def apply_action(self, one_action):
        action = [0 for _ in range(self.n_sellers)]
        action[self.seller_id] = one_action
        return action


class MultiPriceCompetitionEnv:

    metadata = {'render.modes': ['human']}

    def __init__(self, n_sellers=2, n_items=2, customer_covariates=None, max_step=10000):
        """
        n_agents: number of sellers in the pricing competition
        n_items: number of items available, same for each seller
        customer_covariates: TODO the demographic covariates distribution of customers
        maxstep: after max_step steps all agents will return done
        """
        
        self.max_step = max_step
        self.timestep = 0
        self.n_sellers = n_sellers
        self.n_items = n_items

        self.customer_covariates = [1, 1, 1]
        self.customer_embedding = 0
        sale = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        self.seller_list = list(range(self.n_sellers))
        self.sellers = [
            Seller(id, n_sellers, n_items)
            for id in self.seller_list
        ]
        self.last_rewards = [0 for _ in range(self.n_sellers)]
        self.seller_profits = [0 for _ in range(self.n_sellers)]
        self.seller_profits_timeseries = [[] for _ in range(self.n_sellers)]

        self.last_dones = [False for _ in range(self.n_sellers)]

        self.customers = []

        self.state = [self.customer_covariates, self.customer_embedding, sale, self.seller_profits].flatten()

        self.observation_space = [agent.observation_space for agent in self.sellers]
        self.action_space = [agent.action_space for agent in self.sellers]

    def reset(self):
        """Reinitializes variables."""

        self.timestep = 0

        self.seller_profits = [0 for _ in range(self.n_sellers)]
        self.seller_profits_timeseries = [[] for _ in range(self.n_sellers)]

        self.customers = []


    def step(self, action, agent_id, is_last):
        """
        Move one step in the environment.

        Args:
            action: A 2*2 box contains two seller's prices for two items.
        Returns:
            reward: A 2*2 box of the revenues for two sellers on two items representing the reward based on the action chosen.

            newState: A 13*1 box of the current demographic covariants for the customers representing the state of the environment.

            done: A bool flag indicating the end of the episode.
        """
         # action is array of size 4
        action = action.reshape(2)
        assert self.seller_list[agent_id].init is not False, agent_id
        self.seller_list[agent_id].apply_action(action)
        if is_last:

            eps = 1e-7  # for random tie-breaking

            valuations = [10, 10]

            max_utility = 0
            max_utility_item = -1
            max_utility_agent = -1

            # for each item, loop through agents. get lowest price for agent that has capacity left, keep track of which item if any the customer is buying
            for item in range(self.n_items):
                value = valuations[item]
                for agent in range(self.seller_list):
                    util = value - action[agent][item]
                    if util >= 0 and util + (random.random() - 0.5) * eps > max_utility:
                        max_utility = util
                        max_utility_item = item
                        max_utility_agent = agent
            
            if max_utility_agent >= 0:
                self.seller_profits[max_utility_agent] += action[
                    max_utility_agent
                ][max_utility_item]
                sale = (
                    max_utility_item,
                    max_utility_agent,
                    action,
                )
            else:
                sale = (np.nan, np.nan, action)
            for agent in range(self.n_sellers):
                self.seller_profits_timeseries[agent].append(
                    self.seller_profits[agent])

            # set rewards
            rewards = {}
            for agent in self.seller_list:
                if max_utility_agent == agent:
                    rewards[agent] = action[agent][max_utility_item]
                else:
                    rewards[agent] = 0

            # increase time step
            self.timestep += 1

            # set done
            done = self.timestep <= self.max_step

            # flatten and define state
            self.state = [self.customer_covariates, self.customer_embedding, sale.flat, self.agent_profits]

            return self.state, rewards, done

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
            sns.despine()
            return True
        return False

    def get_last_rewards(self):
        return dict(
            zip(
                list(range(self.n_sellers)),
                map(lambda r: np.float64(r), self.last_rewards),
            )
        )

    def get_last_dones(self):
        return dict(zip(self.seller_list, self.last_dones))

    def observe(self, agent):
        o = self.state
        o = np.array(o, dtype=np.float32)
        return o