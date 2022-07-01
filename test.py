from pettingzoo.test import api_test
from pettingzoo.test import render_test
import rl_price_competition_multi_agent_env

env = rl_price_competition_multi_agent_env.env()
# api_test(env, num_cycles=1000, verbose_progress=False)
render_test(env)