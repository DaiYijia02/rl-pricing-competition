import random
import pickle
import os
import numpy as np
import pandas as pd
import math
import sys

import torch
import torch.nn as nn
from sklearn.impute import KNNImputer

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model

        self.ti = 0
        self.last_block = []
        self.curr_alpha0 = -0.2
        self.curr_alpha1 = -0.5


        # self.nnfilename = 'machine_learning_model/nnpickle_model'
        # self.nn_model = pickle.load(open(self.nnfilename, 'rb'))
        self.nn_model  = torch.load('agents/floor6_files/125_nn_model')
        self.nn_model.eval()

        self.knnfilename = 'agents/floor6_files/knnpickle_model'
        with open(self.knnfilename, 'rb') as f:
            self.knn_model = pickle.load(f)


        self.item0filename = 'agents/floor6_files/item0embedding'
        with open(self.item0filename, 'rb') as f0:
            self.item0embedding = pickle.load(f0)

        self.item1filename = 'agents/floor6_files/item1embedding'
        with open(self.item1filename, 'rb') as f1:
            self.item1embedding = pickle.load(f1)

        #TODO load pickled version of the table here
        # self.tablepath = 'machine_learning_model/q_table'
        # with open(self.tablepath, 'rb') as f2:
        #     self.q_table = pickle.load(f2)



     # Search Models
    def simple_gridsearch_rev(self,xmax, ymax,  user, ylow=0, xlow=0, ticks=10, silent=True, nn=False):
        model = self.nn_model
        x = []
        y = []
        rev = []
        
        for i in range(math.floor(ylow*ticks), math.ceil(ymax*ticks) + 1):
            for j in range(math.floor(xlow*ticks), math.ceil(xmax*ticks) + 1):
                x.append(j/ticks)
                y.append(i/ticks)
                if nn:
                    user[0][3] = i/ticks #_tensor
                    user[0][4] = j/ticks #_tensor
                    prob = self.nn_out(user)
                    rev.append(np.dot([0,i/ticks,j/ticks], np.exp(prob)[0]))
                else:
                    user['price_item_0'] = i/ticks
                    user['price_item_1'] = j/ticks
                    prob = model.predict_proba(user)
                    #print(prob)
                    rev.append(np.dot([0,i/ticks,j/ticks], prob[0]))
       
        if not silent:
            print('max is {}'.format(np.amax(rev)))
            print('x: {}, y:{}'.format(x[np.argmax(rev)],y[np.argmax(rev)]))
            
        return x, y, rev

    def adv_gridsearch_rev(self, xmax, ymax, user, xlow=0, ylow=0, ticks=10, max_iter_=8, silent=True, nn=False):
        # xmax=4.1, ymax=2.5, model=self.nn_model, user=X_train_tensor, nn=True, silent=True, max_iter_=15
       
        model = self.nn_model
        ylow = ylow
        yhigh = ymax
        xlow = xlow
        xhigh = xmax
        
        best_x = 0
        best_y = 0
        best_rev = 0.01
        new_rev = 0
        prev_rev = 0
        
        iter_c = 1
        
        while abs(best_rev - prev_rev) > 0 and iter_c < max_iter_:
        
            x, y, rev = self.simple_gridsearch_rev(xmax, ymax, user, ylow, xlow, iter_c*ticks, silent=silent, nn=nn)
            
            prev_rev = new_rev
            new_rev = np.amax(rev)

            if new_rev > best_rev:
                best_rev = new_rev
                best_x = x[np.argmax(rev)]
                best_y = y[np.argmax(rev)]
            
            yhigh = best_y+0.2*best_y
            xhigh = best_x+0.2*best_x
            ylow = best_y-0.2*best_y
            xlow = best_x-0.2*best_x
            iter_c += 1
            
        return best_rev, best_x, best_y

    def rand_grad_search(self,xmax, ymax, user, x_start=0, y_start=0, max_iter_=100, silent=True, nn=False):
        model = self.nn_model
        p_p_rev = 0
        prev_rev = -3
        new_rev = 0
        best_rev = -1
        
        path = [[0, 0]]
        step = 0.1
        
        x = x_start#xmax / 2#
        y = x_start#ymax / 2#y_start
        #
        iter = 0
        
        while iter < max_iter_: #abs(p_p_rev - best_rev) > 1e-6:
            #print("best rev is {}".format(best_rev))
            #print(step)
            if p_p_rev == new_rev:
                if step < 0.005:
                    step = 0.25
                    x = xmax / (np.random.randint(xmax) + 1)
                    y = ymax / (np.random.randint(ymax) + 1)
                else:
                    step = step / 2
            
            search_revs = []
            xs = []
            ys = []
            
            for i in range(8):
                x_sign = 1
                y_sign = 1
                if i % 2 == 0:
                    y_sign = -1
                if math.floor(i / 4) == 0:
                    x_sign = -1
                    
                xs.append(x + x_sign*step)
                ys.append(y + y_sign*step)
                
                if nn:
                    user[0][3] = y + y_sign*step
                    user[0][4] = x + x_sign*step
                    #print(user)
                    prob = self.nn_out(user)
                    #print(np.dot([0,y+y_sign*step,x+x_sign*step], np.exp(prob)[0]))
                    search_revs.append(np.dot([0,y+y_sign*step,x+x_sign*step], np.exp(prob)[0]))
                else:
                    user['price_item_0'] = y + y_sign*step
                    user['price_item_1'] = x + x_sign*step
                    prob = model.predict_proba(user)
                    search_revs.append(np.dot([0,y+y_sign*step,x+x_sign*step], prob[0]))

            p_p_rev = prev_rev
            prev_rev = new_rev
            new_rev = np.amax(search_revs)
            x = xs[np.argmax(search_revs)]
            y = ys[np.argmax(search_revs)]
            

            if new_rev > best_rev:
                best_rev = new_rev
                path.append([x, y])
                
            #print(abs(prev_rev - new_rev))
            iter += 1
        
        if not silent:
            print("best rev is {}".format(best_rev))
            #print(path[-1])
        return path, best_rev

    def rand_init_adv_grid_search(self,xmax, ymax,  user, x_start=0, y_start=0, start_iter_=25, silent=True, nn=False):
        model = self.nn_model
        init_xy, init_rev = self.rand_grad_search(ymax=ymax,xmax=xmax,
                                             user=user,
                                             max_iter_=start_iter_,
                                             x_start=x_start,
                                             y_start=y_start, silent=silent, nn=nn)
        #print(init_xy[-1])
        return self.adv_gridsearch_rev(xmax=init_xy[-1][0]+1*init_xy[-1][0],
                                  ymax=init_xy[-1][1]+1*init_xy[-1][1],
                                   user=user, 
                                  xlow=init_xy[-1][0]+1*init_xy[-1][0], 
                                  ylow=init_xy[-1][1]+1*init_xy[-1][1],
                                  silent=silent, nn=nn)



       #torch output function
    def nn_out(self,tensor):
        with torch.no_grad():
            return self.nn_model(tensor)
    


    def _process_last_sale(self, last_sale, profit_each_team):
        # if this is the beginning of the new game
        self.ti +=1
        if last_sale[2][self.this_agent_number][0] == np.nan:
            return 1
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)
        x=0
        if did_customer_buy_from_me:
            x =1

        y=0
        if did_customer_buy_from_opponent:
            y =1
        #keep track of the last 100 sales
        data = [which_item_customer_bought,x,y,my_last_prices,opponent_last_prices,my_current_profit, opponent_current_profit]
        self.last_block.append(data)


        n = 30
        if len(self.last_block) == (2*n)+1:
            self.last_block.pop(0)

 
        # only make price adjustment after every 50 T
        if (self.ti > n) & (self.ti % n == 1 ):
            # analyze last 50 T
            self.analysis(None, self.last_block[-n:])
            # if len(self.last_block) >50 :
            #     self.analysis(self.last_block[-100:-50],self.last_block[-50:])
            # else:
            #     #TODO make policy decision
            #     self.analysis(None, self.last_block[-50:])



          

    def analysis(self,old_list, new_list):
        if old_list is None:
            # change in profit differences thru out the 50 Ts
            profit_diff = (new_list[-1][5] - new_list[-1][6]) - (new_list[0][5] - new_list[0][6])
            n_me_0 = 0
            n_me_1 = 0
            # n_oppo_0 = 0
            # n_oppo_1 = 0
            item_0_loss = []
            item_1_loss = []

            #TODO if customer bought from neither team, consider lowering price

            for result in new_list:
                if result[0] ==0:
                    n_me_0 += result[1]
                    # n_oppo_0 += result[2]
                elif result[0] ==1:
                    n_me_1 += result[1]
                    # n_oppo_1 += result[2]

                # if customer bought from opponent, what is avg diferences in price
                #item 0
                if (result[2] == 1) & (result[0]== 0):
                    my_p = result[3][0]
                    oppo_p = result[4][0]
                    item_0_loss.append(my_p - oppo_p)

                #item 1
                if (result[2] ==1) & (result[0]== 1):
                    my_p = result[3][1]
                    oppo_p = result[4][1]
                    item_1_loss.append(my_p - oppo_p)


            item_0_loss = self.remove_outlier(item_0_loss)
            item_1_loss = self.remove_outlier(item_1_loss)
            #remove outliers in item_0loss and item_1 loss
            n_oppo_0 = len(item_0_loss)
            n_oppo_1 = len(item_1_loss)
            # self.curr_alpha0 = 1 * self.curr_alpha0
            # self.curr_alpha1 = 1 * self.curr_alpha1

            my_last_block_profit = new_list[-1][5] - new_list[0][5]
            # if opponent outsell me 5 items for item0, I lower my price by that amount
            if n_oppo_1 - n_me_1 >= 1:
                if len(item_0_loss) != 0:
                    avg0 = max(sum(item_0_loss)/len(item_0_loss), 0.02)
                    self.curr_alpha0 +=  -(avg0 + 0.02)
            # if Im performing really raelly well, try incraesing my price
            elif n_me_0 - n_oppo_0 >= 10:
                # self.curr_alpha0 += abs(n_me_0 - n_oppo_0) * 0.005
                # if self.curr_alpha0 <0 :
                #     self.curr_alpha0 += min(0.07, abs(self.curr_alpha0))

                if (self.curr_alpha0 <0 ) :
                    self.curr_alpha0 += min(0.07, abs(self.curr_alpha0))
                # elif (self.curr_alpha0 <0) & (self.ti >60) &(  (new_list[-1][6] - new_list[0][6] - self.oppo_prof)/(self.oppo_prof+0.1) > 2  ):
                #     self.curr_alpha0 = self.curr_alpha0- 0.4

            # if opponent outsell me 5 items for item1, I lower my price by that amount
            if n_oppo_1 - n_me_1 >= 1:
                if len(item_1_loss) != 0:
                    avg1 = max(sum(item_1_loss)/len(item_1_loss),0.02)
                    self.curr_alpha1 +=  -(avg1 + 0.02)
            elif n_me_1 - n_oppo_1 >= 10:
                # self.curr_alpha0 += abs(n_me_0 - n_oppo_0) * 0.005
                if (self.curr_alpha1 <0):
                    self.curr_alpha1 += min(0.07, abs(self.curr_alpha1))
                # elif (self.curr_alpha1 <0) & (self.ti >60) &(  (new_list[-1][6] - new_list[0][6] - self.oppo_prof)/(self.oppo_prof+0.1) > 2  ):
                #     self.curr_alpha1 = self.curr_alpha1- 1
            

        #TODO run T/Z test to test out significant difference in last method tried.
    def remove_outlier(self, l):
        if len(l)<1:
            return []
        data = np.array(l)
        return list(data[abs(data - np.mean(data)) < 2.1 * np.std(data)])
    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        # if this is the beginning of a new game
        if last_sale[2][self.this_agent_number][0] == np.nan:
            self.ti = 0
            self.last_block = []
        #filling in missing user embedding
        if new_buyer_embedding is None :
            #output (1,13)
            temp_embedding = np.empty(10)
            temp_embedding[:] = np.nan

            inp = np.concatenate((new_buyer_covariates, temp_embedding)).reshape(1,-1)
            temp = self.knn_model.transform(inp)
            new_buyer_embedding = temp[:, 3:]
            # new_buyer_embedding = temp.reshape(1,13)[3:]


        i0_dot = np.dot(new_buyer_embedding, self.item0embedding)
        i1_dot = np.dot(new_buyer_embedding, self.item1embedding)
        #preparing Neural Network Input
        train_x = pd.DataFrame(new_buyer_covariates.reshape(1,-1), columns = ['Covariate 1', 'Covariate 2', 'Covariate 3'])
        train_x['price_item_0'] = 1
        train_x['price_item_1'] = 1
        train_x['il_dot'] = i1_dot
        train_x['i0_dot'] = i0_dot

        X_train_tensor = torch.tensor(train_x.values).float()




        #Running the prediction using Network

        # rev, x, y = self.adv_gridsearch_rev(self.nn_model, X_train_tensor, 15, True, True)#,start_iter_=25

        rev, x, y = self.rand_init_adv_grid_search(xmax=4.1, ymax=2.5,user=X_train_tensor, nn=True, silent=True, start_iter_=8)#,start_iter_=25
        # rev, x, y = self.adv_gridsearch_rev(xmax=4.1, ymax=2.5,user=X_train_tensor, nn=True, silent=True, max_iter_=8)
        
        self._process_last_sale(last_sale, profit_each_team)
        # return self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0] + random.random()

        #if i got passed
        # if self.ti >3:
        #     pre = self.last_block[-1]
        #     prepre = self.last_block[-2]
        #     if (self.ti >10) & (pre[-1] > pre[-2] ) & (prepre[-1] < prepre[-2] ):
        #         self.curr_alpha0 = min(-0.6,self.curr_alpha0 - 0.2)
        #         self.curr_alpha0 = min(-1.2,self.curr_alpha0 - 0.5)

        self.check_alpha()

        y = y + self.curr_alpha0
        if y <= 0.000001:
            y= 0.000001

        x = x + self.curr_alpha1
        if x <= 0.000001:
            x= 0.000001
        price = [y, x]

        if self.ti <=7:
            return[sys.float_info.min,sys.float_info.min]
        return price

        # return [10,10]
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to create prices for each item.



        #TODO: implement grid search 
    def check_alpha(self):
        if self.curr_alpha0 > 0.4:
            self.curr_alpha0 = 0.4

        if self.curr_alpha1 > 0.8:
            self.curr_alpha1 = 0.8