import random
import numpy as np
from collections import namedtuple, deque


# ------------------------------------------------ Financial Parameters --------------------------------------------------- #

ANNUAL_VOLAT = 0.12                                # Annual volatility in stock price
BID_ASK_SP = 1 / 8                                 # Bid-ask spread
DAILY_TRADE_VOL = 5e6                              # Average Daily trading volume  
TRAD_DAYS = 250                                    # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)    # Daily volatility in stock price


# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #

TRADING_FEE = 0.01                                                   # Fixed trading fee per share traded
TOTAL_SHARES = 1000000                                               # Total number of shares to sell
STARTING_PRICE = 50                                                  # Starting price per share
LLAMBDA = 1e-6                                                       # Trader's risk aversion
LIQUIDATION_TIME = 60                                                # How many days to sell all the shares. 
NUM_N = 60                                                           # Number of trades
EPSILON = BID_ASK_SP / 2  + TRADING_FEE                              # Fixed Cost of Selling, Updated to include trading fee 
SINGLE_STEP_VARIANCE = (DAILY_VOLAT  * STARTING_PRICE) ** 2          # Calculate single step variance
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)                          # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)                         # Permanent Impact Constant

# ----------------------------------------------------------------------------------------------------------------------- #


# Simulation Environment

class MarketEnvironment():
    
    def __init__(self, randomSeed = 0,
                 lqd_time = LIQUIDATION_TIME,
                 num_tr = NUM_N,
                 lambd = LLAMBDA):
        
        # Set the random seed
        random.seed(randomSeed)
        
        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        
        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        
        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = deque(np.zeros(6))
        
        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting = False
        
        # Set a variable to keep trak of the trade number
        self.k = 0
        
        
    def reset(self, seed = 0, liquid_time = LIQUIDATION_TIME, num_trades = NUM_N, lamb = LLAMBDA):
        
        # Initialize the environment with the given parameters
        self.__init__(randomSeed = seed, lqd_time = liquid_time, num_tr = num_trades, lambd = lamb)
        
        # Set the initial state to [0,0,0,0,0,0,1,1]
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
                                                               self.shares_remaining / self.total_shares])
        return self.initial_state

    
    def start_transactions(self):
        
        # Set transactions on
        self.transacting = True
        
        # Set the minimum number of stocks one can sell
        self.tolerance = 1
        
        # Set the initial capture to zero
        self.totalCapture = 0
        
        # Set the initial previous price to the starting price
        self.prevPrice = self.startingPrice
        
        # Set the initial square of the shares to sell to zero
        self.totalSSSQ = 0
        
        # Set the initial square of the remaing shares to sell to zero
        self.totalSRSQ = 0
        
        # Set the initial utility
        self.prevUtility = self.compute_utility(self.total_shares)

        self.k = 0
        

    def step(self, action):
        
        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass        
        info = Info()
        
        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        info.done = False
                
        # During training, if the DDPG fails to sell all the stocks before the given 
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        if self.transacting and (self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance

        # We don't add noise before the first trade    
        if self.k == 0:
            info.price = self.prevImpactedPrice
        else:
            # Calculate the current stock price using arithmetic brownian motion
            # info.price = self.arithmetic_brownian_motion(self.prevImpactedPrice)
            # Calculate the current stock price using Geometric Brownian Motion
            info.price = self.geometric_brownian_motion(self.prevImpactedPrice)

        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves 
        # according to the Almgren and Chriss price dynamics model. 
        if self.transacting:
            
            # If action is an ndarray then extract the number from the array
            if isinstance(action, np.ndarray):
                action = action.item()            

            # Convert the action to the number of shares to sell in the current step
            # sharesToSellNow = self.fraction_shares_action(action)
            # sharesToSellNow = self.fraction_daily_volume_action(action)
            sharesToSellNow = self.trading_rate_action(action)
            sharesToSellNow = min(sharesToSellNow, self.shares_remaining)

            if self.timeHorizon < 2:
                sharesToSellNow = self.shares_remaining

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer. 
            info.share_to_sell_now = np.around(sharesToSellNow)

            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)
                
            # Apply the temporary impact on the current stock price    
            info.exec_price = info.price - info.currentTemporaryImpact
            
            # Calculate the current total capture
            self.totalCapture += info.share_to_sell_now * info.exec_price

            # Calculate the log return for the current step and save it in the logReturn deque
            self.logReturns.append(np.log(info.price/self.prevPrice))
            self.logReturns.popleft()
            
            # Update the number of shares remaining
            self.shares_remaining -= info.share_to_sell_now
            
            # Calculate the runnig total of the squares of shares sold and shares remaining
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2
                                        
            # Update the variables required for the next step
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact
            
            # Calculate the reward (old code)
            # reward = self.original_reward()

            # New reward components
            reward = self.reward_utility() 
            # reward = self.complex_reward()
            # reward =  0.0
            if self.k == 0: 
                reward = 0.0
            

            # If all the shares have been sold calculate E, V, and U, and give a positive reward.
            if self.shares_remaining <= 0:
                
                # Calculate the implementation shortfall
                info.implementation_shortfall  = self.total_shares * self.startingPrice - self.totalCapture
                
                # Add implementation shortfall to the reward
                # log_term = np.log(1 + abs(info.implementation_shortfall)) * np.sign(info.implementation_shortfall)

                # reward += -log_term  # np.clip(log_term, -10, 10)   

                # Set the done flag to True. This indicates that we have sold all the shares
                info.done = True
        else:
            reward = -1e6 # just make it very big

        # print(reward)
        
        self.k += 1
            
        # Set the new state
        state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares])

        return (state, np.array([reward]), info.done, info)

   
    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        pi = self.gamma * sharesToSell
        return pi

    
    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti
    
    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)         # permanent impact
        st = self.epsilon * sharesToSell    # spread cost
        tt = (self.eta_hat / self.tau) * self.totalSSSQ # emporary impact cost 
        return ft + st + tt

    
    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell        
        tt = self.eta_hat * (sharesToSell ** 2)       
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))       
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)   
        fot = nft / dft       
        return ft + st + (tt * fot)  
        
    
    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)                        
        nst  = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
               - self.liquidation_time * np.sinh(self.kappa * self.tau)        
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)        
        st = nst / dst
        return ft * st
        
        
    def compute_AC_utility(self, sharesToSell):    
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0        
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V
    
    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):       
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list
    
    def compute_utility(self, sharesToSell):
        # time_factor = (self.timeHorizon/self.num_n) ** 0.5
        if self.liquidation_time == 0:
            return 0 
        E = self.get_expected_shortfall(sharesToSell)
        V = self.singleStepVariance * self.tau * self.totalSRSQ
        return namedtuple('Utility', ['E', 'V', 'total'])(E, V, E + self.llambda * V)


    def arithmetic_brownian_motion(self, current_price):
        """
        Price evolves by: S_{t+1} = S_t + sigma * sqrt(tau) * Z
        where Z ~ N(0, 1).
        """
        # sigma * sqrt(tau) for ABM in *price* terms:
        shock = np.sqrt(self.singleStepVariance * self.tau) * random.normalvariate(0, 1)
        return current_price + shock
    
    def geometric_brownian_motion(self, current_price, mu=0.0):
        """
        Price evolves by: S_{t+1} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ).
    
        Here:
        - mu  : drift (you can set it to 0 if you want “zero-drift”),
        - sigma = self.dpv  (daily percentage volatility),
        - dt   : can be 1 day if each step is 1 day, or self.tau if you're modeling fraction-of-day steps.
        """
        # For a daily step, dt = 1. Or if each action is 1/60th of a day, dt = self.tau.
        dt = self.tau  # or self.tau, depending on how you interpret time in your simulation
        sigma = self.dpv  # daily volatility in *log-returns*

        Z = random.normalvariate(0, 1)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        return current_price * np.exp(drift + diffusion)
    
    def fraction_shares_action(self, action):
        # Convert the action to the number of shares to sell in the current step
        return self.shares_remaining * action

    def fraction_daily_volume_action(self, action):
        # Convert the action to the number of shares to sell in the current step
        # fraction of daily volume
        return action * self.dtv * self.tau   
    
    def trading_rate_action(self, action):
        # If you have TOTAL_SHARES to sell over D days:
        # On average, you'd need (TOTAL_SHARES / D) shares/day.
        # You could allow the agent to go up to some factor times that average.
        factor = 2.5  # let the agent trade 3 times as fast as the "average pace"
        max_rate = factor * (self.total_shares / self.num_n)
        return max_rate * action
    
    def original_reward(self):
        currentUtility = self.compute_utility(self.shares_remaining)
        reward = (abs(self.prevUtility.total) - abs(currentUtility.total)) / abs(self.prevUtility.total)
        self.prevUtility = currentUtility
        return reward
           

    def reward_utility(self):
        # New reward components
        currentUtility = self.compute_utility(self.shares_remaining)
        cost_reduction = (self.prevUtility.E - currentUtility.E)/(abs(self.prevUtility.E)+ 1)
        variance_reduction = (self.prevUtility.V - currentUtility.V)/(abs(self.prevUtility.V) + 1)
        self.prevUtility = currentUtility

        # Balanced reward
        reward = (0.7 * cost_reduction) + (0.3 * variance_reduction)
        
        # Clip rewards for stability
        reward = np.clip(reward, -2, 2)

        return reward 

    def complex_reward(self):
        current_utility = self.compute_utility(self.shares_remaining)
        
        # Immediate cost penalty
        cost_penalty = -abs(current_utility.E) / (self.total_shares * self.startingPrice)
        
        # Variance penalty
        variance_penalty = -self.llambda * current_utility.V / (self.total_shares**2 * self.singleStepVariance)
        
        # Time penalty (encourage faster execution)
        time_penalty = -0.1 * (self.k / self.num_n)
        
        # Liquidation bonus
        liquidation_bonus = 10.0 if self.shares_remaining <= 0 else 0
        
        reward = (cost_penalty + variance_penalty + time_penalty + liquidation_bonus) / 10
        
        # Clip final reward
        reward = np.clip(reward, -1, 1)

        return reward
       
        
    def observation_space_dimension(self):
        # Return the dimension of the state
        return 8
    
    
    def action_space_dimension(self):
        # Return the dimension of the action
        return 1
    
    
    def stop_transactions(self):
        # Stop transacting
        self.transacting = False            
            
           