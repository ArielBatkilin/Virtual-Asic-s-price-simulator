import math
import random
import matplotlib.pyplot as plt
import numpy as np




"""importent!: when going over the code, it will be easyer to go throw all the simulations first (on the bottom of the page)"""


class GA:
    """Genetic algoritem that: 1. calculating fitneess of weights 2. selecting the mating pool from the best parents
                               3. making children from parents 4. mutating the children:)"""
    def cal_users_fitness(days_to_look_back):
        """Calculating the fitness value of each solution in the current population.
        The fitness function calulates the money user won during his generation."""
        fitness = np.array(
            [user.history_money[-1] - user.history_money[-1 * days_to_look_back] for user in
             ALL_USERS])
        return fitness

    def select_mating_pool(users_whieghts, fitness, num_parents):
        """Selecting the best individuals in the current generation as parents for producing the offspring of the next generation."""
        parents = np.empty((int(num_parents), users_whieghts.shape[1]))
        for parent_num in range(int(num_parents)):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = users_whieghts[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(parents, offspring_size):
        """making the children"""
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    # def mutation(offspring_crossover, num_mutations=1):
    #     mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    #     # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    #     for idx in range(offspring_crossover.shape[0]):
    #         gene_idx = mutations_counter - 1
    #         for mutation_num in range(num_mutations):
    #             # The random value to be added to the gene.
    #             random_value = np.random.uniform(-1.0, 1.0, 1)
    #             offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
    #             gene_idx = gene_idx + mutations_counter
    #     return offspring_crossover

    def mutation(offspring_crossover, num_mutations=1):
        """Mutation changes a number of genes as defined by the num_mutations argument. The changes are random."""
        mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = random.randint(0, 1)
            # gene_idx = mutations_counter - 1
            for mutation_num in range(num_mutations):
                # The random value to be added to the gene.
                random_value = np.random.uniform(-2, 2, 1)
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                gene_idx = gene_idx + mutations_counter
        return offspring_crossover

class GA_HL:
    """Genetic algoritem (with hidden layer) that: 1. calculating fitneess of weights 2. selecting the mating pool from the best parents
                               3. making children from parents 4. mutating the children:)"""
    def cal_users_fitness(days_to_look_back):
        """Calculating the fitness value of each solution in the current population.
        The fitness function calulates the money user won during his generation."""
        fitness = np.array(
            [user.history_money[-1] - user.history_money[-1 * days_to_look_back] for user in
             ALL_USERS])
        return fitness

    def select_mating_pool(users_whieghts, fitness, num_parents):
        """Selecting the best individuals in the current generation as parents for producing the offspring of the next generation."""
        parents = [np.empty((int(num_parents), users_whieghts[0].shape[1], users_whieghts[0].shape[2])), np.empty((int(num_parents), users_whieghts[1].shape[1]))]
        for parent_num in range(int(num_parents)):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[0][parent_num, :] = users_whieghts[0][max_fitness_idx, :]
            parents[1][parent_num, :] = users_whieghts[1][max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(parents, offspring_size):
        """making the children"""
        offspring = [np.empty((offspring_size[0], offspring_size[1], offspring_size[1])) ,np.empty(offspring_size)]

        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents[0].shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents[0].shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[0][k, 0:crossover_point] = parents[0][parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[0][k, crossover_point:] = parents[0][parent2_idx, crossover_point:]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[1][k, 0:crossover_point] = parents[1][parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[1][k, crossover_point:] = parents[1][parent2_idx, crossover_point:]
        return offspring

    def mutation(offspring_crossover, num_mutations=1):
        """Mutation changes a number of genes as defined by the num_mutations argument. The changes are random."""
        for idx in range(offspring_crossover[0].shape[0]):
            gene_idx = random.randint(0, offspring_crossover[0].shape[1] - 1)
            for mutation_num in range(num_mutations):
                # The random values to be added to the gene.
                random_value1 = np.random.uniform(-2.0, 2.0, 2)
                random_value2 = np.random.uniform(-2.0, 2.0, 1)
                offspring_crossover[0][idx, gene_idx] = offspring_crossover[0][idx, gene_idx] + random_value1
                offspring_crossover[1][idx, gene_idx] = offspring_crossover[1][idx, gene_idx] + random_value2
        return offspring_crossover

class User:
    """represent a user in the world, that have money and can participate in auctions, own asics and activate them"""
    def __init__(self, id, name, money):
        self.id = id
        self.name = name # for recognition
        self.money = money # money
        self.history_money = [self.money for _ in range(DAYS_FOR_GENERATIONS_RAEL+5)] # for evalution of the user, get his money for the last 100 days
        self.asics = [] #user's asics - objects
        self.pools = [] #user's pools that connect to coins - all objects
        ALL_USERS.append(self) #updated the all users list
        self.bidding_weights = [(np.random.rand(basic_cromozome_len, basic_cromozome_len) - 1), (np.random.rand(basic_cromozome_len) - 1)] #the bidding weights, according to the explantion above
        self.bidding_weights_hidden_layer = [np.random.uniform(low=-5.0, high=5.0, size=(basic_cromozome_len, basic_cromozome_len)), np.random.uniform(low=-5.0, high=5.0, size=basic_cromozome_len)]  # the bidding weights, according to the explantion above
        self.bidding_weights_no_hidden_layer = np.random.uniform(low=-5.0, high=5.0, size=basic_cromozome_len)
        self.mybid = 0 # remember my last bid
        self.rank = -1 # the rank of a user is calculated in the learning - ai - genetic alogrithem stage. if you are earning money
        # your rank will be high and therefor your weights will imitate and mutate( self adpat) less.
        self.last_spendings = [] # this list follows the last asics you bought. how many days ago you bought them and in which price.
        # you need to evalute the money earning of a user ( to calculate the rank ), while understanding it has spent money on asics.

    def notify_of_dead_asic(self, asic):
        """remove asic from users asics"""
        self.asics.remove(asic)

    def update_weights(self, weights):
        """updates users weights"""
        self.bidding_weights_no_hidden_layer = weights

    def update_weights_hidden_layer(self, users_biding_weights_part1, users_biding_weights_part2):
        """updates users weights"""
        self.bidding_weights_hidden_layer[0] = users_biding_weights_part1
        self.bidding_weights_hidden_layer[1] = users_biding_weights_part2

    def restart_users_property(self):
        """sets the users property to initial"""
        self.money = STARTING_MONEY
        self.asics = []
        self.history_money = [self.money for _ in range(DAYS_FOR_GENERATIONS_RAEL+5)]
        self.last_spendings = []

    def get_bid_no_hidden_layer(self, global_features):
        """caluclate the user's bid. gets the global features (attributes) of the simulation"""
        # and then calculates the local features.
        features = np.array(global_features) #[self.money, sum(asic.get_hash_rate() for asic in self.asics), len(self.asics)] +
        # calculates from the features and the weights the bid. the bid must be between 0 and the user's money
        self.mybid = min(max(0, np.dot(self.bidding_weights_no_hidden_layer, features)), self.money)
        return self.mybid

    def get_bid(self, global_features):
        """caluclate the user's bid. gets the golbal features (attributes) of the simulation"""
        # and then calculates the local features.
        features = np.array(global_features) #[self.money, sum(asic.get_hash_rate() for asic in self.asics), len(self.asics)] +
        # calculates from the features and the weights the bid. the bid must be between 0 and the user's money
        self.mybid = min(max(0, np.dot(np.tanh(np.dot(self.bidding_weights[0], features)), self.bidding_weights[1])), self.money)
        return self.mybid

    def add_pool(self, pool): #
        """adds data on new pool. last days hash rates remembers how much hash rates have you
        invested in the pool in the last SIMPLE DAYS"""
        self.pools.append({"pool": pool, "last_days_hash_rates": [0 for j in range(EASY_DAYS)]})

    def bid(self, auc, money):
        """bids money on an auction"""
        if money <= user.money:
            auc.send_bid(self, money)

    def add_asic(self, asic, price):
        """add asic and update the relevent lists"""
        self.asics.append(asic)
        self.last_spendings.append([price, ASIC_DEATH_TIME]) # NOTICE, LAST SPENDING IS COUNTING BACKWORDS THE TIME OF THE ASIC MACHINE


    def update_day(self):
        """at the end of the day we calculate our money (money + asics worth) and update the history money list and every 100 days we making a new generation"""
        self.update_last_spendnding()
        self.history_money = self.history_money[1:] + [self.money + sum(asic[0] * (NORMAL_DECAY_FACTOR ** asic[1]) for asic in self.last_spendings)] # UPDATE HISTORY MONEY
        #THIS IS AN IMPORTENT CALCULATION. it is very relevent for the learning because it evluates how much money youre asics worth for the learning
        # you can see it depends on the decay factor. i will try to explain the importance of this calculation.
        # with the user's history money you calculate the ranks - the more you earn (history money ) in relation of your current balance, the higher
        # your rank. in this calculation you evaluate how much an asic is "worth" when you evaluate how good a user in the learning.
        #BIG PROBLEM. this evaluation is makes the asics less worth each day - BUT - asics hash rate decay only when used - maybe it is worth changing
        if self.rank != -1: # RANKS ARE CALCULATED GLOBALY. the init rank is -1 so when the simulation will start the learning it
            # update the ranks and then they wont be -1. and then the users will do imitate and mutation (self adpt)
            self.self_adapt()
            self.imitate()
        #ranks 1 - 10

    def update_day_GA_class(self):
        """at the end of the day we calculate our money (money + asics worth) and update the history money list and every 100 days we making a new generation"""
        self.update_last_spendnding()
        self.history_money = self.history_money[1:] + [self.money + sum(
            asic[0] * (NORMAL_DECAY_FACTOR ** asic[1]) for asic in self.last_spendings)]  # UPDATE HISTORY MONEY
        # THIS IS AN IMPORTENT CALCULATION. it is very relevent for the learning because it evluates how much money youre asics worth for the learning
        # you can see it depends on the decay factor. i will try to explain the importance of this calculation.
        # with the user's history money you calculate the ranks - the more you earn (history money ) in relation of your current balance, the higher
        # your rank. in this calculation you evaluate how much an asic is "worth" when you evaluate how good a user in the learning.
        # BIG PROBLEM. this evaluation is makes the asics less worth each day - BUT - asics hash rate decay only when used - maybe it is worth changing

    def imitate(self): #mutate
        if CHANCE_IMITATE_FUNC(self.rank): # if the lottery (according to your rank) sayed you need to imitate
            for user in ALL_USERS:
                # i dont remeber why they are in the loop:
                new_bw = []
                sbw = self.bidding_weights
                # --
                if user.rank > self.rank:
                    # stop when you found a user with a higher rank ( maybe it is worth to change this algorithem so
                    # it will have higher chances to pick higher ranked users ).
                    ubw = user.bidding_weights
                    for i in range(len(self.bidding_weights)): # for each weight ( matrix or not matrix)
                        slicing_spot = random.randint(0,len(user.bidding_weights[i]) // 2) # pick slicing spot and copy half of others weights
                        new_bw.append(np.array(list(sbw[i][:slicing_spot]) + list(ubw[i][slicing_spot:slicing_spot + len(sbw[i]) // 2])
                                                + list(sbw[i][slicing_spot + len(sbw[i]) // 2 :])))
                        # maybe worth to look at special case of matrixes. this algorithem cannot slice in the middle of a row
                        # and maybe it limits the learning
                    self.bidding_weights = new_bw # update the weights
                    break

    def self_adapt(self): # the mutation function
        for i in range(len(self.bidding_weights)):
            if i in MATRIX_INDEXES:
                for j in range(len(self.bidding_weights[i])):
                    for k in range(len(self.bidding_weights[i][j])): # for each weight
                        if CHANCE_SELF_ADAPT_FUNC(self.rank): # if the weighted lotterd to be mutated
                            self.bidding_weights[i][j][k] = VALUE_SELF_ADAPT_FUNC(self.bidding_weights[i][j][k], self.rank) # mutate the weight

    def class_end_day(self, day): # end day - not for a single user but to all of the users
        random.shuffle(ALL_USERS) # for the imitate function - there is an advantage to the first users.
        if day > LEARNING_START_TIME: #when learning start calculating ranks
            self.class_ranks_update()

    def class_ranks_update(self): # the ranks calaculator
        # gets the value for ranking of each user - earning in relation of balance
        values = np.array([(user.history_money[-1] - user.history_money[0]) / (user.history_money[0] + STARTING_MONEY) for user in ALL_USERS])
        precentiles = [np.percentile(values, i * 10) for i in range(1,11)] # class the balances to 10 diffrent ranks
        for i,value in enumerate(values): # for each user - check its rank according to the precentiles
            for j, prec in enumerate(precentiles):
                if value <= prec:
                    ALL_USERS[i].rank = j + 1
                    break

    def update_last_spendnding(self):
        """update the last spending list"""
        # notice there was a problem in this version of last spending noted above.
        indexes_to_delete = []
        for i, purchase in enumerate(self.last_spendings):
            purchase[1] -= 1 # a day passed
            if purchase[1] < 0:
                indexes_to_delete.append(i) # note to delete this asic because ASICS_DEATH_TIME has passed.
        temp_last_spendings = []
        for i in range(len(self.last_spendings)): # delete all of the wanted to delete asics
            if i not in indexes_to_delete:
                temp_last_spendings.append(self.last_spendings[i])
        self.last_spendings = temp_last_spendings


    def notify_of_asics(self, do_print):
        '''RUNS ON ALL ASICS AND DECIDES IF IT WORTH ACTIVITING AND IF SO ON WHICH COIN'''
        total_mining_invested = [0 for i in range(len(self.pools))] # list that updates within loops. it represents total hash rate invested so far in each pool.
        do_print = False
        total_money_invested = 0 #all electrecity money spent so far
        coins_last_day_hash_rates = [sum(pool["pool"].coin.last_days_hash_rates) / EASY_DAYS for pool in self.pools] # for each pool - average of  total hash rate of last EASY_DAYS
        coins_last_day_values = [sum(pool["pool"].coin.last_days_coin_values) / EASY_DAYS for pool in self.pools] # for each pool - average of coin value of last EASY_DAYS
        owner_last_day_hash_rates = [sum(pool["last_days_hash_rates"]) / EASY_DAYS for pool in self.pools] # for each pool - average of self invested hash rate of last EASY_DAYS
        for asic in self.asics: # for each asic
            if asic.get_elec_price() + total_money_invested < self.money: # if you have money to buy its electricity
                # now we want to find the pool that is the most effective to invest in for this current asic
                max_pool_index = -1
                max_pool_value = - 10 ** 10  # init vars
                for i, pool_object in enumerate(self.pools):
                    pool = pool_object["pool"]
                    owners_last_day_hashrate = owner_last_day_hash_rates[i]
                    # the next calculation is the worth of investing a specific asic machine on a specific pool.
                    # IMPORTENT - ON YAISH'S PAPER THERE IS A MORE ACCURATE CALCULATION FOR THAT.
                    val = coins_last_day_values[i] * asic.get_hash_rate() / (coins_last_day_hash_rates[i] - owners_last_day_hashrate + total_mining_invested[i] + asic.get_hash_rate()) - asic.get_elec_price()
                    if do_print:
                        print(val, asic.get_hash_rate(), coins_last_day_values[i])
                    if val > max_pool_value:
                        max_pool_value = val
                        max_pool_index = i
                # if it is worth it to invest the current asic  on the best pool - invest the asic
                if val > 0:
                    pool = self.pools[max_pool_index]["pool"]
                    pool.notify_of_asic(self, asic)
                    total_mining_invested[max_pool_index] += asic.get_hash_rate()
                    total_money_invested += asic.get_elec_price()
        for i,pool_object in enumerate(self.pools): # update the self.pools list with the current hash rates invested
            pool_object["last_days_hash_rates"] = pool_object["last_days_hash_rates"][1:] + [total_mining_invested[i]]


def get_max_k(lst, k):
    '''
    returns the indexes of the k maximal elements of lst and the value of the k+1'th maximal element.
    this function is an helper function of the auction, in the auction language - the function returns the auction winners and the price.
    '''
    if len(lst) <= k:
        return list(range(len(lst))), 0
    max_vals = [0 for _ in range(k+1)]
    max_indexes = [0 for _ in range(k+1)]
    for i, val in enumerate(lst):
        if val > min(max_vals):
            index_replaced = max_vals.index(min(max_vals))
            max_vals[index_replaced] = val
            max_indexes[index_replaced] = i
    # addition for the end auction funtion
    price = min(max_vals)
    max_indexes.pop(max_vals.index(price))
    # end of addition
    return max_indexes , price

class Auction:
    """a class of an auction. it has some amount of asics that can be bought with auction rules.
    the auction assumes that all of the asics are the same.
    users can bid money on the auction, and can win up to 1 asic machine.
    if there are k asics machines on the auction, the k highest bidders win it with the k+1's bid price."""

    def __init__(self, asics):
        self.bids = []
        self.winners_num = len(asics)
        self.asics = asics
        self.open = True

    def send_bid(self, user, money):
        """users calls this function to send bid"""
        if self.open:
            self.bids.append({"user" : user, "money" : money})

    def end_auction(self):
        '''
        returns the unspent money to its owner's. gives the asics to the winners.
        '''
        if self.open:
            winners_indexes, price = get_max_k([bid["money"] for bid in self.bids], self.winners_num)
            counter = 0
            for i, bid in enumerate(self.bids):
                if i in winners_indexes:
                    bid["user"].money -= price
                    self.asics[counter].set_owner(bid["user"])
                    bid["user"].add_asic(self.asics[counter], price)
                    counter += 1
            self.open = False
            return price
        return 0


class Asic:
    """ a class of asics."""
    def __init__(self, id, type = "normal"):
        self.id = id
        self.type = type # remember the dictionery at the start of the code - the type determents the hash rate, electricity price and so on...
        self.num_of_usages = 0 # for the asics hash rate decay and distruction of an asic
        self.user = None
        self.dead = False

    def get_elec_price(self):
        """returns the asics electricity price"""
        return ASICS_ELMS[self.type]["elec_price"]

    def get_hash_rate(self):
        """returns the asics hash rate"""
        return ASICS_ELMS[self.type]["hash_rate"] * ASICS_ELMS[self.type]["decay_func"](self.num_of_usages) 

    def set_owner(self, user):
        """setting an owner to the asic"""
        self.user = user

    def notify_of_usage(self):
        """if an asic is used on a pool this function is called."""
        if self.user.money >= self.get_elec_price(): # use the asic only if the user has the money
            self.num_of_usages += 1
            self.user.money -= self.get_elec_price()
            if self.num_of_usages > ASIC_DEATH_TIME: # if the asics need to be distorcted
                self.dead = True
                self.user.notify_of_dead_asic(self)
            # updates relevent asic features
            return True
        return False

class Pool:
    """the class of the pool
    in the simulation each coin has only one pool - so effectivly, the price of each asic machine from the pool at one
    turn is coin value * asic hash rate / total hash rate"""

    def __init__(self, coin):
        self.coin = coin
        self.total_hash_rate = 0
        self.asics = []
        self.coin.notify_of_pool(self)
        self.last_days_hash_rates = [0.0001 for _ in range(EASY_DAYS)] # the hash rate of the pool in the last EASY_DAYS

    def notify_of_asic(self, user, asic):
        """"when an asic has enterd the pool."""
        if asic.notify_of_usage(): # notify the asic that it has been used, if the user can pay for its electricity go on
            self.asics.append({"user" : user, "asic" : asic})
        #self.update_hash_rate(asic.get_hash_rate())

    def remove_asic(self, asic): #irelevent function
        remove_index = 0
        for i, self_asic in enumerate(self.asics):
            if asic == self.asic:
                remove_index = i
        #self.update_hash_rate(-self.asics[remove_index]["asic"].get_hash_rate())
        self.asics.remove(self.asics[remove_index])

    def update_hash_rate(self, rate_addition): # irelevent funciton
        self.total_hash_rate += rate_addition

    def get_prize(self, prize): # if the pool won the block this function is called. (there is only one pool so happens every time)
        for asic in self.asics:
            asic["user"].money += prize * asic["asic"].get_hash_rate() / self.total_hash_rate # give money according to the user's hash rate

    def refresh_hashrate(self): #USE ONCE A DAY
        """ this function calculate the hashrate of the pool at the end of the stage when the users decide where to put their asics.
        this function updates the hash rate in the history hashrates"""
        self.total_hash_rate = sum(asic["asic"].get_hash_rate() for asic in self.asics)
        self.last_days_hash_rates = self.last_days_hash_rates[1:] + [self.total_hash_rate]

    def clear(self):
        """HAPPENDS EVERY END OF DAY clearing all the asics in the pool"""
        self.asics = []

class Coin:
    """class of a coin.
    coin value start from an initial point and then increases or decreases by a multiplyer (pos jump, neg jump)
    in the first ASIC DEATH TIME days the coin value increases as well linarly so that in the ASIC DEATH TIME day
    the coin value will go around the initial value. i added this to the simulation because in the first days there are
    not many asics active, and therefor if i wont do that, the first asics buyers will be too rich."""

    def __init__(self, id, value, pos_jump, neg_jump, basic_hash=0):
        self.t = 0 #day
        self.background_value = value #the actual value that was suposed to be with out the first day linear increase
        self.value = 0
        self.pools = []
        self.basic_hash = basic_hash
        self.hash_rate = basic_hash
        id = id
        self.pos_jump = pos_jump
        self.neg_jump = neg_jump
        self.last_days_hash_rates = [0.0001 for _ in range(EASY_DAYS)]
        self.last_days_coin_values = [0 for _ in range(EASY_DAYS)] # the last EASY_DAYS coin value and total hash rate

    def notify_of_pool(self, pool): # simple
        """add pool to this coin"""
        self.pools.append(pool)

    def update_value(self):
        """update the value of the coin"""
        if self.t < ASIC_DEATH_TIME: # update the day
            self.t += 1
        if random.random() < 0.5: #update the backgound value
            self.background_value *= self.pos_jump
        else:
            self.background_value *= self.neg_jump
        self.value = self.background_value * self.t / ASIC_DEATH_TIME # get the current value
        # notice that after the ASIC DEATH TIME day self.t will always be equal to ASIC_DEATH_TIME so that value will be equal to background value


    def get_hash_rate(self): # calculate the hashrate
        """calculates the coins hash rate"""
        return (sum(pool.total_hash_rate for pool in self.pools) + self.basic_hash)

    def end_day(self):
        """smart function that splits the price between the pools, there is only one pool so it is not that relevant"""
        winable_hashrates = self.get_hash_rate()
        self.last_days_hash_rates = self.last_days_hash_rates[1:] + [winable_hashrates]
        self.last_days_coin_values = self.last_days_coin_values[1:] + [self.value]
        if winable_hashrates == 0:
            return None
        for pool in self.pools:
            pool_chance = pool.total_hash_rate / winable_hashrates
            if pool_chance >= random.random():
                pool.get_prize(self.value) # tells the pool it won the price
                return 0
            winable_hashrates -= pool.total_hash_rate

def auction_asics(serial_number_start, num_of_asics, users):
    """this function is a part of the main loop of the simulation
    it starts an auction, gets the user's bids and give the asics to the winners. (and let them pay the price)"""
    auc = Auction([Asic(i + serial_number_start) for i in range(num_of_asics)])
    # the global features of the learning
    global_features = [coin.value / BASE_COIN_VAL for coin in BITCOINS] + [coin.get_hash_rate() / len(ALL_USERS) for coin in BITCOINS]
    for user in users:
        user.bid(auc, user.get_bid(global_features))
    ALL_AUC_PRICES.append(auc.end_auction())

def auction_asics_no_hidden(serial_number_start, num_of_asics, users):
    """this function is a part of the main loop of the simulation
    it starts an auction, gets the user's bids and give the asics to the winners. (and let them pay the price)"""
    auc = Auction([Asic(i + serial_number_start) for i in range(num_of_asics)])
    # the global features of the learning
    global_features = [coin.value / BASE_COIN_VAL for coin in BITCOINS] + [coin.get_hash_rate() / len(ALL_USERS) for coin in BITCOINS]
    for user in users:
        user.bid(auc, user.get_bid_no_hidden_layer(global_features))
    ALL_AUC_PRICES.append(auc.end_auction())

def auction_asics_with_hidden(serial_number_start, num_of_asics, users):
    """this function is a part of the main loop of the simulation
    it starts an auction, gets the user's bids and give the asics to the winners. (and let them pay the price)"""
    auc = Auction([Asic(i + serial_number_start) for i in range(num_of_asics)])
    # the global features of the learning
    global_features = [coin.value / BASE_COIN_VAL for coin in BITCOINS] + [coin.get_hash_rate() / len(ALL_USERS) for coin in BITCOINS]
    for user in users:
        user.bid(auc, user.get_bid(global_features))
    ALL_AUC_PRICES.append(auc.end_auction())

# The simulator constants
NUM_GENERATIONS_START = 30
NUM_GENERATIONS_REAL = 70
DAYS_FOR_GENERATIONS_START = 40
DAYS_FOR_GENERATIONS_RAEL = 200
DAYS = 500 # days of the simulation
NEW_DAILY_USERS = 0 #unrelevant
ASICS_PER_AUC = 4 # amount of asics selling per auction
DAILY_AUCTIONS = 5 # amount of auctions per day
STARTING_MONEY = 30000 # starting money of each user - if higher the market is more stabelized but asics and decisions less mater.
STARTING_USERS = 200 # amount of users in the simulation - more users means more runtime but the simulation will be more realistic
NUM_OF_COINS = 4 # number of coins in the simulation
BASE_COIN_VAL = 500 #starting coin value
POS_COIN_JUMP = 1.005 # the possitive coin jump value
NEG_COIN_JUMP = 1/1.005 # the negetive coin jump value
BASE_HASH_RATE = 400 # the starting hash rate
NORMAL_DECAY_FACTOR = 0.98 # for normal asics - the decay factor of the exponential decay of hash rate function
ASICS_ELMS = {"normal" : {"elec_price" : 0.005, "hash_rate" : 1, "decay_func" : lambda t : NORMAL_DECAY_FACTOR ** t }} # dictionary:
# {asic type name : electricty price, initial hash rate, the decay function of the hash rate
ASIC_DEATH_TIME = 100 # the amount of uses that break asic - problem : the decay is exponential - after 90 uses the asic is pretty usless, therfore it wont be used and wont be broken.
LEARNING_START_TIME = 2 * ASIC_DEATH_TIME # the time when the ai of bidding on asics starts working - reason - you want
# the simulation to stabelize and only then to start the learning (at simulation 1)
EASY_DAYS = 35 # what is importent to know is that in the first EASY_DAYS the simulation does not make perfect sense.
ALL_USERS = [] #all of users in the simulation

#rellevant only for the first simulation
CHANCE_SELF_ADAPT_FUNC = lambda rank : random.random() < 0.4 / rank  #the lottery for mutation of a single weight
CHANCE_IMITATE_FUNC = lambda rank : random.random() < 0.02 / rank #the lottery for imitation of other user
VALUE_SELF_ADAPT_FUNC = lambda value, rank : value * random.uniform(1 + 2 / (rank + 1), 1 / (1 + (2 /(rank + 1)))) # the multipaction value of mutation
# THOSE FUNCTION CAN BE CHANGE WITH INTUATION WITH 2 QUESTIONS IN MIND:
#1. how much do you want the weights to change, high or low mutation and imitation?
#2. how much do you want the weight changing to depend on ranks - high ranks do not get changed or every rank will be changed similarly
MATRIX_INDEXES = [0] #it helps to understand and generalize the weights type of the simulation
# remember the bidding process - get atributes of simulation - apply the weights on the atributes - get the bid.
# the weight can be linear but in this case there is a hidden layer on the weights. therefor the first weights layer is a matrix
# from the input layer to the hidden layer. (optional google of hidden layer)


# creating the objects for the simulation:
BITCOINS = [Coin(0, BASE_COIN_VAL, POS_COIN_JUMP, NEG_COIN_JUMP, basic_hash=BASE_HASH_RATE) for _ in range(NUM_OF_COINS)]
BIT_POOLS = [Pool(BITCOIN) for BITCOIN in BITCOINS]
basic_cromozome_len = 2 * len(BITCOINS) #+ 3
for i in range(STARTING_USERS):
    user = User(i, str(i),STARTING_MONEY)
for user in ALL_USERS:
    for BIT_POOL in BIT_POOLS:
        user.add_pool(BIT_POOL)

# creating the lists for the graphs.
ALL_COIN_VALS = [[] for _ in range(NUM_OF_COINS)] # remeber the coin values for all coins
ALL_TOTAL_HASH_RATES = [[] for _ in range(NUM_OF_COINS)] # rememver the total hash rates of all coins
ALL_ASICS_ACTIVE = [[] for _ in range(NUM_OF_COINS)] # remember the amount of asics active of all coins
ALL_AUC_PRICES = [] # rememver the price that the asics were sold for


def main_with_hidden_layer1():
    """the simulation no generations no hidden layer:)
    each day users buying """
    # staring the simulation
    for day in range(DAYS):
        if day % 1000 == 0:
            print(day)  # print each day
        for _ in range(DAILY_AUCTIONS): # do the dayly auctions
            auction_asics(day * ASICS_PER_AUC, ASICS_PER_AUC, ALL_USERS)
        ALL_USERS[0].class_end_day(day) # and day for the class - for ranks and maybe more
        for user in ALL_USERS: # decide where to put asics and updates the variables of the user
            user.notify_of_asics(False) #GET ASICS INTO THE POOL
            user.update_day()
        for i,BIT_POOL in enumerate(BIT_POOLS): # update the pools
            BIT_POOL.refresh_hashrate()
            ALL_ASICS_ACTIVE[i].append(len(BIT_POOL.asics)) # for plotting
        for i, BITCOIN in enumerate(BITCOINS): # update the coins
            BITCOIN.end_day() # give prize to winner
            BITCOIN.update_value()
            # for ploting
            ALL_TOTAL_HASH_RATES[i].append(BITCOIN.get_hash_rate())
            ALL_COIN_VALS[i].append(BITCOIN.value)
        for BIT_POOL in BIT_POOLS: # end the day in the pools
            BIT_POOL.clear()
        ALL_COIN_VALS.append(BITCOIN.value) # for plooting
        # ploting
        if day == DAYS - 1: # you can choose the condition for graph ploting -  can be every 100 days and so on
            plt.plot(list(range(DAILY_AUCTIONS * (day + 1))), ALL_AUC_PRICES, label = "all auc prices") # all auction prices
            plt.legend()
            plt.show()
            plt.plot([len(user.asics) for user in ALL_USERS],[user.money for user in ALL_USERS],"s") # users money in relation of amount of asics
            plt.plot([len(user.asics) for user in ALL_USERS],[user.history_money[-1] for user in ALL_USERS],"s") # users effective money in relation of amount of asics
            plt.show()
            for i in range(NUM_OF_COINS): # the coin values in each day
                plt.plot(list(range((day + 1))), ALL_COIN_VALS[i], label = "coin values " +str(i))
            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the coins hash rates in each day
                plt.plot(list(range(day + 1)), ALL_TOTAL_HASH_RATES[i], label="hash rates " +str(i))
            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the amount of asics active in each coin in each day
                plt.plot(list(range(day + 1)), ALL_ASICS_ACTIVE[i], label="asics active " + str(i))
            plt.legend()
            plt.show()

def main_with_hidden_layer2():
    """the simulation with generations and with hidden layer:)
    each generation we run the simulation some amount of days, each day users buy asics and activating their asics.
    At the end of the generation we restart each users money and property and doing the learning."""
    users_size = (STARTING_USERS, basic_cromozome_len)
    users_biding_whieghts = [np.random.uniform(low=-5000.0, high=5000.0, size=(STARTING_USERS,
                                                                               basic_cromozome_len, basic_cromozome_len)
                                               ), np.random.uniform(low=-5000.0, high=5000.0,
                                                                    size=users_size)] #initial bidding whieghts
    # updating the users private weights
    for i, user in enumerate(ALL_USERS):
        user.update_weights_hidden_layer(users_biding_whieghts[0][i], users_biding_whieghts[1][i])
    # the first run of the simulation (the generations)
    for generation in range(NUM_GENERATIONS_START):
        print(generation)
        # making the users buy asics and earn money acordind to their biding weights
        for day in range(DAYS_FOR_GENERATIONS_START):
            for _ in range(DAILY_AUCTIONS):  # do the dayly auctions
                auction_asics_with_hidden(((generation * DAYS_FOR_GENERATIONS_START) + day) * ASICS_PER_AUC, ASICS_PER_AUC, ALL_USERS)
            for user in ALL_USERS:  # decide where to put asics and updates the variables of the user
                user.notify_of_asics(False)  # GET ASICS INTO THE POOL
                user.update_day_GA_class()
            for i, BIT_POOL in enumerate(BIT_POOLS):  # update the pools
                BIT_POOL.refresh_hashrate()
                ALL_ASICS_ACTIVE[i].append(len(BIT_POOL.asics))  # for plotting
            for i, BITCOIN in enumerate(BITCOINS):  # update the coins
                BITCOIN.end_day()  # give prize to winner
                BITCOIN.update_value()
                # for ploting
                ALL_TOTAL_HASH_RATES[i].append(BITCOIN.get_hash_rate())
                ALL_COIN_VALS[i].append(BITCOIN.value)
            for BIT_POOL in BIT_POOLS:  # end the day in the pools
                BIT_POOL.clear()
            ALL_COIN_VALS.append(BITCOIN.value)  # for plooting
        # the learning:
        fitness = GA_HL.cal_users_fitness(DAYS_FOR_GENERATIONS_START)
        parents = GA_HL.select_mating_pool(users_biding_whieghts, fitness, STARTING_USERS / 2)
        offspring_crossover = GA_HL.crossover(parents,
                                              offspring_size=(
                                              users_size[0] - parents[0].shape[0], basic_cromozome_len))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GA_HL.mutation(offspring_crossover)
        # Creating the new population based on the parents and offspring.
        users_biding_whieghts[0][0:parents[0].shape[0], :] = parents[0]
        users_biding_whieghts[0][parents[0].shape[0]:, :] = offspring_mutation[0]
        users_biding_whieghts[1][0:parents[1].shape[0], :] = parents[1]
        users_biding_whieghts[1][parents[1].shape[0]:, :] = offspring_mutation[1]
        # updating whieghts, and restarting users belongings
        for i, user in enumerate(ALL_USERS):
            user.update_weights_hidden_layer(users_biding_whieghts[0][i], users_biding_whieghts[1][i])
            user.restart_users_property()
    # the second run of the simulation (the generations)
    for generation in range(NUM_GENERATIONS_REAL):
        print(generation)
        # making the users buy asics and earn money acordind to their biding weights
        for day in range(DAYS_FOR_GENERATIONS_RAEL):
            for _ in range(DAILY_AUCTIONS):  # do the dayly auctions
                auction_asics_with_hidden(((generation * DAYS_FOR_GENERATIONS_START) + day) * ASICS_PER_AUC,
                                          ASICS_PER_AUC, ALL_USERS)
            for user in ALL_USERS:  # decide where to put asics and updates the variables of the user
                user.notify_of_asics(False)  # GET ASICS INTO THE POOL
                user.update_day_GA_class()
            for i, BIT_POOL in enumerate(BIT_POOLS):  # update the pools
                BIT_POOL.refresh_hashrate()
                ALL_ASICS_ACTIVE[i].append(len(BIT_POOL.asics))  # for plotting
            for i, BITCOIN in enumerate(BITCOINS):  # update the coins
                BITCOIN.end_day()  # give prize to winner
                BITCOIN.update_value()
                # for ploting
                ALL_TOTAL_HASH_RATES[i].append(BITCOIN.get_hash_rate())
                ALL_COIN_VALS[i].append(BITCOIN.value)
            for BIT_POOL in BIT_POOLS:  # end the day in the pools
                BIT_POOL.clear()
            ALL_COIN_VALS.append(BITCOIN.value)  # for plooting
        # the learning:
        fitness = GA_HL.cal_users_fitness(DAYS_FOR_GENERATIONS_START)
        parents = GA_HL.select_mating_pool(users_biding_whieghts, fitness, STARTING_USERS / 2)
        offspring_crossover = GA_HL.crossover(parents,
                                              offspring_size=(
                                                  users_size[0] - parents[0].shape[0], basic_cromozome_len))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GA_HL.mutation(offspring_crossover)
        # Creating the new population based on the parents and offspring.
        users_biding_whieghts[0][0:parents[0].shape[0], :] = parents[0]
        users_biding_whieghts[0][parents[0].shape[0]:, :] = offspring_mutation[0]
        users_biding_whieghts[1][0:parents[1].shape[0], :] = parents[1]
        users_biding_whieghts[1][parents[1].shape[0]:, :] = offspring_mutation[1]
        # updating whieghts, and restarting users belongings
        for i, user in enumerate(ALL_USERS):
            user.update_weights_hidden_layer(users_biding_whieghts[0][i], users_biding_whieghts[1][i])
            user.restart_users_property()
        # ploting
        if generation == NUM_GENERATIONS_REAL - 1: # you can choose the condition for graph ploting -  can be every 100 days and so on
            a_sum = 0
            for i in range(1, 11):
                a_sum = a_sum + ALL_AUC_PRICES[-1*i]
            avg = a_sum / 10
            plt.plot(list(range(len(ALL_AUC_PRICES))), ALL_AUC_PRICES, label = "all auc prices. " + "10 last auc avg prices" + str(avg)) # all auction prices
            plt.legend()
            plt.show()
            plt.plot([len(user.asics) for user in ALL_USERS],[user.money for user in ALL_USERS],"s") # users money in relation of amount of asics
            plt.plot([len(user.asics) for user in ALL_USERS],[user.history_money[-1] for user in ALL_USERS],"s") # users effective money in relation of amount of asics
            plt.show()
            for i in range(NUM_OF_COINS): # the coin values in each day
                plt.plot(list(range((((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1))), ALL_COIN_VALS[i], label = "coin values " +str(i))
            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the coins hash rates in each day
                plt.plot(list(range(((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1)), ALL_TOTAL_HASH_RATES[i], label="hash rates " +str(i))
            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the amount of asics active in each coin in each day
                plt.plot(list(range(((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1)), ALL_ASICS_ACTIVE[i], label="asics active " + str(i))
            plt.legend()
            plt.show()

def main_without_hidden_layer():
    """the simulation with generations, without hidden layer:)
    each generation we run the simulation some amount of days, each day users buy asics and activating their asics.
    At the end of the generation we restart each users money and property and doing the learning."""
    users_size = (STARTING_USERS, basic_cromozome_len)
    users_biding_whieghts = np.random.uniform(low=-500.0, high=500.0, size=users_size)#initial bidding whieghts
    # updating the users private weights
    for i, user in enumerate(ALL_USERS):
        user.update_weights(users_biding_whieghts[i])
    # the first run of the simulation (the generations)
    for generation in range(NUM_GENERATIONS_START):
        print(generation)
        # making the users buy asics and earn money acordind to their biding weights
        for day in range(DAYS_FOR_GENERATIONS_START):
            for _ in range(DAILY_AUCTIONS):  # do the dayly auctions
                auction_asics_no_hidden(((generation * DAYS_FOR_GENERATIONS_START) + day) * ASICS_PER_AUC, ASICS_PER_AUC, ALL_USERS)
            for user in ALL_USERS:  # decide where to put asics and updates the variables of the user
                user.notify_of_asics(False)  # GET ASICS INTO THE POOL
                user.update_day_GA_class()
            for i, BIT_POOL in enumerate(BIT_POOLS):  # update the pools
                BIT_POOL.refresh_hashrate()
                ALL_ASICS_ACTIVE[i].append(len(BIT_POOL.asics))  # for plotting
            for i, BITCOIN in enumerate(BITCOINS):  # update the coins
                BITCOIN.end_day()  # give prize to winner
                BITCOIN.update_value()
                # for ploting
                ALL_TOTAL_HASH_RATES[i].append(BITCOIN.get_hash_rate())
                ALL_COIN_VALS[i].append(BITCOIN.value)
            for BIT_POOL in BIT_POOLS:  # end the day in the pools
                BIT_POOL.clear()
            ALL_COIN_VALS.append(BITCOIN.value)  # for plooting
        # the learning:
        fitness = GA.cal_users_fitness(DAYS_FOR_GENERATIONS_START)
        parents = GA.select_mating_pool(users_biding_whieghts, fitness, STARTING_USERS / 2)
        offspring_crossover = GA.crossover(parents,
                                           offspring_size=(users_size[0] - parents.shape[0], basic_cromozome_len))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)
        # Creating the new population based on the parents and offspring.
        users_biding_whieghts[0:parents.shape[0], :] = parents
        users_biding_whieghts[parents.shape[0]:, :] = offspring_mutation
        # updating whieghts, and restarting users belongings
        for i, user in enumerate(ALL_USERS):
            user.update_weights(users_biding_whieghts[i])
            user.restart_users_property()
    # the second run of the simulation (the generations)
    for generation in range(NUM_GENERATIONS_REAL):
        print(generation)
        # making the users buy asics and earn money acordind to their biding weights
        for day in range(DAYS_FOR_GENERATIONS_RAEL):
            for _ in range(DAILY_AUCTIONS):  # do the dayly auctions
                auction_asics_no_hidden(((generation * DAYS_FOR_GENERATIONS_RAEL) + day) * ASICS_PER_AUC,
                                        ASICS_PER_AUC, ALL_USERS)
            for user in ALL_USERS:  # decide where to put asics and updates the variables of the user
                user.notify_of_asics(False)  # GET ASICS INTO THE POOL
                user.update_day_GA_class()
            for i, BIT_POOL in enumerate(BIT_POOLS):  # update the pools
                BIT_POOL.refresh_hashrate()
                # if generation % (NUM_GENERATIONS_REAL // 10) == 0:
                ALL_ASICS_ACTIVE[i].append(len(BIT_POOL.asics))  # for plotting
            for i, BITCOIN in enumerate(BITCOINS):  # update the coins
                BITCOIN.end_day()  # give prize to winner
                BITCOIN.update_value()
                # for ploting
                # if generation % (NUM_GENERATIONS_REAL // 10) == 0:
                ALL_TOTAL_HASH_RATES[i].append(BITCOIN.get_hash_rate())
                ALL_COIN_VALS[i].append(BITCOIN.value)
            for BIT_POOL in BIT_POOLS:  # end the day in the pools
                BIT_POOL.clear()
            # if generation % (NUM_GENERATIONS_REAL // 10) == 0:
            ALL_COIN_VALS.append(BITCOIN.value)  # for plooting
        # the learning:
        fitness = GA.cal_users_fitness(DAYS_FOR_GENERATIONS_RAEL)
        parents = GA.select_mating_pool(users_biding_whieghts, fitness, STARTING_USERS / 2)
        offspring_crossover = GA.crossover(parents,
                                           offspring_size=(
                                               users_size[0] - parents.shape[0], basic_cromozome_len))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)
        # Creating the new population based on the parents and offspring.
        users_biding_whieghts[0:parents.shape[0], :] = parents
        users_biding_whieghts[parents.shape[0]:, :] = offspring_mutation
        # updating whieghts, and restarting users belongings
        for i, user in enumerate(ALL_USERS):
            user.update_weights(users_biding_whieghts[i])
            user.restart_users_property()
        # ploting
        if generation == NUM_GENERATIONS_REAL - 1: # you can choose the condition for graph ploting -  can be every 100 days and so on
            print("last auction price: ",ALL_AUC_PRICES[-1])
            print(ALL_AUC_PRICES)
            # plt.plot(list(range(DAILY_AUCTIONS * (((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1))), ALL_AUC_PRICES, label = "all auc prices") # all auction prices
            a_sum = 0
            for i in range(1, 11):
                a_sum = a_sum + ALL_AUC_PRICES[-1*i]
            avg = a_sum / 10
            plt.plot(list(range(len(ALL_AUC_PRICES))), ALL_AUC_PRICES, label = "all auc prices. " + "10 last auc avg prices" + str(avg)) # all auction prices

            plt.legend()
            plt.show()
            plt.plot([len(user.asics) for user in ALL_USERS],[user.money for user in ALL_USERS],"s") # users money in relation of amount of asics
            plt.plot([len(user.asics) for user in ALL_USERS],[user.money for user in ALL_USERS],"s") # users money in relation of amount of asics

            plt.plot([len(user.asics) for user in ALL_USERS],[user.history_money[-1] for user in ALL_USERS],"s") # users effective money in relation of amount of asics
            plt.plot([len(user.asics) for user in ALL_USERS],[user.history_money[-1] for user in ALL_USERS],"s") # users effective money in relation of amount of asics

            plt.show()
            for i in range(NUM_OF_COINS): # the coin values in each day
                # plt.plot(list(range((((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1))), ALL_COIN_VALS[i], label = "coin values " +str(i))
                plt.plot(list(range(len(ALL_COIN_VALS[i]))), ALL_COIN_VALS[i], label = "coin values " +str(i))

            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the coins hash rates in each day
                # plt.plot(list(range(((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1)), ALL_TOTAL_HASH_RATES[i], label="hash rates " +str(i))
                plt.plot(list(range(len(ALL_TOTAL_HASH_RATES[i]))), ALL_TOTAL_HASH_RATES[i], label="hash rates " +str(i))

            plt.legend()
            plt.show()
            for i in range(NUM_OF_COINS): # the amount of asics active in each coin in each day
                # plt.plot(list(range(((generation * DAYS_FOR_GENERATIONS_RAEL) + day) + NUM_GENERATIONS_START*DAYS_FOR_GENERATIONS_START + 1)), ALL_ASICS_ACTIVE[i], label="asics active " + str(i))
                plt.plot(list(range(len(ALL_ASICS_ACTIVE[i]))), ALL_ASICS_ACTIVE[i], label="asics active " + str(i))
            plt.legend()
            plt.show()

# main_without_hidden_layer()
main_with_hidden_layer2()
# main_with_hidden_layer1()

#DONE 

#IDEA: CREATE COINS BASED ON REAL COINS
# ASICS WITH SPECIFIC COINS OR INTERACTION WITH MINING ON FEW COINS.
#