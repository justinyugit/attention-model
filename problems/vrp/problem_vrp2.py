from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search



class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    ## used for deliverytime_calculate below
    @staticmethod
    def getDist(lat1, lon1, lat2, lon2):
        if lon1 == lon2 and lat1 == lat2:
            return 0.0
        theta = lon1 - lon2
        dist = math.sin(deg2rad(lat1)) * math.sin(deg2rad(lat2)) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.cos(deg2rad(theta))
        if dist > 1.0:
            dist = 1.0
        dist = rad2deg(math.acos(dist)) * 60 * 1.1515 * 1.609344
        return dist
    @staticmethod
    def deg2rad(deg):
        return deg * math.pi / 180.0
    @staticmethod
    def rad2deg(rad):
        return rad * 180.0 / math.pi

    ## needed for relay_time_func below
    #返回下午订单的完成时间
    @staticmethod
    def deliverytime_calculate(l,afternoon_amount,start, finaltime):  #l: 订单所在的grid组成的list, afternoon_amount: grid的订单量，start:起始点的grid，final_time：起始点的时间
        t_order=120  #每一单的配送时间
        current_loc=start
        time_result=finaltime
        while len(l)>0:
            mini=getDist(GPS_location[start][0],GPS_location[start][1],GPS_location[l[0]][0],GPS_location[l[0]][1])*1000/v
            nxt=l[0]
            for i in range(len(l)):
                if getDist(GPS_location[start][0],GPS_location[start][1],GPS_location[l[i]][0],GPS_location[l[i]][1])*1000/v<mini:
                    mini=getDist(GPS_location[start][0],GPS_location[start][1],GPS_location[l[i]][0],GPS_location[l[i]][1])*1000/v
                    nxt=l[i]
            amt=afternoon_amount[nxt]
            time_result=datetime.timedelta(seconds=mini+t_order*amt)+time_result
            l.remove(nxt)
        return time_result

    ## needed for get_costs below
    @staticmethod
    def relay_time_func(route, GPS_location, v_relay, dic_final, morning_amount, afternoon_amount, arrival_time, final_time, final_location):
        t_start="2022-01-01 12:00:00" 
        t_start=datetime.datetime.strptime(t_start, "%Y-%m-%d %H:%M:%S")
        relay_location=(31.25971,121.4227)
        relay_time="2022-01-01 12:00:00" 
        relay_time=datetime.datetime.strptime(relay_time, "%Y-%m-%d %H:%M:%S")
        total_time=0
        for i in range(len(route)):
            l=dic_final[route[i]] #每个路区里的格子
            l_final=[]
            sum_amount=0
            for j in l:
                if afternoon_amount[j]>0:
                    sum_amount=sum_amount+afternoon_amount[j]
                    l_final.append(j)

        
        
            t_temp=getDist(GPS_location[final_location[route[i]]][0],GPS_location[final_location[route[i]]][1],relay_location[0],relay_location[1])*1000/v_relay
            relay_arrival_t=relay_time+datetime.timedelta(seconds=t_temp)
            if (relay_arrival_t-final_time[route[i]]).total_seconds()>0:   #配送员等接驳员 
                total_time=total_time+(relay_arrival_t-t_start).total_seconds()*sum_amount    #从站点到配送员 
                time1=deliverytime_calculate(l_final,afternoon_amount,final_location[route[i]],relay_arrival_t)
                total_time=total_time+(time1-relay_arrival_t).total_seconds()  #从配送员到顾客
            
            
            else:  #接驳员等配送员 ->可以进行订单合并
                total_time=total_time+(final_time[route[i]]-t_start).total_seconds()*sum_amount
                l_final_morning=[] 
                for k in l:
                    if morning_amount[k]>0:
                        if arrival_time[k]>relay_arrival_t:
                            l_final_morning.append(k)
            
                #在下午的订单里不能合并的上午订单
                l_final_afternoon=[]
                for m in l_final:
                    for n in l_final_morning:
                        if getDist(GPS_location[m][0],GPS_location[m][1],GPS_location[n][0],GPS_location[n][1])>0.2:
                            l_final_afternoon.append(m)
                            continue
                #time1=deliverytime_calculate(l_final,afternoon_amount,final_location[route[i]],final_time[route[i]])
                time2=deliverytime_calculate(l_final_afternoon,afternoon_amount,final_location[route[i]],final_time[route[i]])
                #benefit=(time1-time2).total_seconds()
                total_time=total_time+(time2-final_time[route[i]]).total_seconds()
            
            relay_time=relay_arrival_t   
            #relay_location=central[final_location[route[i]]-1]
            relay_location=GPS_location[final_location[route[i]]]
        return total_time 


    ###### NEED TO CHANGE dataset_file parameter when its called as it is now a file and not a dictionary like below in get_costs_original()
    @staticmethod
    def get_costs(dataset_file, route):
        df_result_new=pd.read_csv('data sample 3000.csv')

        for _idx, _row in df_result_new.iterrows():
            afternoon_amount=eval(_row['afternoon_amount'])
            sum_amount=0
            for value in afternoon_amount.values():
                sum_amount=sum_amount+value 
            aa=eval(_row['arrival_time'])
            for i in aa.keys():
             aa[i]=datetime.datetime.strptime(aa[i][:19], "%Y-%m-%d %H:%M:%S")
            bb=eval(_row['final_time'])
            for i in bb.keys():
                bb[i]=datetime.datetime.strptime(bb[i][:19], "%Y-%m-%d %H:%M:%S")
    
            reward=relay_time_func(route, eval(_row['GPS_location']), v_relay, dic_final, eval(_row['morning_amount']), eval(_row['afternoon_amount']), aa, bb, eval(_row['final_location']))/sum_amount
            return reward #the average relay orders' delivey time

    # pi is the solution path (matrix: batch_size, node_num)
    # dataset['demand'] coordinate (x,y)
    # pi : 0,3,2,5,1
    # dataset : (1,0)(3,2)(4,5)(7,1)(3,2)
    # output: solution path length 
    @staticmethod
    def get_costs_original(dataset, pi):
        batch_size, graph_size = dataset['demand'].size() ## dataset is input, rows: batch size cols: graph_size. But we don't need demand
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) ## sum all lengths between coords
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)


    ## no need to change, eval search method
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

## no need to look
class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args): ##used below in __init__ to generate num_samples # of dictionaries with loc, demand, depot, which will be stored in data var.
    #depot, loc, demand, capacity, *args = args
    loc, morning_amt, afternoon_amt, final_t, final_loc, arrival_t, *args = args
    #grid_size = 1
    #if len(args) > 0:
    #    depot_types, customer_types, grid_size = args
    return {
        #original code commented out
        #'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        #'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        #'depot': torch.tensor(depot, dtype=torch.float) / grid_size
        'loc' : torch.tensor(loc, dtype=torch.float)
        'morning_amt' : torch.tensor(morning_amt, dtype=torch.float) 
        'afternoon_amt' : torch.tensor(afternoon_amt, dtype=torch.float) 
        'arrival_t' : torch.tensor(arrival_t, dtype=torch.float)
        'final_t' : torch.tensor(final_t, dtype=torch.float) 
        'final_loc' : torch.tensor(final_loc, dtype=torch.float)
        'subarea_dic' : torch.tensor(subarea_dic, dtype=torch.float) 
        
    }



## very important 
class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None: ## we can input our dataset here via a pickle file.
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else: ## randomly generate data if no custom pickle file is inputed

            ########################################### vehicle capacities
            
            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = { ## changes capacity based on number of packages per sample specified.
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    ################# coordinates of the customers' homes that need to be visited
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    ################## how many packages
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size], 
                    
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples) ## data contains num_samples amount of dictionaries. each dictionary has loc with 50 coords.
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
