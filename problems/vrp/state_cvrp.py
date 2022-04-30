import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    ## INPUT HAS GRID ID    OUTPUT HAS SUBAREA

    ## Question: what does each step represent? Is it a different day or is it delivery courier moving to a different grid ID to drop off a package?

    # Fixed input
    # average gps from all the grid ids in one subarea

    coords: torch.Tensor  # Depot + loc #### Every Grid ID or subarea # gps

    # demand: torch.Tensor ##### we can remove this
    morning_amount: torch.Tensor  # analogous to demand
    afternoon_amount: torch.Tensor  # analogous to demand
    final_time: torch.Tensor
    final_location: torch.Tensor
    arrival_time: torch.Tensor
    subarea_dic: torch.Tensor

    # coord.id = id coord.lat = lattitude coord.long = longitude coord.morning_amount coord.afternoon_amount .....

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor  ## keeps track of previous state
    # used_capacity: torch.Tensor ## related to the demand - so can be removed?
    is_busy_in_morning: torch.Tensor  # adding
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor  ## self explanatory
    i: torch.Tensor  # Keeps track of step

    # each step is a new subarea, check subarea_dic to store all grids into the coords

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):  ## discuss at meeting
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):  ## only run once per instance of the cvrp?
        ## input is a dictionary with depot, loc, and demand.
        depot = input['depot']  ## coord of the station
        loc = input['loc']  ## coords of all of the grid IDS
        demand = input['demand']  ## can remove this part

        batch_size, n_loc, _ = loc.size()  ## n_loc = number of coords?
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),  ## brings together depot coord with all of the loc coords
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),  ##  batchsize vector with lengths
            cur_coord=input['depot'][:, None, :],
            # Add step dimension ## set cur_coord to the station due to starting point
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    ## not used
    def get_final_cost(self):  ## check if everything has run thru, if so then add final norm of coords to total.

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    ###################################### read and change
    def update(self, selected):  #### grid.subarea need

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2,
                                                                   dim=-1)  # (batch_dim, 1) ## finds the distance between the prev and curr coord

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        # selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]  ## remove demand

        #### DO NOT NEED NOW
        # Increase capacity if depot is not visited, otherwise set to 0
        # used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        # used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()  ## related to demand, remove it

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(  ## updates all of the instance variables at the top and increments the iteration id.
            prev_a=prev_a,  visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.coords.size(-2) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    ## visited vector is the batch size number of subareas
    def get_mask(
            self):  ## need to mask all grid ID from one subarea. search grid every grid ID to see if it belongs in subarea.
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        # exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :,
        #                                           None] > self.VEHICLE_CAPACITY)  ## need to change this to remove demand
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc ## apply bitwise OR

        ##### Ask question about how the dtypes factor into these bitwise operations ^ v
        ## The tensors need to be the same.

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  ## apply bitwise AND
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions