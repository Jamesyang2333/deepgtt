
import numpy as np
import torch, h5py, os
from collections import namedtuple
from itertools import groupby

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=0):
    """
    a (array[int32])
    add one to all elements of a
    """
    return np.concatenate((a + 1, [PAD]*(max_length - len(a))))

def pad_arrays(a, length=0):
    # pad to the longest seq_len of a batch or a specified length
    if length == 0:
        max_length = max(map(len, a))
    else:
        max_length = length
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.float32)
    return torch.LongTensor(a)

def pad_array_float(a, max_length, PAD=0):
    """
    a (array[int32])
    add one to all elements of a
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays_float(a, length=0):
    # pad to the longest seq_len of a batch or a specified length
    if length == 0:
        max_length = max(map(len, a))
    else:
        max_length = length
    a = [pad_array_float(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.FloatTensor(a)

def rle(l):
    return [k for k, g in groupby(l)]

def get_sub_trips(trips, times, indices):
    result_trips = []
    result_times = []
    result_info = []
    for i in range(len(trips)):
        trip = trips[i]
        time = times[i]
        trip_len = len(time)
        index = indices[i]
        subtrip_len = np.random.choice(max(1, trip_len-trip_len//2)) + trip_len//2 + 1
        start = np.random.choice(max(1, trip_len-subtrip_len+1))
        end = min(start+subtrip_len-1, trip_len)
        if index[end] - index[start] + 1 < 5:
            result_trips.append(trip)
            result_times.append(time[trip_len-1] - time[0])
            result_info.append([0, trip_len-1, time, trip, index])
        else:
            result_trips.append(trip[index[start]:index[end]+1])
            result_times.append(time[end] - time[start])
            result_info.append([start, end, time, trip, index])
    return result_trips, torch.tensor(result_times, dtype=torch.float32), result_info

#xs = np.array([np.r_[1], np.r_[1, 2, 3], np.r_[2, 1]])


class SlotData():
    def __init__(self, trips, times, ratios, S, distances, maxlen=200, indices=None, alltimes=None):
        """
        trips (n, *): each element is a sequence of road segments
        times (n, ): each element is a travel cost
        ratios (n, [number of gps points]): offset of each gps point
        S (138, 148) or (num_channel, 138, 148): traffic state matrix
        """
        ## filter out the trips that are too long for too short (containing)
        idx = [i for (i, trip) in enumerate(trips) if (len(rle(trip)) <= maxlen and len(rle(trip)) >=5)]
        self.trips = trips[idx]
        self.times = times[idx]
        if not (indices is None):
            self.indices = indices[idx]
        if not (alltimes is None):
            self.alltimes = alltimes[idx]
        self.ratios = ratios[idx]
        self.distances = distances[idx]
        self.S = torch.tensor(S, dtype=torch.float32)
        ## (1, num_channel, height, width)
        if self.S.dim() == 2:
            self.S.unsqueeze_(0).unsqueeze_(0)
        elif self.S.dim() == 3:
            self.S.unsqueeze_(0)
        ## re-arrange the trips by the length in reverse order
        idx = argsort(self.trips)
        # self.trips type: list of list
        # why not converting trips to torch tensor? each entry has variable length
        self.trips = self.trips[idx]
        if not (alltimes is None):
            self.alltimes = self.alltimes[idx]
        # self.times type: torch tensor
        self.times = torch.tensor(self.times[idx], dtype=torch.float32)
        # self.midtimes type: list of list
        if not (indices is None):
            self.indices = self.indices[idx]
        # self.ratios type: torch tensor
#         self.ratios = torch.tensor(self.ratios[idx], dtype=torch.float32)
        # self.ratios type: list of list
        # why not converting trips to torch tensor? each entry has variable length
        self.ratios = self.ratios[idx]
        # self.distances type: torch tensor
        self.distances = torch.tensor(self.distances[idx], dtype=torch.float32)

        self.ntrips = len(self.trips)
        self.start = 0


    def random_emit(self, batch_size):
        """
        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        SD = namedtuple('SD', ['trips', 'times', 'ratios', 'S', 'distances', 'offset'])
        start = np.random.choice(max(1, self.ntrips-batch_size+1))
        end = min(start+batch_size, self.ntrips)
        
        # remember that we added one to all road ids to distinguish 0
        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        distances = self.distances[start:end]
        
        ratios = torch.ones(trips.shape)
        # code to enable offset
#         offset_pad = pad_arrays_float(self.ratios[start:end])
#         ratios[:, 0] = torch.ones(list(trips.shape)[0]) - offset_pad[:, 0]
#         row_idx = list(range(trips.shape[0]))
#         col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
#         # get the index of the last offset value for each trip
#         col_idx_ratio = list(map(lambda t:len(t)-1, self.ratios[start:end]))
#         ratios[row_idx, col_idx] = offset_pad[row_idx, col_idx_ratio]
        
        return SD(trips=trips, times=times, ratios=ratios, S=self.S, distances=distances, offset=self.ratios[start:end])
    
    def random_emit_sub(self, batch_size):
        """
        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        SD = namedtuple('SD', ['trips', 'times', 'ratios', 'S', 'distances', 'info', 'offset'])
        start = np.random.choice(max(1, self.ntrips-(batch_size//2)+1))
        end = min(start+(batch_size//2), self.ntrips)
#         trip_rle = [rle(trip) for trip in self.trips[start:end]]
        trip_rle = self.trips[start:end]
        max_length = max(map(len, trip_rle))
        sub_trips, sub_times, info= get_sub_trips(self.trips[start:end], self.alltimes[start:end], self.indices[start:end])
        # convert the list of subtrips (variable length) to 2 dimensional torch tensors
        # torch.Size([50, max_length])
        sub_trips_pad = pad_arrays(sub_trips, max_length)
        # concatenate two torch tensors
        # trips: list of torch tensors (len 150)
        trips = torch.cat((pad_arrays(trip_rle), sub_trips_pad), dim=0)
        # 2-dimensional numpy array
        # (150, )
        times = torch.cat((self.times[start:end], sub_times), dim=0)
        # 2-dimensional numpy array
        # (150, )
        distances = torch.cat((self.distances[start:end, 0], self.distances[start:end, 1], self.distances[start:end, 2]))
        
        ratios = torch.ones(trips.shape)
        # code to enable offset
#         # set the ratio for the whole trips
#         offset_pad = pad_arrays_float(self.ratios[start:end])
#         row_idx_1 = list(range(trips.shape[0] // 2))
#         col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
#         # get the index of the last offset value for each trip
#         col_idx_ratio = list(map(lambda t:len(t)-1, self.ratios[start:end]))
#         ratios[row_idx_1, 0] = torch.ones(trips.shape[0] // 2) - offset_pad[:, 0]
#         ratios[row_idx_1, col_idx] = offset_pad[row_idx_1, col_idx_ratio]
#         # set the ratio for the subtrips
#         row_idx_2 = list(range(trips.shape[0] // 2, trips.shape[0]))
#         col_idx_2 = list(map(lambda t:len(t)-1, sub_trips))
#         ratios[row_idx_2, 0] = torch.ones(trips.shape[0] // 2) - offset_pad[row_idx_1, [record[0] for record in info]]
#         ratios[row_idx_2, col_idx_2] = offset_pad[row_idx_1, [record[1] for record in info]]
        
        return SD(trips=trips, times=times, ratios=ratios, S=self.S, distances=distances, info=info, offset=self.ratios[start:end])

    def order_emit(self, batch_size):
        """
        Reset the `start` every time the current slot has been traversed
        and return none.

        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        if self.start >= self.ntrips:
            self.start = 0
            return None
        SD = namedtuple('SD', ['trips', 'times', 'ratios', 'S', 'distances'])
        start = self.start
        end = min(start+batch_size, self.ntrips)
        self.start += batch_size

        # remember that we added one to all road ids to distinguish 0
        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        distances = self.distances[start:end]
        
        ratios = torch.ones(trips.shape)
        # code to enable offset
#         offset_pad = pad_arrays_float(self.ratios[start:end])
#         ratios[:, 0] = torch.ones(list(trips.shape)[0]) - offset_pad[:, 0]
#         row_idx = list(range(trips.shape[0]))
#         col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
#         # get the index of the last offset value for each trip
#         col_idx_ratio = list(map(lambda t:len(t)-1, self.ratios[start:end]))
#         ratios[row_idx, col_idx] = offset_pad[row_idx, col_idx_ratio]
        
        return SD(trips=trips, times=times, ratios=ratios, S=self.S, distances=distances)


class DataLoader():
    def __init__(self, trainpath, num_slot=71):
        """
        trainpath (string): The h5file path
        num_slot (int): The number of slots in a day
        """
        self.trainpath = trainpath
        self.num_slot = num_slot
        self.slotdata_pool = []
        ## `weights[i]` is proportional to the number of trips in `slotdata_pool[i]`
        self.weights = None
        ## The length of `slotdata_pool`
        self.length = 0
        ## The current index of the order emit
        self.order_idx = -1

    def read_file(self, fname, enable_subtraj=False):
        """
        Reading one h5file and appending the data into `slotdata_pool`. This function
        should only be called by `read_files()`.
        """
        with h5py.File(fname) as f:
            # modify the number of slots if different,
            # dataset 150103 only has 55 slots
            if len(list(f.keys())) != self.num_slot:
                self.num_slot = len(list(f.keys()))
            for slot in range(1, self.num_slot+1):
                S = np.rot90(f["/{}/S".format(slot)][...]).copy()
                n = f["/{}/ntrips".format(slot)][...]
                if n == 0: continue
                trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]
                times = [f["/{}/time/{}".format(slot, i)][...] for i in range(1, n+1)]
                # pass the ratio array into ratios (for barefoot dataset)
#                 ratios = [f["/{}/ratio/{}".format(slot, i)][...] for i in range(1, n+1)]
                # pass the offset array into ratios (for osmnx dataset only)
                ratios = [f["/{}/offset/{}".format(slot, i)][...] for i in range(1, n+1)]
                distances = [f["/{}/distance/{}".format(slot, i)][...] for i in range(1, n+1)]
                if enable_subtraj:
                    indices = [f["/{}/indices/{}".format(slot, i)][...] for i in range(1, n+1)]
                    alltimes = [f["/{}/times/{}".format(slot, i)][...] for i in range(1, n+1)]
                if enable_subtraj:
                    self.slotdata_pool.append(
                        SlotData(np.array(trips), np.array(times), np.array(ratios), S,
                                 np.array(distances), indices=np.array(indices), alltimes=np.array(alltimes)))
                else:
                    self.slotdata_pool.append(
                        SlotData(np.array(trips), np.array(times), np.array(ratios), S,
                                 np.array(distances)))

    def read_files(self, fname_lst, enable_subtraj=False):
        """
        Reading a list of h5file and appending the data into `slotdata_pool`.
        enable_subtraj: whether to read in the midtimes array for subtrajectory construction
        """
        for fname in fname_lst:
            fname = os.path.basename(fname)
            print("Reading {}...".format(fname))
            self.read_file(os.path.join(self.trainpath, fname), enable_subtraj)
            print("Done.")
        self.weights = np.array(list(map(lambda s:s.ntrips, self.slotdata_pool)))
        self.weights = self.weights / np.sum(self.weights)
        self.length = len(self.weights)
        self.order = np.random.permutation(self.length)
        self.order_idx = 0

    def random_emit(self, batch_size):
        """
        Return a batch of data randomly.
        """
        i = np.random.choice(self.length, p=self.weights)
        return self.slotdata_pool[i].random_emit(batch_size)
    
    def random_emit_sub(self, batch_size):
        """
        Return a batch of data randomly.
        """
        i = np.random.choice(self.length, p=self.weights)
        return self.slotdata_pool[i].random_emit_sub(batch_size)

    def order_emit(self, batch_size):
        """
        Visiting the `slotdata_pool` according to `order` and returning the data in the
        slot `slotdata_pool[i]` orderly.
        """
        i = self.order[self.order_idx]
        data = self.slotdata_pool[i].order_emit(batch_size)
        if data is None: ## move to the next slot
            self.order_idx += 1
            if self.order_idx >= self.length:
                self.order_idx = 0
                self.order = np.random.permutation(self.length)
            i = self.order[self.order_idx]
            data = self.slotdata_pool[i].order_emit(batch_size)
        return data
