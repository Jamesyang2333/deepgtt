
import numpy as np
import torch, h5py, os
from collections import namedtuple
import json
import db_osmnx_utils as db_utils
import networkx as nx
import math
from scipy import signal

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
    plus one so the index match the osmnx fid
    """
    return np.concatenate((a+1, [PAD]*(max_length - len(a))))

# no longer needed for fmm data because the path always has distinct entries.
def rle(l):
    return [k for k, g in groupby(l)]

def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

def feature_completion(link_array, G):
    result = np.copy(link_array)
    count_fixed = 0
    count_missing = 0
    fix_indices = []
    for i in range(14963):
        if not np.any(link_array[i]):
            count_missing += 1
            real_neighbor = 0
            val = np.zeros((1, 3))
            print(G.neighbors(i))
            for n in G.neighbors(i):
                if np.any(link_array[n]):
                    val = val + link_array[n]
                    real_neighbor += 1
            if real_neighbor != 0:
                count_fixed += 1
                result[i] = val / real_neighbor
                fix_indices.append(i)
#     print("Number of nodes without feature: "  + str(count_missing))  
#     print("fill the value for "  + str(count_fixed) + " nodes after first iteration")     
    return result, fix_indices

def feature_completion_spatial(fmap):
#     print("number of non-empty features before completion: " + str(np.sum(np.any(link_array, axis=1))))
    result = np.copy(fmap)
    conv = np.ones((3,3))
    conv_map = signal.convolve2d(fmap, conv, boundary='symm', mode='same')
    conv_bitmap = signal.convolve2d(fmap!=0, conv, boundary='symm', mode='same')
    result_smooth = conv_map / conv_bitmap
    result[np.logical_and((result==0), (conv_bitmap!=0))] = result_smooth[np.logical_and((result==0), (conv_bitmap!=0))]
#     print("number of empty features after completion: " + str(np.sum(np.any(result, axis=1))))
    return result
#xs = np.array([np.r_[1], np.r_[1, 2, 3], np.r_[2, 1]])

def feature_completion_fast(link_array, adj):
#     print("number of non-empty features before completion: " + str(np.sum(np.any(link_array, axis=1))))
    result = np.copy(link_array)
    adj_full = adj.toarray()
    adj_full[np.nonzero(adj_full)] = 1
    sum_result = np.matmul(adj_full, link_array)
    result_smooth = np.divide(sum_result, np.tile(np.expand_dims(np.matmul(adj_full, np.any(link_array, axis=1)), axis=0).transpose(), (1, 3)))
    indices =  np.logical_not(np.any(link_array, axis=1))
    result[indices] = result_smooth[indices]
    result[np.any(np.isnan(result), axis=1)] = np.zeros((3))
#     print("number of empty features after completion: " + str(np.sum(np.any(result, axis=1))))
    return result
#xs = np.array([np.r_[1], np.r_[1, 2, 3], np.r_[2, 1]])

def gps2webmercator(lon, lat):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def gps2webmercator_lon(lon):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    return semimajoraxis * east

def gps2webmercator_lat(lat):
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return 3189068.5 * math.log((1 + t) / (1 - t))

# harbin stat
# lon_min = 126.506130
# lat_min = 45.657920
# lon_max = 126.771862
# lat_max = 45.830905
# chengdu stat
lon_min = 104.04214
lat_min = 30.65294
lon_max = 104.12958
lat_max = 30.72775

minx, miny = gps2webmercator(lon_min, lat_min)
maxx, maxy = gps2webmercator(lon_max, lat_max)
lon_range = maxx - minx
lat_range = maxy - miny
vfunc_lon = np.vectorize(gps2webmercator_lon)
vfunc_lat = np.vectorize(gps2webmercator_lat)
print(lon_range / 200)
print(lat_range / 200)

class SlotData():
    def __init__(self, trips, times, ratios, S, distances, links=None, lon=None, lat=None, maxlen=200):
        """
        trips (n, *): each element is a sequence of road segments
        times (n, ): each element is a travel cost
        ratios (n, 2): end road segments ratio
        S (138, 148) or (num_channel, 138, 148): traffic state matrix
        """
        ## filter out the trips that are too long
        idx = [i for (i, trip) in enumerate(trips) if len(trip) <= maxlen]
        self.trips = trips[idx]
        self.times = times[idx]
        if not lon.shape==():
            self.lon = lon[idx]
            self.lat = lat[idx]
        else:
            self.lon = None
            self.lat = None
        self.ratios = ratios[idx]
        self.distances = distances[idx]
        self.S = torch.tensor(S, dtype=torch.float32)
        if not (links is None):
            self.links = torch.tensor(links, dtype=torch.float32)
        else:
            self.links = None
        ## (1, num_channel, height, width)
        if self.S.dim() == 2:
            self.S.unsqueeze_(0).unsqueeze_(0)
        elif self.S.dim() == 3:
            self.S.unsqueeze_(0)
        ## re-arrange the trips by the length in reverse order
        idx = argsort(self.trips)
        self.trips = self.trips[idx]
        self.times = torch.tensor(self.times[idx], dtype=torch.float32)
        if not lon.shape==():
            self.lon = self.lon[idx]
            self.lat = self.lat[idx]
        self.ratios = torch.tensor(self.ratios[idx], dtype=torch.float32)
        self.distances = torch.tensor(self.distances[idx], dtype=torch.float32)

        self.ntrips = len(self.trips)
        self.start = 0
        self.order=None
        self.orderidx=0


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
        SD = namedtuple('SD', ['trips', 'times', 'lon_idx', 'lat_idx', 'ratios', 'S', 'distances', 'links'])
        start = np.random.choice(max(1, self.ntrips-batch_size+1))
        end = min(start+batch_size, self.ntrips)

#         # draw minibatch with replacement during training
#         if self.orderidx >= math.ceil(self.ntrips/batch_size):
#             self.orderidx = 0
#             return None
#         SD = namedtuple('SD', ['trips', 'times', 'lon_idx', 'lat_idx', 'ratios', 'S', 'distances', 'links'])
#         start = np.random.choice(max(1, self.ntrips-batch_size+1))
#         end = min(start+batch_size, self.ntrips)
#         self.orderidx += 1
        
#         # draw minibatch without replacement during training
#         if self.order is None:
#             self.order = np.random.permutation(math.ceil(self.ntrips/batch_size))
#         if self.orderidx >= math.ceil(self.ntrips/batch_size):
#             self.orderidx = 0
#             self.order = np.random.permutation(math.ceil(self.ntrips/batch_size))
#             return None
#         SD = namedtuple('SD', ['trips', 'times', 'lon_idx', 'lat_idx', 'ratios', 'S', 'distances', 'links'])
#         start = self.order[self.orderidx] * batch_size
#         end = min(start+batch_size, self.ntrips)
#         self.orderidx += 1

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        # calculate the feature map index for each road link
        lon_idx = []
        lat_idx = []
#         lon_idx = [torch.tensor(np.floor((vfunc_lon(lon_rec + lon_min) - minx) / (lon_range / 18)), dtype=torch.long) for lon_rec in self.lon[start:end]]
#         for lon_rec in lon_idx:
#             lon_rec[lon_rec==18] = 17
#         lat_idx = [torch.tensor(np.floor((vfunc_lat(lat_rec + lat_min) - miny) / (lat_range / 17)), dtype=torch.long) for lat_rec in self.lat[start:end]]
#         for lat_rec in lat_idx:
#             lat_rec[lat_rec==17] = 16
#         lon_idx = [torch.tensor(np.floor((vfunc_lon(lon_rec + lon_min) - minx) / (lon_range / 6)), dtype=torch.long) for lon_rec in self.lon[start:end]]
#         for lon_rec in lon_idx:
#             lon_rec[lon_rec==6] = 5
#         lat_idx = [torch.tensor(np.floor((vfunc_lat(lat_rec + lat_min) - miny) / (lat_range / 6)), dtype=torch.long) for lat_rec in self.lat[start:end]]
#         for lat_rec in lat_idx:
#             lat_rec[lat_rec==6] = 5
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times, lon_idx=lon_idx, lat_idx=lat_idx, ratios=ratios, S=self.S, distances=distances, links=self.links)

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
        SD = namedtuple('SD', ['trips', 'times', 'lon_idx', 'lat_idx', 'ratios', 'S', 'distances', 'links'])
        start = self.start
        end = min(start+batch_size, self.ntrips)
        self.start += batch_size

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        # calculate the feature map index for each road link
        lon_idx = []
        lat_idx = []
#         lon_idx = [torch.tensor(np.floor((vfunc_lon(lon_rec + lon_min) - minx) / (lon_range / 18)), dtype=torch.long) for lon_rec in self.lon[start:end]]
#         for lon_rec in lon_idx:
#             lon_rec[lon_rec==18] = 17
#         lat_idx = [torch.tensor(np.floor((vfunc_lat(lat_rec + lat_min) - miny) / (lat_range / 17)), dtype=torch.long) for lat_rec in self.lat[start:end]]
#         for lat_rec in lat_idx:
#             lat_rec[lat_rec==17] = 16
#         lon_idx = [torch.tensor(np.floor((vfunc_lon(lon_rec + lon_min) - minx) / (lon_range / 6)), dtype=torch.long) for lon_rec in self.lon[start:end]]
#         for lon_rec in lon_idx:
#             lon_rec[lon_rec==6] = 5
#         lat_idx = [torch.tensor(np.floor((vfunc_lat(lat_rec + lat_min) - miny) / (lat_range / 6)), dtype=torch.long) for lat_rec in self.lat[start:end]]
#         for lat_rec in lat_idx:
#             lat_rec[lat_rec==6] = 5
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times, lon_idx=lon_idx, lat_idx=lat_idx, ratios=ratios, S=self.S, distances=distances, links=self.links)


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
        self.adj = db_utils.get_adj()
        print('finished constructing adjacency matrix')

    def read_file(self, fname, use_gnn=False):
        """
        Reading one h5file and appending the data into `slotdata_pool`. This function
        should only be called by `read_files()`.
        """
        with h5py.File(fname) as f:
            # modify the number of slots if different,
            # dataset 150103 only has 55 slots
            if len(list(f.keys())) != self.num_slot:
                self.num_slot = len(list(f.keys()))
            adj = db_utils.get_scipy_adj()
            G = nx.from_scipy_sparse_matrix(adj)
            for slot in range(1, self.num_slot+1):
                if use_gnn:
                    links = f["/{}/Links".format(slot)][...]
                    links = json.loads(links.tobytes())
                    # 14963 is the total number of road segments in the Harbin dataset
                    link_array = np.zeros((14963, 3))                         
                    for fid, value in links.items():
                        # use fid because gat doesn't consider the zero index in the graph
                        link_array[int(fid)][0] = value['I']
                        link_array[int(fid)][1] = value['O']
                        link_array[int(fid)][2] = value['S']
                    for i in range(3):
                        link_array = feature_completion_fast(link_array, adj)
#                 print(np.sum(result1-result2))
#                 print(result1[indices][:10])
#                 print(result2[indices][:10])
#                 indices =  np.logical_not(np.any(link_array, axis=1))
                # we have to use transpose here otherwise the order is wrong
#                 S = np.rot90(f["/{}/S".format(slot)][...]).copy()
                S = np.transpose(f["/{}/S".format(slot)][...]).copy()
# #                 print('percentage of zero entry in current: ', np.count_nonzero((S==0)) / (138*148))
#                 for i in range(5):
#                     S = feature_completion_spatial(S)
#                 print('percentage of zero entry after 1 iter: ', np.count_nonzero((S==0)) / (138*148))
                n = f["/{}/ntrips".format(slot)][...]
                if n == 0: continue
                trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]
                times = [f["/{}/time/{}".format(slot, i)][...] for i in range(1, n+1)]
                lon = []
                lat = []
                lon = [f["/{}/lon/{}".format(slot, i)][...] for i in range(1, n+1)]
                lat = [f["/{}/lat/{}".format(slot, i)][...] for i in range(1, n+1)]    
                ratios = [f["/{}/ratio/{}".format(slot, i)][...] for i in range(1, n+1)]
                distances = [f["/{}/distance/{}".format(slot, i)][...] for i in range(1, n+1)]
                if use_gnn: 
                    self.slotdata_pool.append(
                    SlotData(np.array(trips), np.array(times), np.array(ratios), S,
                             np.array(distances), links=np.array(link_array), lon=np.array(lon), lat=np.array(lat)))
                else:
                    self.slotdata_pool.append(
                    SlotData(np.array(trips), np.array(times), np.array(ratios), S,
                             np.array(distances), lon=np.array(lon), lat=np.array(lat)))

    def read_files(self, fname_lst, use_gnn=False):
        """
        Reading a list of h5file and appending the data into `slotdata_pool`.
        """
        for fname in fname_lst:
            fname = os.path.basename(fname)
            print("Reading {}...".format(fname))
            self.read_file(os.path.join(self.trainpath, fname), use_gnn)
            print("Done.")
        self.weights = np.array(list(map(lambda s:s.ntrips, self.slotdata_pool)))
        self.weights = self.weights / np.sum(self.weights)
        self.length = len(self.weights)
        self.order = np.random.permutation(self.length)
        self.order_idx = 0
        self.idx_list = list(range(self.length))

    def random_emit(self, batch_size):
        """
        Return a batch of data randomly.
        """
        i = np.random.choice(self.length, p=self.weights)
        return self.slotdata_pool[i].random_emit(batch_size)
        
#         idx = np.random.randint(len(self.idx_list))
#         i = self.idx_list[idx]
#         data = self.slotdata_pool[i].random_emit(batch_size)
# #         print(i)
#         while data is None: ## move to the next slot
#             self.idx_list.remove(i)
#             if not self.idx_list:
#                 print('depleted all training data')
#                 self.idx_list = list(range(self.length))
#             idx = np.random.randint(len(self.idx_list))
#             i = self.idx_list[idx]
#             data = self.slotdata_pool[i].random_emit(batch_size)
# #             print(i)
#         return data

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

    def get_adj(self):
        return self.adj