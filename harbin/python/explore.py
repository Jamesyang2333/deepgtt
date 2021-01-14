import h5py
f = h5py.File("/Project0551/jingyi/deepgtt/data/trainpath/150105.h5", 'r')
print(list(f.keys()))
dset = f['1']
print(dset.shape)