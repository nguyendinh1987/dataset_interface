import random
# from moviepy.editor import *
import numpy as np
import numpy.random as rng
import cv2
import pickle
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import copy
import sys
from tensorflow.python.keras.utils.data_utils import Sequence

class sthsthtfgen(Sequence):
    def __init__(self,dataobj,bts=10,isshuffle=False,merge_sgch=False,n=1,**kargs):
        self.dataobj = dataobj
        self.filelist = dataobj.files[0].values
        self.nfiles = len(self.filelist)
        self.bts = bts
        self.isshuffle = isshuffle
        self.merge_sgch = merge_sgch
        self.nbatches = int(np.ceil(self.nfiles/float(self.bts)))
        self.nbatches_train = int(np.floor(self.nbatches/n))
        self.kargs = kargs
        print("batch_size: {}".format(self.bts))
        print("number of batches: {}".format(self.nbatches))
    
    def __len__(self):
        return self.nbatches_train #np.ceil(len(self.filelist)/float(self.bts)).astype(np.int16)
    
    def __getitem__(self,idx):
        # print("{} of {}".format(p+1,npatches))
        internal_bsz = min((idx+1)*self.bts,self.nfiles)-idx*self.bts
        # print("int_bsz: {}; bsz: {}".format(internal_bsz,bsz))
        startp = idx*self.bts
        endp = startp+internal_bsz
        vlist = self.filelist[startp:endp]
        internal_kargs = copy.copy(self.kargs)
        internal_kargs["nv"] = len(vlist)
        data,target = self.dataobj.get_batch(vlist=vlist,**internal_kargs)
        if self.merge_sgch:
            data = self.dataobj.merge_segchg(data)
        # shuffle list when last sample is loaded
        if idx+1 == self.nbatches and self.isshuffle:
            print("Doing shuffle data [{}]".format(idx+1))
            self.filelist = shuffle(self.filelist)
        return (data,target)
