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

class saaagg(object):
    def __init__(self,**kargs):
        self.dataRoot = kargs["dataRoot"]
        assert os.path.isdir(self.dataRoot), "{} does not exist".format(self.dataRoot)
        
        # deside data list
        '''
        mode = "train": get train list
        mode = "val"  : get val list
        mode = "test" : get test list
        mode = "all"  : get all files. It is used to generate data for domain adaptation
        mode = "split": get all files and doing data split to generate train/val/test lists
        '''
        self.mode = "train" if "mode" not in kargs.keys() else kargs["mode"]
        if self.mode in ["train","val","test"]:
            # self.splitId = 0 if "splitId" not in kargs.keys() else kargs["splitId"]
            # self.split_file = self.dataRoot+"/datasplit/{}list0{}.txt".format(self.mode,self.splitId)
            self.splitFile = kargs["splitFile"]
            assert os.path.isfile(self.splitFile), "{} does not exist".format(self.splitFile)
        else:
            # self.splitId = None
            self.splitFile = None
        
        # data global information
        self.dataSource= self.dataRoot+"/Videos" if "videoSource" not in kargs.keys() else kargs["videoSource"]
        self.dataOutpath = kargs["output"]
        self.classIdFile = self.dataRoot+"/Videos/ClassId.txt" if "classIdFile" not in kargs.keys() else kargs["classIdFile"]
        self.subVideoFolder = "" if "subVideoFolder" not in kargs.keys() else kargs["subVideoFolder"]
        self.vList = None
        self.vTarget = None
        self.nv = None
        self.classIds = None
        assert os.path.isdir(self.dataSource),"{} does not exist".format(self.dataSource)
        assert os.path.isfile(self.classIdFile),"{} does not exist".format(self.classIdFile)
        if not os.path.isdir(self.dataOutpath):
            opt = input("Data framework will generate folder {} automatically? Type 'y' to proceed. ".format(self.dataOutpath))
            if opt == "y":
                os.makedirs(self.dataOutpath)
            else:
                sys.exit()

        self.getClassIds()
        if self.mode in ["train","val","test"]:
            self.getVListFromFile()
        else:
            self.getVList()
        
    ##############################################
    ##############################################
    # def _get_classId(self,fid):
    #     print(self.vList[fid])
    #     return self.classIds[self.vList[fid].split["/"][0]]
    ##############################################
    ##############################################
    def genbatch_for_classification(self,filelist=None,wh=None,bts=10,isshuffle=True,lbconf={}):
        if filelist is None:
            filelist = self.vList
        nfiles = len(filelist)
        npatches = np.floor(nfiles/bts).astype(np.int16)
        if npatches*bts<nfiles:
            npatches += 1
        print(nfiles)
        print(npatches)
        while True:
            if isshuffle:
                filelist = shuffle(filelist)
            for p in range(npatches):
                internal_bts = min((p+1)*bts,nfiles)-p*bts
                startp = p*bts
                endp = startp+internal_bts
                vlist = filelist[startp:endp]
                internal_kargs = copy.copy(lbconf)
                internal_kargs["bts"]=internal_bts
                internal_kargs["vids"]=range(internal_bts)
                internal_kargs["wh"]=wh
                internal_kargs["vlist"]=vlist
                internal_kargs["isshuffle"]=False
                data,target = self.loadbatch_for_classification(**internal_kargs)
                bsz,sg,h,w,c = data.shape
                data = np.transpose(data,(0,2,3,4,1)).reshape((bsz,h,w,sg*c))
                del internal_kargs
                yield (data,target)
        
    def loadbatch_for_classification(self,vids=None,wh=None,
                                          bts=10,
                                          nseg=None,
                                          vlist=None,
                                          isgrey = False,
                                          isshuffle=True,
                                          return_array=False,
                                          isshow=False):
        '''
        vlist should be a path of self.vList
        '''
        if return_array:
            assert wh is not None, "wh cannot be None if you want to return a type of array"
            assert nseg is not None,"nseg cannot be None if you want to return a type of array"
        if vlist is None:
            vlist = self.vList
        if vids is None:
            vids = rng.choice(len(vlist),bts,replace=False)
        else:
            bts = len(vids)
        assert len(vlist)>=bts,"Number of videos is less than batchsize"

        videos = []
        targets = []
        NonAggId = -1
        for k in self.classIds.keys():
            if "Non" in k:
                NonAggId = self.classIds[k]
                break
        assert NonAggId > 0, "Your class Id setup dont have 'Non'"

        for vid in vids:
            vpath = self.getVpath(vid,vlist)
            if "NonAgg" in vpath:
                targets.append(NonAggId)
            else:
                targets.append(1-NonAggId)
            org_v = self.loadVideo(vpath,wh=wh,isgrey=isgrey)
            data_patch = np.linspace(0,len(org_v),nseg+1,endpoint=True).astype(np.int16)
            seg_v = []
            for pid in range(1,len(data_patch)):
                fid = rng.choice(range(data_patch[pid-1],data_patch[pid]),1,replace=False)[0]
                seg_v.append(org_v[fid])
            videos.append(seg_v)
        
        if isshuffle:
            videos,targets = shuffle(videos,targets)
        
        if isshow:
            for vid, v in enumerate(videos):
                for fr in v:
                    cv2.imshow("{}".format(targets[vid]),fr)
                    cv2.waitKey(25)
                cv2.destroyAllWindows()

        if return_array:
            for vid in range(len(videos)):
                videos[vid] = np.array(videos[vid])
            videos = np.array(videos)
            targets = np.array(targets)
        
        return videos,targets
    def traintestsplit(self,vlist=None,savetofiles=None,testratio=0.3,random_state=42):
        if vlist is None:
            vlist = self.vList
        y = range(len(vlist))
        vListTrain, vListTest, y_train, y_test = train_test_split(vlist, y, test_size=testratio, random_state=random_state)
        if savetofiles is not None:
            listtowrite = [vListTrain,vListTest]
            for lid in range(2):
                f = open(savetofiles[lid], "w")
                for l in listtowrite[lid]:
                    f.write(l+"\n")
                f.close()
        return vListTrain,vListTest
    def getVpath(self,vid,vlist=None,fullpath=True):
        if vlist is None:
            vlist = self.vList
        vpath = os.path.join(self.dataSource,self.subVideoFolder,vlist[vid])
        return vpath
    
    def getVfolder(self):
        return os.path.join(self.dataSource,self.subVideoFolder)
    def getnclasses(self):
        return len(self.classIds.keys())

    def loadVideo(self,vpath,wh=None,isgrey=False,isshow=False):
        # vpath should be a full path
        cap = cv2.VideoCapture(vpath)
        if cap.isOpened()==False:
            print("Error to open file {}".format(vpath))
            return []
        v = []
        while cap.isOpened():
            r,frame = cap.read()
            if r:
                if wh is not None:
                    frame = cv2.resize(frame,tuple(wh))
                if isgrey:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame,axis=-1)
                v.append(frame)
            else:
                break
        cap.release()
        if isshow:
            for f in v:
                cv2.imshow("{}".format(vpath.split("/")[-1]),f)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        return v

    def getClassIds(self):
        with open(self.classIdFile, 'r') as f:
            classes = f.readlines()
            classes = map(lambda cls: cls.replace('\n','').split(' '), classes)
            classes = dict(map(lambda cls: (cls[1], int(cls[0])), classes))
        self.classIds = classes
        return classes

    def getVListFromFile(self):
        print("Get video list from {}".format(self.splitFile))
        f = open(self.splitFile,"r")
        self.vList = f.readlines()
        f.close()
        for fid in range(len(self.vList)):
            self.vList[fid] = self.vList[fid][0:-1] # removing \n
        print("There are {} videos".format(len(self.vList)))
        return self.vList

    def getVList(self):
        print("Get video list")
        if self.vList is not None:
            return self.vList
        self.vList = []
        for f in os.scandir(os.path.join(self.dataSource,self.subVideoFolder)):
            if ".txt" in f.path:
                continue
            for l in os.scandir(f.path):
                subnames = l.path.split("/")
                self.vList.append(subnames[-2]+"/"+subnames[-1])
        return self.vList

    def getVListWithSpec(self,spec):
        vlist = []
        for sp in spec:
            vlist.extend([self.vList[vid] for vid in range(len(self.vList)) if sp in self.vList[vid]])
        return vlist 
        
