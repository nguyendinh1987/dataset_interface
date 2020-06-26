import numpy as np
import numpy.random as rng
import cv2
import glob
import sys, os
from sklearn.utils import shuffle
import pandas as pd 
import matplotlib.pyplot as plt
import copy

class sthsth(object):
    '''
    This object is programe to interface with sth-sth dataset and generate train data for video classification/zeroshot learning
    '''
    def __init__(self,**kargs):
        assert kargs["rootF"] is not None, "No path to dataset"
        
        self.mode = "split" if "mode" not in kargs.keys() else kargs["mode"]
        if self.mode == "split":
            self.filelistFo = kargs["filelistF"]
        else:
            self.filelistFi = kargs["filelistFi"]

        self.rootFo = kargs["rootF"]
        self.verbose = True if "verbose" not in kargs.keys() else kargs["verbose"]
        self.outFo = "/opt/workspace/output" if "prog_output" not in kargs.keys() else kargs["prog_output"]
        self.labelFi = "something-something-v1-labels.csv" if "alllabels_file" not in kargs.keys() else kargs["alllabels_file"]
        ## load data info
        print("Get labels")
        self._getAllLabels()
        print("Get file list")
        self._getFileList()
    
    ## utis
    def _getAllLabels(self):
        self.labels = pd.read_csv(self.labelFi,usecols=[0,1],header=None)
        if self.verbose:
            print("There are {} actions".format(self.labels.shape[0]))
            # print(self.labels.iloc[0][0])
            # print(self.labels[self.labels[0]=="Folding something"].index.tolist()[0])
            # to get label description: use self.labels.iloc[id][0]
            # to get label id: use self.labels[self.labels[0]==<label_description>].index.tolist()[0]
        return True
    
    def _getFileList(self):
        self.files = pd.read_csv(self.filelistFi,header=None)
        if self.verbose:
            print("There are {} files".format(self.files.shape[0]))
        return True

    def _getLabelId(self,vnum):
        findex = self.files[self.files[0]==vnum].index.tolist()[0]
        fdes = self.files.iloc[findex][1]
        lid = self.labels[self.labels[0]==fdes].index.tolist()[0]
        # if self.verbose:
        #     print("vnum: {}; lid: {}; des: {}".format(vnum,lid,fdes))
        return lid
    
    ## interface with data
    def get_nclasses(self):
        return self.labels.shape[0]

    def get_video(self,vnum = None,wh = None, rootF = "20bn-something-something-v1",isshow=False,ts=15,isgrey=False):
        if vnum is None:
            vnum = rng.randint(0,self.files.shape[0])
        path_to_video = os.path.join(self.rootFo,rootF,str(vnum))
        frame_list = glob.glob(path_to_video+"/*.jpg")
        frame_nums = [int(fn.split("/")[-1].split(".")[0]) for fn in frame_list]
        frame_indices = np.argsort(frame_nums)
        frame_list = [frame_list[fid] for fid in frame_indices]

        v = []
        for fl in frame_list:
            frame = cv2.imread(fl)
            if isgrey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if wh is not None:
                frame = cv2.resize(frame,(wh[0],wh[1]))
            v.append(frame)

        if isshow:
            for f in v:
                cv2.imshow("v",f)
                cv2.waitKey(ts)
                cv2.destroyWindow("v")
        
        return v
    
    def get_hist_vlength(self,rootF = "20bn-something-something-v1"):
        vl = [0 for i in range(self.files.shape[0])]
        for f in range(len(vl)):
            if f%50==0:
                print("process [{}/{}]".format(f,self.files.shape[0]))
            path_to_video = os.path.join(self.rootFo,rootF,str(self.files.iloc[f][0]))
            frames = glob.glob(path_to_video+"/*.jpg")
            assert len(frames)>0, "Error reading {}".format(path_to_video)
            vl[f] = len(frames)

        vl = np.array(vl)
        n, bins, patches = plt.hist(vl,50, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('video length')
        plt.ylabel('Prob')
        plt.title('Histogram of video lengths')
        # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(40, 160)
        # plt.ylim(0, 0.03)
        plt.grid(True)
        plt.show()

        return n,bins,patches
    
    def merge_segchg(self,data,do_check=False):
        '''
        data must have five dimension [batch_size,seg,h,w,c]
        Function will convert it into [batch_size,h,w,c*seg]
        '''
        bsz,sg,h,w,c = data.shape
        data = np.transpose(data,(0,2,3,4,1)).reshape((bsz,h,w,sg*c))
        if do_check and c == 1:
            for bid in range(bsz):
                for fid in range(sg):
                    cv2.imshow("frames_{}".format(bid),data[bid,:,:,fid]/255)
                    cv2.waitKey(150)
                cv2.destroyWindow("frames_{}".format(bid))
        return data
    
    def gen_batch(self,isshuffle=True,merge_sgch=False,**kargs):
        # print("xxxx")
        bsz = kargs["bsz"]
        filelist = self.files[0].values
        nfiles = filelist.shape[0]
        # print("nfiles: {}; bsz: {}".format(nfiles,bsz))
        npatches = np.floor(nfiles/bsz).astype(np.int16)+1
        # print(npatches)
        
        while True:
            if isshuffle:
                filelist = shuffle(filelist)
            for p in range(npatches):
                # print("{} of {}".format(p+1,npatches))
                internal_bsz = min((p+1)*bsz,nfiles)-p*bsz
                # print("int_bsz: {}; bsz: {}".format(internal_bsz,bsz))
                internal_kargs = copy.copy(kargs)
                internal_kargs["bsz"] = internal_bsz
                startp = p*bsz
                endp = startp+internal_bsz
                vlist = filelist[startp:endp]
                data,target = self.get_batch(vlist=vlist,**internal_kargs)
                if merge_sgch:
                    data = self.merge_segchg(data)
                del internal_kargs
                # print("end gen_batch")
                yield (data,target)

    def get_batch(self,vlist = None, bsz = 1, nseg=8, wh = None, isgrey=False, check_bsz = False):
        if vlist is None:
            fcids = rng.choice(self.files.shape[0],bsz,replace=False)
            vlist = self.files[0][fcids]
        else:
            assert len(vlist)>=bsz, "number of videos in vlist should be larger than batch size (bsz)"
        # print(vlist)

        if isgrey:
            data = np.zeros((bsz,nseg,wh[1],wh[0],1))
        else:
            data = np.zeros((bsz,nseg,wh[1],wh[0],3))
        
        target = [0 for i in range(bsz)]
        for bid, vid in enumerate(vlist):
            if bid+1 > bsz:
                break
            v = self.get_video(vnum=vid,wh=wh,isgrey=isgrey)#,isshow=check_bsz,ts=50)
            data_patch = np.linspace(0,len(v),nseg,endpoint=False).astype(np.int16)
            for pid in range(1,len(data_patch)):
                fid = rng.choice(range(data_patch[pid-1],data_patch[pid]),1,replace=False)[0]
                if isgrey:
                    data[bid,pid-1] = np.expand_dims(v[fid],axis=-1)
                else:
                    data[bid,pid-1] = v[fid]
            target[bid] = self._getLabelId(vid)
        
        if check_bsz:
            for bid in range(bsz):
                for fid in range(nseg):
                    cv2.imshow("frames_{}".format(bid),data[bid,fid]/255)
                    cv2.waitKey(150)
                cv2.destroyWindow("frames_{}".format(bid))
        # print("end get_batch")
        return data, target
# class lfw_face(object):
#     def __init__(self,rootF,**kargs):
#         assert rootF is not None, "No path to dataset"
#         self.rootFo = rootF
#         self.verbose = True if "verbose" not in kargs.keys() else kargs["verbose"]
#         self.outFo = "/opt" if "prog_output" not in kargs.keys() else kargs["prog_output"]
#         self.useAllClasses = False if "use_all" not in kargs.keys() else kargs["use_all"]
        
#         self.labelFi = "AllLabels.txt" if "alllabels_file" not in kargs.keys() else kargs["alllabels_file"]
#         self.allLabels_file = os.path.join(self.outFo,self.labelFi)
#         self.nAllClasses = None
#         self.allLabels = None
#         self._getAllLabels()
        
#         self.focuslabelFi = "train_0.txt" if "sublabels_file" not in kargs.keys() else kargs["sublabels_file"]
#         self.focusLabels_file = os.path.join(self.outFo,self.focuslabelFi)
#         # self.focus_rate will be used if focuslabelFi does not exist
#         self.focus_rate = 0.7 if "focus_rate" not in kargs.keys() else kargs["focus_rate"]
#         self.nFocusClasses = None
#         self.focusLabels = None
#         if self.useAllClasses:
#             self.focusLabels = self.allLabels
#             self.nFocusClasses = self.nAllClasses
#             if self.verbose:
#                 print("Focus classes has {} identified people.".format(self.nFocusClasses))

#         else:
#             self._getFocusLabels()
        
#         self.faceList = self._getFaceList()
#         if self.verbose:
#             nc = [len(l) for l in self.faceList]
#             n = np.sum(np.array(nc))
#             print("There are {} images for {} identified faces".format(n,self.nFocusClasses))
        
#         self.maskedFaceRootFo = None if "maskedface" not in kargs.keys() else kargs["maskedface"]
#         if self.maskedFaceRootFo is not None:
#             self.maskfaceList = self._getFaceList(group="maskface")
#             for index in range(len(self.maskfaceList)):
#                 assert self._getId(self.maskfaceList[index][0].split("/")[0]) == self._getId(self.faceList[index][0].split("/")[0]), "not synchonize between faceList and maskfaceList at "+str(index)
#             if self.verbose:
#                 nc = [len(l) for l in self.maskfaceList]
#                 n = np.sum(np.array(nc))
#                 print("There are {} images for {} identified faces".format(n,self.nFocusClasses))
#         else:
#             self.maskfaceList = None
#     #####
#     def getFaceList(self,group="face",fullpath=False,withlabelId=False):
#         if fullpath:
#             tmp = self.faceList if group=="face" else self.maskfaceList
#             rootFo = self.rootFo if group=="face" else self.maskedFaceRootFo
#             Flist = []
#             for l in tmp:
#                 for lf in l:
#                     Flist.append(os.path.join(rootFo,lf))
#         else: 
#             Flist = self.faceList if group=="face" else self.maskfaceList
#         if withlabelId:
#             # labelIds = []
#             # for l in Flist:
#             #     print(l)
#             #     labelIds.append(self._getId(l.split("/")[-2]))
#             labelIds = [self._getId(l.split("/")[-2]) for l in Flist]
#             return [Flist,labelIds]
#         else:
#             return Flist
#     def getnClasses(self):
#         return self.nAllClasses
#     def getAllIds(self):
#         idList = np.zeros((1,len(self.focusLabels)))
#         idx = -1
#         for k,v in self.focusLabels.items():
#             idx += 1
#             idList[0,idx] = int(v)
#         return idList
#     def getId(self,labels):
#         return [self._getId(l) for l in labels]
        
#     def getRefList(self,nr=1,group="face",fullpath=False):
#         fList = self.faceList if group=="face" else self.maskfaceList
#         rootFo= self.rootFo if group=="face" else self.maskedFaceRootFo
#         ns = [len(l) for l in fList]
#         min_ns = min(nr,min(ns))
#         refIList = []
#         refLList = []
#         for l in fList:
#             samples = rng.choice(len(l),min_ns,replace=False)
#             for s in samples:
#                 if fullpath:
#                     refIList.append(os.path.join(rootFo,l[s]))
#                 else:
#                     refIList.append(l[s])
#                 refLList.append(self._getId(l[s].split("/")[0]))

#         return [refIList,refLList],min_ns
    
#     #####
#     def _getId(self,label):
#         return self.focusLabels[label]
#     def _getAllLabels(self):
#         if not os.path.isfile(self.allLabels_file): # get from data root folder and write down file
#             print(self.rootFo)
#             Ids = [f.path.split("/")[-1] for f in os.scandir(self.rootFo) if f.is_dir()]
#             print(Ids)
#             self.nAllClasses = len(Ids)
#             self.allLabels = {Ids[i]:i for i in range(self.nAllClasses)}

#             ofw = open(self.allLabels_file,"w")
#             for k in self.allLabels.keys():
#                 ofw.write(k+" "+str(self.allLabels[k])+"\n")
#             ofw.close() 

#         else: # load from file
#             ofw = open(self.allLabels_file,"r")
#             label_list = ofw.readlines()
#             ofw.close()
#             self.nAllClasses = len(label_list)
#             self.allLabels = {label_list[i].split(" ")[0]:int(label_list[i].split(" ")[1])for i in range(self.nAllClasses)}
        
#         # # sort list follow id
#         # tmp = ['' for i in range(self.nAllClasses)]
#         # for k,v in self.allLabels.items():
#         #     tmp[v] = k+"*"+str(v)

#         # self.allLabels = {tmp[i].split("*")[0]:int(tmp[i].split("*")[1]) for i in range(self.nAllClasses)}
#         # for k,v in self.allLabels.items():
#         #     print("{}: {}".format(k,v))
#         if self.verbose:
#             print("LFW has {} identified people.".format(self.nAllClasses))
                
#         return True

#     def _getFocusLabels(self):
#         if os.path.isfile(self.focusLabels_file): # load from file
#             ofw = open(self.focusLabels_file,"r")
#             label_list = ofw.readlines()
#             ofw.close()
#             self.nFocusClasses = len(label_list)
#             self.focusLabels = {label_list[i].split(" ")[0]:int(label_list[i].split(" ")[1])for i in range(self.nFocusClasses)}
#         else: # split from AllLabels and save to file
#             assert self.allLabels is not None, "Please load All labels first"
#             splitP = int(self.nAllClasses*self.focus_rate)

#             focusLabels = shuffle(range(self.nAllClasses))[:splitP]
            
#             self.nFocusClasses = splitP
#             self.focusLabels = {}
#             restlabels = {}
#             for k,v in self.allLabels.items():
#                 if v in focusLabels:
#                     self.focusLabels[k] = v
#                 else:
#                     restlabels[k] = v
            
#             # save focuslabels:
#             ofw = open(self.focusLabels_file,"w")
#             for k in self.focusLabels.keys():
#                 ofw.write(k+" "+str(self.focusLabels[k])+"\n")
#             ofw.close()
            
#             # save the rest labels
#             rest_file = os.path.join(self.outFo,"rest_"+self.focuslabelFi)
#             ofw = open(rest_file,"w")
#             for k in restlabels.keys():
#                 ofw.write(k+" "+str(restlabels[k])+"\n")
#             ofw.close()
        
#         # # sort list follow id
#         # tmp = ['' for i in range(self.nAllClasses)]
#         # for k,v in self.focusLabels.items():
#         #     tmp[v] = k+"*"+str(v)
#         # self.focusLabels = {tmp[i].split("*")[0]:int(tmp[i].split("*")[1]) for i in range(self.nAllClasses) if len(tmp[i])>0}
        
#         if self.verbose:
#             print("Focus classes has {} identified people.".format(self.nFocusClasses))

#         return True
    
#     def _getFaceList(self,group="face"):
#         assert self.focusLabels is not None, "Please update focusLabels first by execute a function _getFocusLabels()"
    
#         tmp = [[] for i in range(self.nAllClasses)]
    
#         for l,v in self.focusLabels.items():
#             if group == "face":
#                 watchF = os.path.join(self.rootFo,l)
#             elif group == "maskface":
#                 watchF = os.path.join(self.maskedFaceRootFo,l)
#             else:
#                 raise ValueError("group param expect 'face' or 'maskface' but got '{}'".format(group))
            
#             for f in os.scandir(watchF):
#                 subnames = f.path.split("/")
#                 tmp[v].append(subnames[-2]+"/"+subnames[-1])
        
#         faceList = [tmp[i] for i in range(self.nAllClasses) if len(tmp[i])>0]
        
#         return faceList

#     def load_samples(self,samples=None,**kargs):
#         if samples is None:
#             print("No samples to load")
#             return None,None
#         imgs = [i for i in samples]
#         labels = [i for i in samples]
#         for sid,s in enumerate(samples):
#             imgs[sid],labels[sid] = self.load_sample(sample=s,**kargs)
#         return imgs,labels

#     def load_sample(self,sample=None,face_type="face",wh=None,isshow=False):
#         '''
#         sample: is in format "label/img.jpg"
#         '''

#         assert sample is not None, "No sample is provided"
#         image_root = self.rootFo if face_type=="face" else self.maskedFaceRootFo
#         I = cv2.imread(os.path.join(image_root,sample))
#         if wh is not None:
#             I = cv2.resize(I,(wh[1],wh[0]))
#         label = sample.split("/")[0]
#         if isshow:
#             cv2.imshow(label,I)
#             cv2.waitKey()
#             cv2.destroyWindow(label)
#         return I, label
    
#     ######################################################
#     def full_mask_face_pair_gen(self,**kargs):
#         while True:
#             pairs,targets = self.load_fullface_maskface_pair(**kargs)
#             yield (pairs,targets)

#     def load_fullface_maskface_pair(self,wh=None,bz=20,check_pair=False):
#         matchn = int(bz/2)
#         targets = np.zeros((bz,1))
#         targets[0:matchn,0] = 1 # match
#         pairs = [[] for i in range(2)]
#         fullface_label_col = []
#         maskface_label_col = []
#         for bzid in range(bz):
#             fid = rng.randint(0,len(self.faceList))
#             # make sure maskface has image
#             while len(self.maskfaceList[fid])==0:
#                 fid = rng.randint(0,len(self.faceList))
#             fullface_label_col.append(self._getId(self.faceList[fid][0].split("/")[0]))
#             imgid = rng.choice(len(self.faceList[fid]),1,replace=False)[0]
            
#             fullfaceI,_ = self.load_sample(self.faceList[fid][imgid],face_type="face",wh=wh)
#             pairs[0].append(fullfaceI)
#             if bzid >= matchn:
#                 ct = True
#                 while ct:
#                     nfid = rng.randint(0,len(self.faceList))
#                     if nfid != fid and len(self.maskfaceList[nfid])>0:
#                         ct = False
#                 fid = nfid
#             maskface_label_col.append(self._getId(self.faceList[fid][0].split("/")[0]))
#             imgid = rng.choice(len(self.maskfaceList[fid]),1,replace=False)[0]
#             maskfaceI,_ = self.load_sample(self.maskfaceList[fid][imgid],face_type="maskface",wh=wh)
#             pairs[1].append(maskfaceI)
        
#         pairs[0],pairs[1],targets,maskface_label_col,fullface_label_col = shuffle(pairs[0],
#                                                                                   pairs[1],
#                                                                                   targets,maskface_label_col,
#                                                                                   fullface_label_col)
#         pairs[0] = np.array(pairs[0])
#         pairs[1] = np.array(pairs[1])
#         if check_pair:
#             for k in range(bz):
#                 cv2.imshow("{}_{}_0".format(fullface_label_col[k], targets[k,0]),pairs[0][k])
#                 cv2.imshow("{}_{}_1".format(maskface_label_col[k], targets[k,0]),pairs[1][k])
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
#         return pairs,targets
        