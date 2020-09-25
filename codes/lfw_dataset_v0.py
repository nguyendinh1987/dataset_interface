import numpy as np
import numpy.random as rng
import cv2
import glob
import sys, os
from sklearn.utils import shuffle
import copy
import face_recognition # install by "pip install face-recognition"

class lfw_face(object):
    def __init__(self,**kargs):
        self.rootFo = kargs["fullfaceroot"]
        self.verbose = True if "verbose" not in kargs.keys() else kargs["verbose"]
        self.outFo = "/opt" if "prog_output" not in kargs.keys() else kargs["prog_output"]
        self.useAllClasses = False if "use_all" not in kargs.keys() else kargs["use_all"]
        
        self.labelFi = "AllLabels.txt" if "alllabels_file" not in kargs.keys() else kargs["alllabels_file"]
        self.allLabels_file = os.path.join(self.outFo,self.labelFi)
        self.nAllClasses = None
        self.allLabels = None
        self._getAllLabels()
        
        self.focuslabelFi = "train_0.txt" if "sublabels_file" not in kargs.keys() else kargs["sublabels_file"]
        self.focusLabels_file = os.path.join(self.outFo,self.focuslabelFi)
        # self.focus_rate will be used if focuslabelFi does not exist
        self.focus_rate = 0.7 if "focus_rate" not in kargs.keys() else kargs["focus_rate"]
        self.nFocusClasses = None
        self.focusLabels = None
        if self.useAllClasses:
            self.focusLabels = self.allLabels
            self.nFocusClasses = self.nAllClasses
            if self.verbose:
                print("Focus classes has {} identified people.".format(self.nFocusClasses))
                print("###################################################################################")
                print(" ")

        else:
            self._getFocusLabels()
        
        self.faceList = self._getFaceList()
        if self.verbose:
            nc = [len(l) for l in self.faceList]
            n = np.sum(np.array(nc))
            print("There are {} full face images for {} identified faces".format(n,self.nFocusClasses))
            print("###################################################################################")
            print(" ")
        
        self.maskedFaceRootFo = None if "maskedfaceroot" not in kargs.keys() else kargs["maskedfaceroot"]
        if self.maskedFaceRootFo is not None:
            self.maskfaceList = self._getFaceList(group="maskface")
            for index in range(len(self.maskfaceList)):
                assert self._getId(self.maskfaceList[index][0].split("/")[0]) == self._getId(self.faceList[index][0].split("/")[0]), "not synchonize between faceList and maskfaceList at "+str(index)
            if self.verbose:
                nc = [len(l) for l in self.maskfaceList]
                n = np.sum(np.array(nc))
                print("There are {} masked face images for {} identified faces".format(n,self.nFocusClasses))
                print("###################################################################################")
                print(" ")
        else:
            self.maskfaceList = None
    #####
    def _getId(self,label):
        return self.focusLabels[label]
    def _getAllLabels(self):
        if not os.path.isfile(self.allLabels_file): # get from data root folder and write down file
            print(self.rootFo)
            Ids = [f.path.split("/")[-1] for f in os.scandir(self.rootFo) if f.is_dir()]
            print(Ids)
            self.nAllClasses = len(Ids)
            self.allLabels = {Ids[i]:i for i in range(self.nAllClasses)}

            ofw = open(self.allLabels_file,"w")
            for k in self.allLabels.keys():
                ofw.write(k+" "+str(self.allLabels[k])+"\n")
            ofw.close() 

        else: # load from file
            ofw = open(self.allLabels_file,"r")
            label_list = ofw.readlines()
            ofw.close()
            self.nAllClasses = len(label_list)
            self.allLabels = {label_list[i].split(" ")[0]:int(label_list[i].split(" ")[1])for i in range(self.nAllClasses)}
        
        # # sort list follow id
        # tmp = ['' for i in range(self.nAllClasses)]
        # for k,v in self.allLabels.items():
        #     tmp[v] = k+"*"+str(v)

        # self.allLabels = {tmp[i].split("*")[0]:int(tmp[i].split("*")[1]) for i in range(self.nAllClasses)}
        # for k,v in self.allLabels.items():
        #     print("{}: {}".format(k,v))
        if self.verbose:
            print("LFW has {} identified people.".format(self.nAllClasses))
            print("###################################################################################")
            print(" ")
                
        return True

    def _getFocusLabels(self):
        if os.path.isfile(self.focusLabels_file): # load from file
            ofw = open(self.focusLabels_file,"r")
            label_list = ofw.readlines()
            ofw.close()
            self.nFocusClasses = len(label_list)
            self.focusLabels = {label_list[i].split(" ")[0]:int(label_list[i].split(" ")[1])for i in range(self.nFocusClasses)}
        else: # split from AllLabels and save to file
            assert self.allLabels is not None, "Please load All labels first"
            splitP = int(self.nAllClasses*self.focus_rate)

            focusLabels = shuffle(range(self.nAllClasses))[:splitP]
            
            self.nFocusClasses = splitP
            self.focusLabels = {}
            restlabels = {}
            for k,v in self.allLabels.items():
                if v in focusLabels:
                    self.focusLabels[k] = v
                else:
                    restlabels[k] = v
            
            # save focuslabels:
            ofw = open(self.focusLabels_file,"w")
            for k in self.focusLabels.keys():
                ofw.write(k+" "+str(self.focusLabels[k])+"\n")
            ofw.close()
            
            # save the rest labels
            rest_file = os.path.join(self.outFo,"rest_"+self.focuslabelFi)
            ofw = open(rest_file,"w")
            for k in restlabels.keys():
                ofw.write(k+" "+str(restlabels[k])+"\n")
            ofw.close()
        
        # # sort list follow id
        # tmp = ['' for i in range(self.nAllClasses)]
        # for k,v in self.focusLabels.items():
        #     tmp[v] = k+"*"+str(v)
        # self.focusLabels = {tmp[i].split("*")[0]:int(tmp[i].split("*")[1]) for i in range(self.nAllClasses) if len(tmp[i])>0}
        
        if self.verbose:
            print("Focus classes has {} identified people.".format(self.nFocusClasses))
            print("###################################################################################")
            print(" ")
        return True
    
    def _getFaceList(self,group="face"):
        assert self.focusLabels is not None, "Please update focusLabels first by execute a function _getFocusLabels()"
    
        tmp = [[] for i in range(self.nAllClasses)]
    
        for l,v in self.focusLabels.items():
            if group == "face":
                watchF = os.path.join(self.rootFo,l)
            elif group == "maskface":
                watchF = os.path.join(self.maskedFaceRootFo,l)
            else:
                raise ValueError("group param expect 'face' or 'maskface' but got '{}'".format(group))
            
            for f in os.scandir(watchF):
                subnames = f.path.split("/")
                tmp[v].append(subnames[-2]+"/"+subnames[-1])
        
        faceList = [tmp[i] for i in range(self.nAllClasses) if len(tmp[i])>0]
        
        return faceList
    #####
    def getFaceList(self,group="face",fullpath=False,withlabelId=False):
        if fullpath:
            tmp = self.faceList if group=="face" else self.maskfaceList
            rootFo = self.rootFo if group=="face" else self.maskedFaceRootFo
            Flist = []
            for faceid in range(len(tmp)):
                Flist.append([os.path.join(rootFo,lf) for lf in tmp[faceid]])
        else: 
            Flist = self.faceList if group=="face" else self.maskfaceList
        if withlabelId:
            # labelIds = []
            # for l in Flist:
            #     print(l)
            #     labelIds.append(self._getId(l.split("/")[-2]))
            labelIds = [self._getId(l[0].split("/")[-2]) for l in Flist]
            return [Flist,labelIds]
        else:
            return [Flist]
    def getnClasses(self):
        return self.nAllClasses
    def getAllIds(self):
        idList = np.zeros((1,len(self.focusLabels)))
        idx = -1
        for k,v in self.focusLabels.items():
            idx += 1
            idList[0,idx] = int(v)
        return idList
    def getId(self,labels):
        return [self._getId(l) for l in labels]   
    def getRefList(self,nr=1,group="face",fullpath=False):
        fList = self.faceList if group=="face" else self.maskfaceList
        rootFo= self.rootFo if group=="face" else self.maskedFaceRootFo
        ns = [len(l) for l in fList]
        min_ns = min(nr,min(ns))
        refIList = []
        refLList = []
        for l in fList:
            samples = rng.choice(len(l),min_ns,replace=False)
            for s in samples:
                if fullpath:
                    refIList.append(os.path.join(rootFo,l[s]))
                else:
                    refIList.append(l[s])
                refLList.append(self._getId(l[s].split("/")[0]))

        return [refIList,refLList],min_ns 
    def load_samples(self,samples=None,**kargs):
        if samples is None:
            print("No samples to load")
            return None,None
        imgs = [i for i in samples]
        labels = [i for i in samples]
        for sid,s in enumerate(samples):
            imgs[sid],labels[sid] = self.load_sample(sample=s,**kargs)
        return imgs,labels
    def load_sample(self,sample=None,subpath=None,face_type="face",wh=None,isshow=False):
        '''
        sample: is in format "label/img.jpg"
        '''

        assert sample is not None, "No sample is provided"
        image_root = self.rootFo if face_type=="face" else self.maskedFaceRootFo
        I = cv2.imread(os.path.join(image_root,sample))
        if wh is not None:
            I = cv2.resize(I,(wh[1],wh[0]))
        label = sample.split("/")[0]
        if subpath == "up-half":
            h = I.shape[0]
            I = I[:int(h/2),:,:]
        elif subpath == "down-half":
            h = I.shape[0]
            I = I[:int(h/2),:,:]                        
        else:
            pass
        if isshow:
            cv2.imshow(label,I)
            cv2.waitKey()
            cv2.destroyWindow(label)
        return I, label
    def facelocalization(self,sample,model="hog",wh=None,isshow=False):
        '''
        Should not use this function with cropped face
        '''
        I = face_recognition.load_image_file(sample)
        face_locations = face_recognition.face_locations(I, model=model)
        if isshow:
            I2s = copy.copy(I)
            I2s = cv2.cvtColor(I2s, cv2.COLOR_RGB2BGR)
            color = (255, 0, 0)
            thickness = 2
            for lo in face_locations:
                start_point = (lo[0],lo[1])
                end_point = (lo[2],lo[3])
                I2s = cv2.rectangle(I2s, start_point, end_point, color, thickness) 
            label = sample.split("/")[0]
            cv2.imshow(label,I2s)
            cv2.waitKey()
            cv2.destroyWindow(label)
        return I,face_locations
    
    def facelandmark(self,sample,model="hog",wh=None,isshow=False):
        I = face_recognition.load_image_file(sample)
        face_landmarks_list = face_recognition.face_landmarks(I)
        for item in face_landmarks_list:
            for k,v in item.items():
                print(k)
                for p in v:
                    centerOfCircle = p
                    radius = 10
                    color = [255,255,0]
                    thickness = 5
                    I = cv2.circle(I, centerOfCircle, radius, color, thickness)
                cv2.imshow("test",I)
                cv2.waitKey()
                cv2.destroyWindow("test")
    
    ######################################################
    def full_mask_face_pair_gen(self,**kargs):
        while True:
            pairs,targets = self.load_fullface_maskface_pair(**kargs)
            yield (pairs,targets)

    def load_fullface_maskface_pair(self,wh=None,bz=20,check_pair=False):
        matchn = int(bz/2)
        targets = np.zeros((bz,1))
        targets[0:matchn,0] = 1 # match
        pairs = [[] for i in range(2)]
        fullface_label_col = []
        maskface_label_col = []
        for bzid in range(bz):
            fid = rng.randint(0,len(self.faceList))
            # make sure maskface has image
            while len(self.maskfaceList[fid])==0:
                fid = rng.randint(0,len(self.faceList))
            fullface_label_col.append(self._getId(self.faceList[fid][0].split("/")[0]))
            imgid = rng.choice(len(self.faceList[fid]),1,replace=False)[0]
            
            fullfaceI,_ = self.load_sample(self.faceList[fid][imgid],face_type="face",wh=wh)
            pairs[0].append(fullfaceI)
            if bzid >= matchn:
                ct = True
                while ct:
                    nfid = rng.randint(0,len(self.faceList))
                    if nfid != fid and len(self.maskfaceList[nfid])>0:
                        ct = False
                fid = nfid
            maskface_label_col.append(self._getId(self.faceList[fid][0].split("/")[0]))
            imgid = rng.choice(len(self.maskfaceList[fid]),1,replace=False)[0]
            maskfaceI,_ = self.load_sample(self.maskfaceList[fid][imgid],face_type="maskface",wh=wh)
            pairs[1].append(maskfaceI)
        
        pairs[0],pairs[1],targets,maskface_label_col,fullface_label_col = shuffle(pairs[0],
                                                                                  pairs[1],
                                                                                  targets,maskface_label_col,
                                                                                  fullface_label_col)
        pairs[0] = np.array(pairs[0])
        pairs[1] = np.array(pairs[1])
        if check_pair:
            for k in range(bz):
                cv2.imshow("{}_{}_0".format(fullface_label_col[k], targets[k,0]),pairs[0][k])
                cv2.imshow("{}_{}_1".format(maskface_label_col[k], targets[k,0]),pairs[1][k])
                cv2.waitKey()
                cv2.destroyAllWindows()
        return pairs,targets
        