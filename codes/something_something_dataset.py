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

        # self.tfgen_opt = False if "tfgen" not in kargs.keys() else kargs["tfgen"]
        # if self.tfgen_opt:
        #     from .sthsthtfgenerator import sthsthtfgen as tfgen
        #     print(tfgen)

        self.print_ctcrop_warning = True
    
    ## utis
    def _getAllLabels(self):
        self.labels = pd.read_csv(self.labelFi,usecols=[0,1],header=None).fillna(" ")
        if self.verbose:
            print("There are {} actions".format(self.labels.shape[0]))
            print(self.labels.head(5))
            # print(self.labels.iloc[0][0])
            # print(self.labels[self.labels[0]=="Folding something"].index.tolist()[0])
            # to get label description: use self.labels.iloc[id][0]
            # to get label id: use self.labels[self.labels[0]==<label_description>].index.tolist()[0]
        return True
    
    def _getFileList(self):
        self.files = pd.read_csv(self.filelistFi,header=None).fillna(" ")
        if self.verbose:
            print("There are {} files".format(self.files.shape[0]))
            print(self.files.head(50))
        return True

    def _getLabelId(self,vnum):
        findex = self.files[self.files[0]==vnum].index.tolist()[0]
        fdes = self.files.iloc[findex][1:3]
        # print((self.labels[0]==fdes[1]) & (self.labels[1]==fdes[2]))
        # print(self.labels[0]==fdes[0])
        lid = self.labels[(self.labels[0]==fdes[1]) & (self.labels[1]==fdes[2])].index.tolist()[0]
        # if self.verbose:
        #     print("vnum: {}; lid: {}; des: {}".format(vnum,lid,fdes))
        return lid

    def _getLabelDes(self,lId):
        ldes = self.labels.iloc[lId][0]+"_"+self.labels.iloc[lId][1]
        return ldes
    
    ## interface with data
    def get_nclasses(self):
        return self.labels.shape[0]
    def get_labels(self):
        return self.labels[0].tolist()
    def get_nvideos_per_label(self,l=None):
        if l is None:
            return None
        else:
            vindices = self.files[self.files[1]==l].index.tolist()
            vlist = self.files.iloc[vindices][0].tolist()
            return len(vlist),vlist

    def get_video(self,vnum = None,wh = None, rootF = "20bn-something-something-v1",isshow=False,ts=15,isgrey=False):
        if vnum is None:
            vnum = self.files.iloc[rng.randint(0,self.files.shape[0])][0]
        path_to_video = os.path.join(self.rootFo,rootF,str(vnum))
        frame_list = glob.glob(path_to_video+"/*.jpg")
        frame_nums = [int(fn.split("/")[-1].split(".")[0]) for fn in frame_list]
        frame_indices = np.argsort(frame_nums)
        frame_list = [frame_list[fid] for fid in frame_indices]

        v = []
        for fl in frame_list:
            frame = cv2.imread(fl)
            if wh is not None:
                frame = cv2.resize(frame,(wh[0],wh[1]))
            if isgrey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame,axis=-1)
            v.append(frame)

        if isshow:
            for f in v:
                cv2.imshow("v",f)
                cv2.waitKey(ts)
                cv2.destroyWindow("v")
        
        return v

    def get_similar_videos_list(self,vidx = None):
        if vidx is None:
            return None
        print(self.files.head(10))
        vnum = self.files[0][vidx]
        lid = self._getLabelId(vnum)
        cldes = self.labels.iloc[lid][0:2]
        print(cldes)
        print(lid)
        vindices = self.files[(self.files[1]==cldes[0]) & (self.files[2]==cldes[1])].index.tolist()
        vlist = self.files.iloc[vindices][0].tolist()
        # print(vlist)
        return vlist
    
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
    
    def vcrop(self,v,newhw,ctcrop,ncrop):
        hw = v[0].shape[0:2]

        assert hw[0] >= newhw[0]
        assert hw[1] >= newhw[1]
        vs = []
        for nc in range(ncrop):
            if ctcrop:
                hwc = [x/2 for x in hw]
                hwdis = [newhw[i]/2 for i in range(2)]
                hwstart = [max(0,int(hwc[i]-hwdis[i])) for i in range(2)]
            else:
                hwstart = [0,0]
                if hw[0]-newhw[0]>0:
                    hwstart[0] = rng.choice(hw[0]-newhw[0],1,replace=False)[0]
                if hw[1]-newhw[1]>0:
                    hwstart[1] = rng.choice(hw[1]-newhw[1],1,replace=False)[0]

            hwend = [min(hw[i],hwstart[i]+newhw[i]) for i in range(2)]
            
            # print("orginal hw: {}".format(hw))
            # print("new hw: {}".format(newhw))
            # print("hw start: {}".format(hwstart))
            # print("hw end: {}".format(hwend))
            
            newv = copy.copy(v)
            for fid in range(len(newv)):
                # newv[fid] = cv2.rectangle(newv[fid],(hwstart[1],hwstart[0]),(hwend[1],hwend[0]),color=(0,255,0),thickness=3)
                # newv[fid] = cv2.resize(newv[fid],(newsize[0],newsize[1]))
                newv[fid] = newv[fid][hwstart[0]:hwend[0],hwstart[1]:hwend[1]]
            vs.append(newv)

        return vs

    def do_crop(self,v,adjratio,ctcrop,ncrop):
        c = v[0].shape[2]
        orgs = v[0].shape[0:2]

        newhw = [int(adjratio*orgs[0]),int(adjratio*orgs[1])]        

        if ctcrop:
            if self.print_ctcrop_warning:
                print("***************************************************")
                print("set ctcrop to False for multiple cropping processes")
                print("***************************************************")
                self.print_ctcrop_warning = False
            ncrop = 1
        
        vs = self.vcrop(v,newhw,ctcrop,ncrop)

        return vs

    def video_augs(self,v,fwh,intraratio,ctcrop,ncrop,vscale=None,flipf=False):
        '''
        fwh: final size of augmented frames
        kratio; [True,False] keep original ratio or not
        minsz: if kratio == True, frames are resized so that the shorted size = minsz
        rsize; It is rateresize. if kratio == False, frames are resized randomly with ratio from 1 to rsize
        ctcrop: if it is True, only provide center crop, otherwise it will provide random crop
        ncrop: a number of crop
        #########################################################################################################
        if you want to use flipf, please take care videos relating to move sth from left to right and vise versa
        #########################################################################################################
        '''
        hw = v[0].shape[0:2]
        c = v[0].shape[2]
        if vscale is not None:
            rsize = rng.uniform(vscale[0],vscale[1])
            newsize = [int(hw[i]*rsize) for i in range(2)]
            for fid in range(len(v)):
                v[fid] = cv2.resize(v[fid],tuple(newsize))

        vs = self.do_crop(v,intraratio,ctcrop,ncrop)
        
        for vid in range(len(vs)):
            doflipf = flipf and np.random.randint(2)==1
            for fid in range(len(vs[vid])):
                vs[vid][fid] = cv2.resize(vs[vid][fid],tuple(fwh))
                if doflipf:
                    vs[vid][fid] = np.flip(vs[vid][fid],1)
                if c==1:
                    vs[vid][fid] = np.expand_dims(vs[vid][fid],axis=-1)

        return vs
        
    def gen_batch(self,opt="np",bsz=10,isshuffle=True,merge_sgch=False,n=1,**kargs):
        if opt=="np":
            return self.gen_batch_np(bsz=bsz,isshuffle=isshuffle,merge_sgch=merge_sgch,**kargs)
        elif opt=="tf":
            return self.gen_batch_tf(bsz=bsz,isshuffle=isshuffle,merge_sgch=merge_sgch,n=n,**kargs)
        else:
            pass
    def gen_batch_tf(self,bsz=10,isshuffle=True,merge_sgch=False,**kargs):
        from .sthsthtfgenerator import sthsthtfgen as tfgen
        tfgenerator = tfgen(self,bts=bsz,isshuffle=isshuffle,merge_sgch=merge_sgch,**kargs)
        return tfgenerator

    def gen_batch_np(self,bsz=10,isshuffle=True,merge_sgch=False,n=None,**kargs):
        # print("xxxx")
        # bsz = kargs["bsz"]
        filelist = self.files[0].values
        nfiles = filelist.shape[0]
        # print("nfiles: {}; bsz: {}".format(nfiles,bsz))
        npatches = np.floor(nfiles/bsz).astype(np.int16)
        if npatches*bsz < nfiles:
            npatches += 1
        # print(npatches)
        
        while True:
            print(npatches)
            if isshuffle:
                filelist = shuffle(filelist)
            for p in range(npatches):
                # print("{} of {}".format(p+1,npatches))
                internal_bsz = min((p+1)*bsz,nfiles)-p*bsz
                # print("int_bsz: {}; bsz: {}".format(internal_bsz,bsz))
                startp = p*bsz
                endp = startp+internal_bsz
                vlist = filelist[startp:endp]
                internal_kargs = copy.copy(kargs)
                internal_kargs["nv"] = len(vlist)
                data,target = self.get_batch(vlist=vlist,**internal_kargs)
                if merge_sgch:
                    data = self.merge_segchg(data)
                del internal_kargs
                # print("end gen_batch")
                yield (data,target)

    def get_batch(self,vlist = None, nv = 1, nseg=8, wh = None, isgrey=False, augconf = None, check_bsz = False,ts=150):
        if vlist is None:
            fcids = rng.choice(self.files.shape[0],nv,replace=False)
            vlist = self.files[0][fcids]
            if check_bsz:
                print("Selected ids: {}".format(fcids))
                _ = input("Press any key to ctnue")
        else:
            assert len(vlist)>=nv, "number of videos in vlist should be larger than batch size (bsz)"
        # print(vlist)
        
        if augconf is not None:
            fwh = augconf['fwh'] # to video with its original size for doing augmentation
        else:
            fwh = wh

        data = []
        target = []#[0 for i in range(bsz)]

        for bid, vid in enumerate(vlist):
            if bid+1 > nv:
                break

            vframes = self.get_video(vnum=vid,isgrey=isgrey)#,isshow=check_bsz,ts=50)
            cutpoints = np.linspace(0,len(vframes),nseg+1,endpoint=True).astype(np.int16)
            
            selframes = []
            for pid in range(1,len(cutpoints)):
                selid = rng.choice(range(cutpoints[pid-1],cutpoints[pid]),1,replace=False)[0]
                selframes.append(vframes[selid])
            
            augvideos = None
            if augconf is not None:
                augvideos = self.video_augs(selframes,**augconf)
            else:
                for fid in range(len(selframes)):
                    selframes[fid] = cv2.resize(selframes[fid],tuple(fwh))
                augvideos = [selframes]
            
            tag = self._getLabelId(vid)
            for augv in augvideos:
                augv = np.array(augv)
                data.append(augv)
                target.append(tag)

        data,target = shuffle(data,target)
        data = np.array(data)
        # print(data.shape)
        if check_bsz:
            for bid in range(data.shape[0]):
                vname = self._getLabelDes(target[bid])
                print("{}: {}".format(target[bid],vname))
                _ = input("press any key to continue")
                for fid in range(nseg):
                    cv2.imshow("frames_{}".format(vname),data[bid,fid]/255)
                    cv2.waitKey(ts)
                cv2.destroyWindow("frames_{}".format(vname))
        # print("end get_batch")
        return data, target
