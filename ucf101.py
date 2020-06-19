import random
# from moviepy.editor import *
import numpy as np
import numpy.random as rng
import cv2
import pickle
import os

class ucf101(object):
    def __init__(self,data_source=None,data_outpath=None,splitId=1,mode="train"):
        assert data_source is not None, "Please provide partition to data_source"
        assert splitId in [1,2,3], "splitId must be in [1,2,3]"
        assert mode in ["train","test"], "mode must be in ['train','test']"
        self.mode=mode
        self.split_file = data_source+"/ucfTrainTestlist/{}list0{}.txt".format(mode,splitId)
        self.data_source= data_source+"/UCF-101"
        self.data_outpath = data_outpath
        self.classID_file = data_source+"/ucfTrainTestlist/classInd.txt"
        self.vlist = None
        self.vtarget = None
        self.nv = None

        with open(self.classID_file, 'r') as f:
            classes = f.readlines()
            classes = map(lambda cls: cls.replace('\n','').split(' '), classes)
            classes = dict(map(lambda cls: (cls[1], int(cls[0])), classes))
        self.classIDs = classes
    ####################################################################################################
    # UTILS
    ####################################################################################################
    def get_number_of_videos(self):
        return self.nv
    def get_classes(self):
        """
        Returns
        -------
        dict
            Dictionary of class names and numeral id
            Example: {'Class1': 1, 'Class2': 2}
        """
        return self.classIDs

    def get_class_id(self,vname):
        if vname.split("_")[1]=="HandStandPushups":
            label_id = self.classIDs["HandstandPushups"]
        else:
            label_id = self.classIDs[vname.split("_")[1]]
        return label_id

    def get_label_name(self,ID):
        for k,v in self.classIDs.items():
            if v==ID:
                return k
        return None

    def get_fullpath(self,vname):
        if vname.split("_")[1]=="HandStandPushups":
            vname = self.data_source+"/HandstandPushups/"+vname
        else:
            vname = self.data_source+"/"+vname.split("_")[1]+"/"+vname
        return vname

    ####################################################################################################
    ####################################################################################################
    def load_all(self,wh,debug=True):
        def display(loaded_videos):
            for vid,vn in enumerate(loaded_videos["vname"]):
                if vid+1%5==0:
                    break
                startf = loaded_videos["start_end_fid"][0,0]
                endf = loaded_videos["start_end_fid"][0,1]
                for fid in range(startf,endf):
                    f = loaded_videos["data"][fid,:]
                    I = f.reshape([wh[1],wh[0],3])
                    cv2.imshow("{}_{}".format(vn,loaded_videos["label"][vid]),I)
                    cv2.waitKey()
                cv2.destroyWindow("{}_{}".format(vn,loaded_videos["label"][vid]))

        assert self.data_outpath is not None, "This process take long time to complete. please provide data_outpath to save the loaded data"
        if os.path.isfile(self.data_outpath):
            opt = input("{} is existed. If you want to load it, pls type [yes]: ")
            if opt == "yes":
                f = open(self.data_outpath,"rb")
                self.loaded_videos = pickle.load(f)
                f.close()
                if debug:
                    display(self.loaded_videos)
                return self.loaded_videos

        # check output folder
        outfolder = self.data_outpath[0:-len(self.data_outpath.split("/")[-1])]
        print(outfolder)
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        
        if self.vlist is None:
            self.get_split_info()
        loaded_videos = {"vname":self.vlist,
                         "label":[],
                         "start_end_fid":None,
                         "data":None}
        prevendfid = 0
        for vid,v in enumerate(self.vlist):
            v,l = self.get_sample_video(self.get_fullpath(v),wh=wh)
            loaded_videos["label"].append(l)
            startfid = prevendfid
            endfid = startfid+len(v)
            prevendfid = endfid
            print(len(v))
            for f in v:
                flatten_f = f.reshape((1,-1))
                flatten_stnd_id = np.array([[startfid,endfid]])
                if loaded_videos["data"] is None:
                    loaded_videos["data"] = flatten_f
                    loaded_videos["start_end_fid"] = flatten_stnd_id
                else:
                    loaded_videos["data"] = np.vstack((loaded_videos["data"],flatten_f))
                    loaded_videos["start_end_fid"] = np.vstack((loaded_videos["start_end_fid"],flatten_stnd_id))
                        
            if vid%2==0:
                print("loaded [{}/{}]".format(vid,self.nv))
                
        
        if debug:
            display(loaded_videos)

        self.loaded_videos = loaded_videos

        f = open(self.data_outpath,"wb")
        pickle.dump(loaded_videos,f)
        f.close()
        return self.loaded_videos

    def get_split_info(self):
        """
        Loads a text file with a list of filenames that should be used as dataset
        
        Returns
        -------
        list(string)
            Returns a list of filenames for dataset
        """
        if self.vlist is not None:
            print("get_split_info has been ran before")
            return [self.vlist,self.vtarget]

        with open(self.split_file, 'r') as f:
            split_info = f.readlines()
            split_info = list(map(lambda file: file.replace('\n','').split('/')[1], split_info))
        if self.mode=="test":
            vlist = split_info
            vtarget = None
        else:
            vlist = list(map(lambda file: file.split(" ")[0],split_info))
            vtarget = list(map(lambda file: int(file.split(" ")[1]),split_info))
        self.vlist = vlist
        self.vtarget = vtarget
        self.nv = len(vlist)
        return [vlist,vtarget]

    def get_sample_video(self,vname=None,isshow=False,wh=None,verbose=False):
        if vname is None:
            assert self.nv is not None, "Please run a function get_split_info() first or provide path to video"
            vname = self.vlist[rng.choice(self.nv,size=(1,),replace=False)[0]]
            vname = self.get_fullpath(vname) #self.data_source+"/"+vname.split("_")[1]+"/"+vname
        if self.vtarget is None:
            label = None
        else:
            label = self.get_class_id(vname)
            # if vname.split("_")[1]=="HandStandPushups":
            #     label = self.classIDs["HandstandPushups"]
            # else:
            #     label = self.classIDs[vname.split("_")[1]]
        
        if verbose:
            print("Reading video {} with label {}".format(vname,label))
        
        vcap = cv2.VideoCapture(vname)
        vframes = []

        while(vcap.isOpened()):
            ret, frame = vcap.read()
            if ret:
                if wh is None:
                    vframes.append(frame)
                else:
                    vframes.append(cv2.resize(frame,(wh[1],wh[0])))
            else:
                break
        vcap.release()

        if isshow:
            for f in vframes:
                cv2.imshow("video",f)
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    break
            cv2.destroyWindow("video")
            
        return vframes,label
    
    def get_batch(self,N=10,IDs=None,vl=50,wh=[256,256],check_batch=False,iscolor=True,provide_vtensor=False):
        videos = []#np.zeros((N,wh[1],wh[0],3))
        if self.mode == "train":
            if IDs is None:
                targets = np.zeros((N,len(self.classIDs.keys())))
            else:
                N = len(IDs)
                targets = np.zeros((N,len(self.classIDs.keys())))
                for id,vid in enumerate(IDs):
                    targets[id,vid-1] = 1
        else:
            targets = None
            if IDs is not None:
                N = len(IDs)

        labels = []
        loaded_video = []
        if IDs is not None:
            watchingIDs = np.unique(np.array(IDs))
            watchingVs = [[] for i in range(watchingIDs.shape[0])]
            for v in self.vlist:
                lv = self.get_class_id(v)
                # if v.split("_")[1]=="HandStandPushups":
                #     lv = self.classIDs["HandstandPushups"]
                # else:
                #     lv = self.classIDs[v.split("_")[1]]
                if lv in watchingIDs:
                    watchingVs[np.where(watchingIDs==lv)[0][0]].append(v)
        for vId in range(N):
            while(True):
                if IDs is None:
                    vname = self.vlist[rng.choice(self.nv,size=(1,),replace=False)[0]]
                else:
                    # get a random video from videos belonging to class ID
                    ##############
                    lv = IDs[vId]
                    vlist = watchingVs[np.where(watchingIDs==lv)[0][0]]
                    # print("{}: {}".format(lv,len(vlist)))
                    vname = vlist[rng.choice(len(vlist),size=(1,),replace=False)[0]]
                    ##############
                vname_short = vname.split("_")[2]+"_"+vname.split("_")[3]
                if vname_short in loaded_video:
                    continue
                loaded_video.append(vname_short)
                break
            
            vname = self.get_fullpath(vname)
            # if vname.split("_")[1] == "HandStandPushups":
            #     vname = self.data_source+"/HandstandPushups/"+vname
            # else:    
            #     vname = self.data_source+"/"+vname.split("_")[1]+"/"+vname
            
            V,l = self.get_sample_video(vname=vname,wh=wh)
            videos.append(V)
            if l is None:
                labels.append("None")
            else:
                labels.append(self.get_label_name(l)+"_"+str(l))
            if IDs is None and self.mode == "train":
                targets[vId,int(l)-1] = 1
        
        # if provide_vtensor:
        #     print("THIS OPTION WILL BE REMOVED IN FUTURE!!!!!!!!!")
        #     video_tensor = np.zeros((N,vl,wh[1],wh[0],3))
        #     for vId in range(N):
        #         duplastf = False
        #         if len(videos[vId]) <= vl:
        #             startp = 0
        #             if len(videos[vId])<vl:
        #                 duplastf = True
        #         else:
        #             startp = rng.randint(0,len(videos[vId])-vl)
                
        #         addfc = -1
        #         for fId in range(startp,min(len(videos[vId]),startp+vl)):
        #             addfc += 1
        #             # video_tensor[vId,:,:,3*addfc:3*(addfc+1)] = videos[vId][fId]
        #             video_tensor[vId,addfc,:,:,:] = videos[vId][fId]
        #         if duplastf:
        #             startp = len(videos[vId])
        #             for f in range(vl-len(videos[vId])):
        #                 # video_tensor[vId,:,:,3*(startp+f):3*(startp+f+1)] = videos[vId][-1]
        #                 video_tensor[vId,startp+f,:,:,:] = videos[vId][-1]

        #     if check_batch:
        #         for vId in range(N):
        #             for f in range(vl):
        #                 cv2.imshow("video {}".format(labels[vId]),video_tensor[vId,f,:,:,:]/255)
        #                 if cv2.waitKey(15) & 0xFF == ord("q"):
        #                     break
        #             cv2.destroyAllWindows()
        # else:
        #     print("PLEASE SET provide_vtensor=True to get video tensor for c3d network")
        #     video_tensor = None
        video_tensor = None
        return videos,labels,video_tensor,targets

    
        
'''
def random_frames(video, target_num_frames):
    """
    From an input video clip it randomly selects and returns a specified number of frames in sorted order
    
    Parameters
    ----------
    video : array, required
        Array of multiple elements (frames)
    target_num_frames : int, required
        How many elements (frames) to randomly select and return
    
    Returns
    -------
    array
        Subsample of input array containing randomly selected elements (frames)
    """
    num_frames = video.shape[0]
    frames_idxs = random.sample(range(0, num_frames), target_num_frames)
    frames_idxs.sort()
    return video[frames_idxs]


def crop_frame(img, width, height, x, y):
    """
    Returns a crop of image (frame) based on specified parameters
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, required
        Width of the crop
    height : int, required
        Height of the crop
    x : int, required
        X position of the crop, by default None
    y : int, required
        Y position of the crop, by default None
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    img = img[y:y+height, x:x+width]
    return img

def random_crop(img, width=112, height=112):
    """
    Returns a random crop of image (frame) based on specified width and height
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, optional
        Width of the crop, by default 112
    height : int, optional
        Height of the crop, by default 112
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    return crop_frame(img, width, height, x, y)


def crop_center(img, width=112, height=112):
    """
    Returns a centered crop of image (frame) based on specified width and height
    
    Parameters
    ----------
    img : array, required
        Array representing an image (frame)
    width : int, optional
        Width of the crop, by default 112
    height : int, optional
        Height of the crop, by default 112
    
    Returns
    -------
    array
        Cropped image (frame) based on input parameters
    """
    x = (img.shape[1] - width) // 2
    y = (img.shape[0] - height) // 2
    return crop_frame(img, width, height, x, y)


def extract_clips(video, frames_per_clip=16, step=None):
    """
    Extracts clips from input clip based on specified parameters
    
    Parameters
    ----------
    video : array
        Array representing a video consisting of multiple frames (images)
    frames_per_clip : int, optional
        Number of frames for output clips, by default 16
    step : int, optional
        Step value to iterate through input frames, by default None
    
    Returns
    -------
    array
        Array of video clips extracted from input video clip
    """
    if step is None:
        step = frames_per_clip
    total_frames = video.shape[0]
    assert frames_per_clip <= total_frames
    extracted_clips = [np.asarray(video[i:i+frames_per_clip]) for i in range(0, total_frames-frames_per_clip, step)]
    
    return np.asarray(extracted_clips, dtype=video.dtype)


def close_clip(video):
    """
    Closes the connection to the video file
    
    Parameters
    ----------
    video : VideoFileClip object
        MoviePy VideoFileClip object to close and delete
    """
    if video is not None:
        video.close()
    del video


def calculate_mean_std(x, channels_first=False, verbose=0):
    """
    Calculates channel-wise mean and std
    
    Parameters
    ----------
    x : array
        Array representing a collection of images (frames) or
        collection of collections of images (frames) - namely video
    channels_first : bool, optional
        Leave False, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array of shape [2, num_channels]
        Array with per channel mean and std for all the frames
    """
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    all_mean = []
    all_std = []    
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):
        if ndim ==5: # videos
            mean = x[:,:,:,:,c].mean()
            std = x[:,:,:,:,c].std()
        elif ndim ==4: # images rgb or grayscale
            mean = x[:,:,:,c].mean()
            std = x[:,:,:,c].std()
        if verbose:
            print("Channel %s mean before: %s" % (c, mean))   
            print("Channel %s std before: %s" % (c, std))
            
        all_mean.append(mean)
        all_std.append(std)
    
    return np.stack((all_mean, all_std))


def preprocess_input(x, mean_std, divide_std=False, channels_first=False, verbose=0):
    """
    Channel-wise substraction of mean from the input and optional division by std
    
    Parameters
    ----------
    x : array
        Input array of images (frames) or videos
    mean_std : array
        Array of shape [2, num_channels] with per-channel mean and std
    divide_std : bool, optional
        Add division by std or not, by default False
    channels_first : bool, optional
        Leave False, otherwise not implemented, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array
        Returns input array after applying preprocessing steps
    """
    x = np.asarray(x, dtype=np.float32)    
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):  
        if ndim ==5: # videos
            x[:,:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,:,c] /= mean_std[1][c]
            if verbose:
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,:,c].std()))
        elif ndim ==4: # images rgb or grayscale
            x[:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,c] /= mean_std[1][c]   
            if verbose:        
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,c].std()))            
    return x


def predict_c3d(x, model):
    """
    Runs predictions on specified model and returns them
    
    Parameters
    ----------
    x : array
        Input array with data propper for input shape of the model
    model : Keras model object
        Model object that will be used for inferencing
    
    Returns
    -------
    array
        Array with output predictions returned by Keras model
    """
    pred = []
    for batch in x:
        pred.append(model.predict(batch))
    return pred

'''