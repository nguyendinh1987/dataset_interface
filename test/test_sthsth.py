import os, sys
import numpy as np
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.something_something_dataset import sthsth
import cv2

config = {}
config["rootF"] = "/opt/Data/sthsthv1/videos"
config["mode"] = "train"
config["alllabels_file"] = "/opt/Data/sthsthv1/annotations/something-something-v1-labels.csv"
config["filelistFi"] = "/opt/Data/sthsthv1/annotations/something-something-v1-validation.csv"
config["verbose"] = True

sthsth_data = sthsth(**config)
test_config = {}
test_config["get_video"] = False
test_config["_getLabelId"] = True
test_config["_getLabelDes"] = False
test_config["get_batch"] = False
test_config["gen_batch_np"] = False
test_config["gen_batch_tf"] = False
test_config["get_hist_vlength"] = False
test_config["merge_segchg"] = False
test_config["get_similar_videos"] = False

if test_config["get_video"]:
    print("Test get_video")
    v = sthsth_data.get_video(vnum=17,isshow=True,ts=10,isgrey=True)
    for fid,f in enumerate(v):
        cv2.imwrite("/opt/workspace/output/tmpdata/{}.jpg".format(fid),f)

if test_config["_getLabelId"]:
    print("test _getLabelId")
    lid = sthsth_data._getLabelId(vnum=68044)
    print(lid)

if test_config["_getLabelDes"]:
    print("test _getLabelDes")
    ldes = sthsth_data._getLabelDes(lId = 0)
    print(ldes)

if test_config["get_batch"]:
    print("test get_batch")
    augconf = {}
    augconf["fwh"] = [100,100]
    augconf["intraratio"] = 0.85
    augconf["keepratio"] = False
    augconf["change_intraratio"]=True
    augconf["ctcrop"] = False
    augconf["ncrop"] = 5
    augconf["vscale"] = [0.7,1.5]
    sthsth_data.get_batch(nv=5,wh=[100,100],check_bsz=True,augconf=augconf)

if test_config["gen_batch_np"]:
    print("test gen_batch_np")
    augconf = {}
    augconf["fwh"] = [100,100]
    augconf["intraratio"] = 0.85
    augconf["keepratio"] = False
    augconf["change_intraratio"]=True
    augconf["ctcrop"] = False
    augconf["ncrop"] = 5
    augconf["vscale"] = [0.7,1.5]
    batch_gen = sthsth_data.gen_batch(opt="np",isshuffle=True,bsz=3,wh=[130,100],check_bsz=True,ts=1000,augconf=augconf)
    for _ in range(5):
        print("start gen batch")
        d,t = next(batch_gen)
        print(d.shape)
        print(t)

if test_config["gen_batch_tf"]:
    print("test gen_batch_tf")
    augconf = {}
    augconf["fwh"] = [100,100]
    augconf["intraratio"] = 0.7
    augconf["keepratio"] = False
    augconf["change_intraratio"]=True
    augconf["ctcrop"] = False
    augconf["ncrop"] = 5
    augconf["vscale"] = [0.7,1.5]
    batch_gen = sthsth_data.gen_batch(opt="tf",isshuffle=True,bsz=1,wh=[130,100],ngenclips=3,check_bsz=False,ts=10,augconf=augconf)
    for idx in range(2):
        if idx%100==0:
            print("-->[{}/20000]".format(idx))
        d,t = batch_gen.__getitem__(idx)
        bz,nseg,_,_,_ = d.shape
        for bid in range(bz):
            vname = sthsth_data._getLabelDes(t[bid])    
            for fid in range(nseg):
                cv2.imshow("{}".format(vname),d[bid,fid]/255)
                cv2.waitKey(500)
            cv2.destroyWindow("{}".format(vname))
            print("Done one clip")

if test_config["get_hist_vlength"]:
    print("test get_hist_vlength")
    _,_,_ = sthsth_data.get_hist_vlength()

if test_config["merge_segchg"]:
    print("check merge seg and channel")
    print("Loading data ....")
    data,_ = sthsth_data.get_batch(bsz=10,wh=[130,100],check_bsz=True,isgrey=True)
    print(("Merging dimension ...."))
    data = sthsth_data.merge_segchg(data,do_check=True)

if test_config["get_similar_videos"]:
    vlist = sthsth_data.get_similar_videos_list(vidx = 1)
    print(vlist)

