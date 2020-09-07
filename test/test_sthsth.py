import os, sys
import numpy as np
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.something_something_dataset import sthsth
import cv2

config = {}
config["rootF"] = "/opt/Data/sthsthV1"
config["mode"] = "train"
config["alllabels_file"] = "/opt/Data/sthsthV1/datasplit_label/something-something-v1-labels.csv"
config["filelistFi"] = "/opt/Data/sthsthV1/datasplit_label/something-something-v1-train.csv"

sthsth_data = sthsth(**config)
test_config = {}
test_config["get_video"] = True
test_config["_getLabelId"] = False
test_config["get_batch"] = False
test_config["gen_batch"] = False
test_config["get_hist_vlength"] = False
test_config["merge_segchg"] = False

if test_config["get_video"]:
    print("Test get_video")
    v = sthsth_data.get_video(vnum=17,isshow=True,ts=10,isgrey=True)
    for fid,f in enumerate(v):
        cv2.imwrite("/opt/workspace/output/tmpdata/{}.jpg".format(fid),f)

if test_config["_getLabelId"]:
    print("test _getLabelId")
    lid = sthsth_data._getLabelId(vnum=11778)

if test_config["get_batch"]:
    print("test get_batch")
    augconf = {}
    augconf["fwh"] = [100,100]
    augconf["kratio"] = False
    augconf["minsz"] = 128
    augconf["rsize"] = 2.0
    augconf["ctcrop"] = True
    augconf["ncrop"] = 4
    sthsth_data.get_batch(nv=5,wh=[130,100],check_bsz=True,augconf=augconf)

if test_config["gen_batch"]:
    print("test gen_batch")
    batch_gen = sthsth_data.gen_batch(isshuffle=True,bsz=5,wh=[130,100],check_bsz=True)
    for _ in range(5):
        print("start gen batch")
        for a in batch_gen:
            print("data shape")
            print(a[0].shape)
            break

if test_config["get_hist_vlength"]:
    print("test get_hist_vlength")
    _,_,_ = sthsth_data.get_hist_vlength()

if test_config["merge_segchg"]:
    print("check merge seg and channel")
    print("Loading data ....")
    data,_ = sthsth_data.get_batch(bsz=10,wh=[130,100],check_bsz=True,isgrey=True)
    print(("Merging dimension ...."))
    data = sthsth_data.merge_segchg(data,do_check=True)

