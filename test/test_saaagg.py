import os, sys
import numpy as np
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.saaagg import saaagg

data_config = {}
data_config["dataRoot"] = "/opt/Data/SAA_Aggression"
data_config["mode"] = "train"
data_config["splitFile"] = "/opt/workspace/output/datasplit/speclist_train.txt"
data_config["output"] = "/opt/workspace/output"
data_config["subVideoFolder"] = "TrainVal_Full"
saadata = saaagg(**data_config)
test_opt = {}
test_opt["getVideoWithSpec"] = True
test_opt["getVpath"] = False
test_opt["loadVideo"] = False
test_opt["loadbatch_for_classification"] = False
test_opt["genbatch_for_classification"] = False
test_opt["traintestsplit"] = False

if test_opt["getVideoWithSpec"]:
    # this combination has 200 samples, could be enough for domain adaptation
    # spec = ["NonAgg_20180606","NonAgg_20180604"]
    specs = ["BoxingSpeedBag",
            "Fencing",
            "Punch",
            "Drumming",
            "HeadMassage",
            "JumpingJack",
            "JumpRope",
            "PullUps",
            "PushUps",
            "taichi",
            "WallPushups",
            #### Real videos
            "5_31_2016",
            "19_3_2018",
            "20180604",
            "20180606",
            "20180703",
            "a_",
            "AGT_",
            "b_",
            "c_",
            "e_",
            "f1_",
            "f2_",
            "f3_",
            "f4_",
            "f5_",
            "GeylangMockup",
            "p1_",
            "p2_",
            "p3_",
            "p4_",
            "p5_",
            "p6_",
            "p7_",
            "p8_",
            "p9_",
            "Tanglin",
            "vid1",
            "vid2",
            "vid3",
            "vid4",
            "vid5",
            "vid6",
            "vid7",
            "vid8"]
    for spec in specs:
        for act in ["Agg","NonAgg"]:
            tmp = act+"_"+spec
            vlist = saadata.getVListWithSpec([tmp])
            print("{}: {}".format(tmp,len(vlist)))
        print("==============================")

if test_opt["getVpath"]:
    saadata.getVpath(10)

if test_opt["loadVideo"]:
    vpath = saadata.getVpath(15)
    print(vpath)
    V = saadata.loadVideo(vpath,isshow=True)

if test_opt["loadbatch_for_classification"]:
    saadata.loadbatch_for_classification(isgrey=False,isshow=True)

if test_opt["genbatch_for_classification"]:
    lbconf={}
    lbconf["isgrey"]=True
    lbconf["return_array"]=True
    lbconf["isshow"]=False
    lbconf["nseg"]=8
    datagen = saadata.genbatch_for_classification(bts=10,wh=[120,100],lbconf=lbconf)
    for r in range(175):
        print(r)
        for data,target in datagen:
            print(data.shape)
            break
if test_opt["traintestsplit"]:
    vtrain,vtest = saadata.traintestsplit(savetofiles=["/opt/workspace/output/train0.txt","/opt/workspace/output/val0.txt"])
    print(len(vtrain))
    print(len(vtest))

