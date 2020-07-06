import os, sys
import numpy as np
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.saaagg import saaagg

data_config = {}
data_config["dataRoot"] = "/opt/Data/SAA_Aggression"
data_config["mode"] = "all"
data_config["output"] = "/opt/workspace/output"
data_config["subVideoFolder"] = "TrainVal_Full"
saadata = saaagg(**data_config)
test_opt = {}
test_opt["getVideoWithSpec"] = False
test_opt["getVpath"] = False
test_opt["loadVideo"] = False
test_opt["loadbatch_for_classification"] = False
test_opt["genbatch_for_classification"] = False
test_opt["traintestsplit"] = True

if test_opt["getVideoWithSpec"]:
    # this combination has 200 samples, could be enough for domain adaptation
    spec = ["NonAgg_20180606","NonAgg_20180604"]
    vlist = saadata.getVideoWithSpec(spec)
    print(len(vlist))

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
    lbconf["isgrey"]=False
    lbconf["return_array"]=True
    lbconf["isshow"]=True
    lbconf["nseg"]=8
    datagen = saadata.genbatch_for_classification(bts=3,wh=[120,100],lbconf=lbconf)
    for r in range(10):
        for data,target in datagen:
            print(data.shape)
            break
if test_opt["traintestsplit"]:
    vtrain,vtest = saadata.traintestsplit(savetofiles=["/opt/workspace/output/train0.txt","/opt/workspace/output/val0.txt"])
    print(len(vtrain))
    print(len(vtest))

