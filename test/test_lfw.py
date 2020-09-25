import os, sys
import numpy as np
# os.environ["PYTHONPATH"] = "/opt/workspace/python"
sys.path.append("/opt/workspace/python/dataset_interface/codes")
from lfw_dataset import lfw_face

config = {"verbose":True
          ,"imagesF":"/opt/Data/lfw"
          ,"cache": "/opt/workspace/output/lfw"
          ,"datainfofile":"/opt/workspace/output/lfw/datainfo.txt"
          ,"focuslistF":"/opt/workspace/output/lfw/test.txt"
          }

facedata = lfw_face(**config)
# facedata.datasplit()
for i in range(10,20):
    facedata.loadsample(id=i,isshow=True)

# facelist = facedata.getFaceList(group="maskedface")
# print(facelist[20][0])
# I,l = facedata.load_samples(samples=[facelist[20][0],facelist[21][0]],face_type="maskface",isshow=True)
# facedata.load_fullface_maskface_pair(wh=[96,96],check_pair=True)