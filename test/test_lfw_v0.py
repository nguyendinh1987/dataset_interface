import os, sys
import numpy as np
# os.environ["PYTHONPATH"] = "/opt/workspace/python"
sys.path.append("/opt/workspace/python")
from lfw_dataset import lfw_face

config = {"verbose":True,
          "prog_output":"/opt/workspace/face_identification/data/lfw",
          "maskedface":"/opt/Data/processed_lfw/maskfaces",
          "use_all":False,
          "sublabels_file":"rest_train_0.txt"}

facedata = lfw_face(rootF="/opt/Data/processed_lfw/croppedfaces",**config)
# facelist = facedata.getFaceList(group="maskedface")
# print(facelist[20][0])
# I,l = facedata.load_samples(samples=[facelist[20][0],facelist[21][0]],face_type="maskface",isshow=True)
# facedata.load_fullface_maskface_pair(wh=[96,96],check_pair=True)