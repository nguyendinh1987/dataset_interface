import os, sys
import numpy as np
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.something_something_dataset import sthsth

config = {}
config["rootF"] = "/opt/Data/sthsthV1"
config["mode"] = "train"
config["alllabels_file"] = "/opt/Data/sthsthV1/datasplit_label/something-something-v1-labels.csv"
config["filelistFi"] = "/opt/Data/sthsthV1/datasplit_label/something-something-v1-train.csv"

sthsth_data = sthsth(**config)

