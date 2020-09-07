import os, sys
import numpy as np
# os.environ["PYTHONPATH"] = "/opt/workspace/python"
sys.path.append("/opt/workspace/python/dataset_interface")
from codes.imagenet_download import imagenet_API_interface

imgAPI = imagenet_API_interface()
print("load wordidmap from url")
imgAPI.download_wordnetmap("/opt/Data/imagenet/wordidmap.txt")
print("start downloading images")
wordids = imgAPI.download_images("bird","/opt/Data/imagenet/images/bird")