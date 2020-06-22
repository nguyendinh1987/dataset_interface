from xml.dom import minidom
from random import shuffle, randint
import numpy as np
import sys
import os
import glob
import time
import cv2
import copy

VOCOpts = {'DataRoot_test': '/media/kakadinh/Data/WorkOld/Datasets/Public_Dataset/VOC/VOCdevkit_test',
           'DataRoot_trainval': '/media/kakadinh/Data/WorkOld/Datasets/Public_Dataset/VOC/VOCdevkit_trainval',
           'Data': ['VOC2007','VOC2012'], # access by Data_id
           'anno': 'Annotations',
           'ImageSets': ['ImageSets/Layout','ImageSets/Main','ImageSets/Segmentation','ImageSets/Action'], # access by Set_id
           'ImgSource': 'JPEGImages',
           'Seg_GTmap': ['SegmentationClass','SegmentationObject'], # access by Seg_id
           'Classes': ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
                       'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant',
                       'sheep','sofa','train','tvmonitor'], # access by cls_id
            'Actions': ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto',
                        'usingcomputer', 'walking','others'], # access by act_id
            'view': ['frontal','rear','left','right','Unspecified'],
            'minoverlap': 0.5,
            'img_ext': 'jpg',
            'others': 'others'}

def xml2dict(f,need_print=False):
    Annotated_Info = minidom.parse(f)
    objects = Annotated_Info.getElementsByTagName('object')
    load_objects = {'name': [], # class name
                    'pose': [], # view
                    'truncated': [],
                    'difficult': [],
                    'occluded':[],
                    'bndbox': []}

    if len(objects) == 0:
        print("{} does not have objects information".format(f))
        return load_objects
    else:
        for obj in objects:
            # load class name
            load_objects['name'].append(obj.getElementsByTagName('name')[0].childNodes[0].data)
            # load box
            load_box = np.zeros((1,4))
            box = obj.getElementsByTagName('bndbox')
            load_box[0,0] = int(box[0].getElementsByTagName('xmin')[0].childNodes[0].data)
            load_box[0,1] = int(box[0].getElementsByTagName('ymin')[0].childNodes[0].data)
            load_box[0,2] = int(box[0].getElementsByTagName('xmax')[0].childNodes[0].data)
            load_box[0,3] = int(box[0].getElementsByTagName('ymax')[0].childNodes[0].data)
            load_objects['bndbox'].append(load_box.astype(int))
            # load view
            view = obj.getElementsByTagName('pose')
            if len(view) == 0:
                load_objects['pose'].append('')
            else:    
                load_objects['pose'].append(view[0].childNodes[0].data)
            # load difficult
            diff = obj.getElementsByTagName('difficult')
            if len(diff) == 0:
                load_objects['difficult'].append(None)
            else:
                load_objects['difficult'].append(int(diff[0].childNodes[0].data))
            # load truncated
            trunc = obj.getElementsByTagName('truncated')
            if len(trunc) == 0:
                load_objects['truncated'].append(None)
            else:
                load_objects['truncated'].append(int(trunc[0].childNodes[0].data))
            # load occluded
            occ = obj.getElementsByTagName('occluded')
            if len(occ) == 0:
                load_objects['occluded'].append(None)
            else:
                load_objects['occluded'].append(int(occ[0].childNodes[0].data))
    
    if need_print:
        for i in range(len(load_objects['name'])):
            print(load_objects['name'][i])
            print(load_objects['pose'][i])
            print(load_objects['truncated'][i])
            print(load_objects['difficult'][i])
            print(load_objects['occluded'][i])
            print(load_objects['bndbox'][i])
    
    return load_objects

def Show_single_image_objdet_gt(Img_name,VOCOpts = VOCOpts,Data_id=0,train_test=0):
    ######################################################################
    ######################################################################
    if len(Img_name.split('.')) > 2:
        print("{} has more than 1 dot. Cannot process".format(Img_name))
        return 1

    if train_test == 0:
        Imgpath = VOCOpts['DataRoot_trainval']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['ImgSource']+'/'+Img_name
        Annopath = VOCOpts['DataRoot_trainval']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['anno']+'/'+Img_name.split('.')[0]+'.xml'
    else:
        Imgpath = VOCOpts['DataRoot_test']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['ImgSource']+'/'+Img_name
        Annopath = VOCOpts['DataRoot_test']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['anno']+'/'+Img_name.split('.')[0]+'.xml'
    # print(Imgpath)
    # print(Annopath)

    Annotated_Info = minidom.parse(Annopath)
    if Annotated_Info.getElementsByTagName('filename')[0].childNodes[0].data.encode('utf-8') != Img_name:
        print("Accessed wrong annotation file")
        return 1
    
    load_objects = xml2dict(Annopath)

    # Show the Image and anotated object bounding boxes
    img = cv2.imread(Imgpath)
    if len(load_objects['name'])>0:
        for i in range(len(load_objects['bndbox'])):
            box = load_objects['bndbox'][i]
            if load_objects['difficult'][i] == 1:
                cv2.rectangle(img,(box[0,0]-1, box[0,1]-1),(box[0,2]-1, box[0,3]-1),(255,0,0),3) # -1 because annotation file assumes that top-left pixel of image is at [1,1]            
            else:
                cv2.rectangle(img,(box[0,0]-1, box[0,1]-1),(box[0,2]-1, box[0,3]-1),(0,255,0),3) # -1 because annotation file assumes that top-left pixel of image is at [1,1]
    cv2.imshow(Img_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def cropOutObjs(Outpath,VOCOpts=VOCOpts,Data_id=0,train=True,showI=False,ignore_difficulty=True):
    # get image folders
    if train:
        Imgpath = VOCOpts['DataRoot_trainval']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['ImgSource']
        Annopath = VOCOpts['DataRoot_trainval']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['anno']
    else:
        Imgpath = VOCOpts['DataRoot_test']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['ImgSource']
        Annopath = VOCOpts['DataRoot_test']+'/'+VOCOpts['Data'][Data_id]+'/'+VOCOpts['anno']
    # check outpath
    print("check outpath")
    if not os.path.isdir(Outpath):
        os.makedirs(Outpath)
    # get list of images
    imgs = glob.glob(os.path.join(Imgpath,"*{}".format(VOCOpts['img_ext'])))
    print("There are {} images in source".format(len(imgs)))
    nobjects = []
    objectname = []
    for imgidx, img in enumerate(imgs):
        print("Processing [{}/{}:]".format(imgidx,len(imgs)))
        Img_name = img.split("/")[-1]
        Imgannopath = os.path.join(Annopath,Img_name.split('.')[0]+'.xml')
        Annotated_Info = minidom.parse(Imgannopath)

        if Annotated_Info.getElementsByTagName('filename')[0].childNodes[0].data != Img_name:
            print("Accessed wrong annotation file")
            return 1
        load_objects = xml2dict(Imgannopath)
        I = cv2.imread(img)
        if showI:
            # show image
            if len(load_objects['name'])>0:
                for i in range(len(load_objects['bndbox'])):
                    box = load_objects['bndbox'][i]
                    print(load_objects['difficult'][i])
                    print(type(load_objects['difficult'][i]))
                    if load_objects['difficult'][i] == 1:
                        cv2.rectangle(I,(box[0,0]-1, box[0,1]-1),(box[0,2]-1, box[0,3]-1),(255,0,0),3) # -1 because annotation file assumes that top-left pixel of image is at [1,1]            
                    else:
                        cv2.rectangle(I,(box[0,0]-1, box[0,1]-1),(box[0,2]-1, box[0,3]-1),(0,255,0),3) # -1 because annotation file assumes that top-left pixel of image is at [1,1]
            cv2.imshow(Img_name,I)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # crop and save
        if len(load_objects['name'])>0:
            for i in range(len(load_objects['bndbox'])):
                if load_objects['difficult'][i] == 1 and ignore_difficulty:
                    continue
                outfolder = Outpath+"/"+load_objects['name'][i]
                if not os.path.isdir(outfolder):
                    os.makedirs(outfolder)
                # else:
                    # raise Exception("files will be overwrited.")
                if not load_objects['name'][i] in objectname:
                        objectname.append(load_objects['name'][i])
                        nobjects.append(0)

                objindex = objectname.index(load_objects['name'][i])
                no = nobjects[objindex]
                box = load_objects['bndbox'][i]
                cropI = np.copy(I[box[0,1]:box[0,3],box[0,0]:box[0,2]])
                cv2.imwrite(os.path.join(outfolder,str(no)+".jpg"),cropI)
                nobjects[objindex] += 1
        # if (imgidx+1)%5 == 0:
        #     break
    return 0

# if __name__ == '__main__':
#     ## you may need to check VOC directory
#     # newVOCOpts = copy.deepcopy(VOCOpts)
#     # newVOCOpts['DataRoot_trainval'] = 'fsfsdf'
#     # newVOCOpts['DataRoot_test'] = 'fsfsdf'
#     cropOutObjs("../data/VOCObjs")

