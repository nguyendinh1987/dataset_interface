import urllib.request
import os, sys

class imagenet_API_interface(object):
    def __init__(self):
        self.wordnetIDmap_url="http://www.image-net.org/archive/words.txt"
        self.imageurl_list_prefix="http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid="
        self.wordidmap = None
        self.cache = {}
    def download_wordnetmap(self,saveto="/opt/tmp/imagenet_wordnetmap.txt",update_wordidmap=True):
        self.cache["path_to_wordnetmap"] = saveto
        file_from_url = urllib.request.urlopen(self.wordnetIDmap_url)
        lines = []
        for l in file_from_url:
            de_line = l.decode("utf-8")
            # for cchr in de_line:
            #     print("{}-{}".format(cchr,ord(cchr)))
            # break
            # print(de_line[:-2])
            # print("-------")
            # lns = de_line[:-1].split(chr(9))
            # for ln in lns:
            #     print(ln)
            # print("---------")
            lines.append(de_line)
        
        if update_wordidmap:
            self.wordidmap = {}
            for ln in lines:
                sln = ln.split(chr(9))
                self.wordidmap[sln[0]]=sln[1]
        fi = open(saveto,"w")
        fi.writelines(lines)
        fi.close()
        return True
    def get_wordid(self,word_node):
        wordids = []
        for k,v in self.wordidmap.items():
            if word_node in v:
                wordids.append(k)
        print("there are {} wordids found".format(len(wordids)))
        return wordids
    
    def download_images(self,word_node,output="/opt/Data/imagenet/images/tmp"):
        wordids = self.get_wordid(word_node)
        n_loaded_image = 0
        n_cannot_load_image = 0
        for wid in wordids:
            urllink = self.imageurl_list_prefix+wid
            print("loading from "+urllink)
            file_from_url = urllib.request.urlopen(urllink)
            lines = []
            for l in file_from_url:
                de_line = l.decode("utf-8")
                if len(de_line)<10:
                    continue
                # print(de_line)
                # print(de_line[:-2])
                lines.append(de_line)

            txtpath = os.path.join(output,"image_info.txt")
            print("Recording image list to "+txtpath)
            fi = open(txtpath,"w")
            fi.writelines(lines)
            fi.close()
            print("Downloading image .....")
            for l in lines:
                sln = l.split(" ")
                # print(sln[0])
                # print(sln[1])
                imgname = sln[0]+"."+sln[1].split(".")[-1]
                imgpath = os.path.join(output,imgname)

                can_download = True
                try: 
                    urllib.request.urlopen(sln[1])
                except urllib.error.URLError as e:
                    can_download = False
                    n_cannot_load_image += 1
                    print("Cannot open {}".format(sln[1]))
                    print(e)
                if can_download:
                    n_loaded_image += 1
                    urllib.request.urlretrieve(sln[1],imgpath)
                if n_loaded_image%20==0:
                    print("Downloaded: {}; False: {}".format(n_loaded_image,n_cannot_load_image))

        return True

    def print_wordidmap(self):
        for k,v in self.wordidmap.items():
            print("{}: {}".format(k,v))