

import pandas as pd
from PIL import Image
import pylab
import time as time
import json
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from .spine import dataset_info

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']

        else:
            cats = self.dataset['categories']
            # print(' ')
            # print('keypoints的cat就只有人1种')
            # print(cats)
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
            # print(cats)
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            # 根据imgIds找到所有的ann
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            # 通过各类条件如catIds对anns进行筛选
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:

            ids = [ann['id'] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        # 画点之间的连接线
                        plt.plot(x[sk], y[sk], linewidth=1, color='#9EFF00')
                # 画点
                for i, (xi, yi, zi) in enumerate(zip(x, y, v)):
                    if zi > 0:
                        kpt_id = dataset_info['keypoint_info'][i]['id']
                        color = [c / 255.0 for c in dataset_info['keypoint_info'][i]['color']]
                        # color = '#' + self.rgb2hex((color[0], color[1], color[2])).replace('0x','').upper()
                        plt.text(xi, yi, str(kpt_id), fontsize=4, color=[0, 0, 0])
                        plt.plot(xi, yi, 'o', markersize=2, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.3)

    def showAnnsLines(self, anns, tan_theta, cobb):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0

        ax = plt.gca()
        ax.set_autoscale_on(False)

        for ann in anns:
            # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        # 画点之间的连接线
                        plt.plot(x[sk], y[sk], linewidth=1, color='#9EFF00')

                plt.text(300, 300, cobb, fontdict={'family': 'serif', 'size': 16, 'color': '#C00000'}, ha='center',
                         va='center')


                # 找到最大值及其索引
                max_slope = np.max(tan_theta)
                max_index = np.argmax(tan_theta)

                min_slope = np.min(tan_theta)
                min_index = np.argmin(tan_theta)



                # 画点
                for i, (xi, yi, zi) in enumerate(zip(x, y, v)):
                    if zi > 0:
                        kpt_id = dataset_info['keypoint_info'][i]['id']
                        color = [c / 255.0 for c in dataset_info['keypoint_info'][i]['color']]
                        # color = '#' + self.rgb2hex((color[0], color[1], color[2])).replace('0x','').upper()
                        plt.text(xi, yi, str(kpt_id), fontsize=4, color=[0, 0, 0])
                        plt.plot(xi, yi, 'o', markersize=2, markerfacecolor=color, markeredgecolor=color,
                                 markeredgewidth=0.3)

                        line_length = 160
                        if i == max_index:
                            slope = max_slope
                            delta_x = line_length / 2 / np.sqrt(1 + slope ** 2)  # 根据斜率和长度计算增量
                            x1 = xi - delta_x
                            y1 = yi - slope * delta_x
                            x2 = xi + delta_x
                            y2 = yi + slope * delta_x
                            plt.plot([x1, x2], [y1, y2], linewidth=0.5, color='red')

                        elif i == min_index:
                            slope = min_slope
                            delta_x = line_length / 2 / np.sqrt(1 + slope ** 2)
                            x1 = xi - delta_x
                            y1 = yi - slope * delta_x
                            x2 = xi + delta_x
                            y2 = yi + slope * delta_x
                            plt.plot([x1, x2], [y1, y2], linewidth=1, color='red')


                        #
                        # slope = tan_theta[i]
                        #
                        # line_length = 40
                        #
                        # # 计算斜率对应的增量
                        # delta_x = line_length / 2 / np.sqrt(1 + slope ** 2)  # 根据斜率和长度计算增量
                        #
                        # # 计算两个端点
                        # x1 = xi - delta_x
                        # y1 = yi - slope * delta_x
                        #
                        # x2 = xi + delta_x
                        # y2 = yi + slope * delta_x
                        #
                        # plt.plot([x1, x2], [y1, y2], linewidth=1, color='red')