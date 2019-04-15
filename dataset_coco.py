from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import os
import skimage.io
import logging
from distutils.version import LooseVersion


class Dataset(object):

    def __init__(self):
        self._image_ids = []
        self.image_info = []
        self.class_info = [] # 无背景，故只有80个classes
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # 是否class已经存在
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source":source,    # 数据集名字
            "id":class_id,      # 类别id
            "name":class_name,  # 类别注释即名字
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self):
        # 准备数据
        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images) # 内部id
        # 将class source和class id映射到内部id
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # set数据集
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                if source == info['source']:
                    self.source_class_ids[source].append(i)


    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        # 返回内部id
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id] # 因为内部id直接按序分配，直接索引就得到了
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_num_set_box_gt(self, image_id):
        raise NotImplementedError

class CocoDataset(Dataset):


    def load_coco(self, dataset_dir, subset, year="2014"):

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        class_ids = sorted(coco.getCatIds())

        image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]['name'])

        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        # print("dd")

    def load_num_set_box_gt(self, image_id):
        """
        :return: ndarray num_gt (num_classes=80, )
                 每个类别的数量
        """
        num_gt = np.zeros(80)
        set_gt = np.zeros(80)
        bbox_gt = []
        class_ids = []

        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_num_set_gt(image_id)


        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:

            # crowd,skip
            if annotation['iscrowd']:
                continue

            # too small obj that less than 1pixel area,skip
            m = self.annToMask(annotation, image_info["height"], image_info["width"])
            if m.max() < 1:
                continue

            # 获取内部classid通过source和实际classid
            # 内部classid 应该属于[0,80) 并且是顺序一定

            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

            #
            num_gt[class_id] += 1
            x, y, w, h = annotation['bbox']
            bbox_gt.append([x,y,x + w,y+h])
            # bbox_gt.append([x,y,w,h])
            class_ids.append(class_id)

        # S
        set_gt[np.where(np.logical_and(num_gt >= 0, num_gt <= 4))] = 1
        # Shat
        set_gt[np.where(num_gt >= 5)] = 2
        # A 不贡献loss置为-1 选择百分之10贡献loss
        idx = np.where(num_gt == 0)
        set_gt[idx] = -1
        choices = np.ceil(idx[0].shape[0] * 0.1).astype(np.int32)
        ind = np.random.choice(idx[0], choices, replace=False)
        set_gt[list(ind)] = 0

        class_ids = np.array(class_ids, dtype=np.int32)
        bbox_gt = np.stack(bbox_gt, axis=0).astype(np.float32)

        return num_gt.astype(np.int32), set_gt.astype(np.int32), bbox_gt, class_ids
        # num_gt:(80,) 类别的数量
        # set_gt:(80,) -1不贡献loss 0A 1S 2Shat
        # bbox_gt:(?,4) x1,y1,x2,y2   PS:x2,y2 不在box里面
        # class_ids:(?,) 取值为[0,80)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# coco's mean heights is :  483.5902057185654
# coco's mean widths is :  483.5902057185654
# coco's max heights is :  640
# coco's max widths is :  640
# coco's min heights is :  51
# coco's min widths is :  51  这么小！
# coco's mean aspect is : 0.885000054379151 height/width 480/640=0.75
# coco's max aspect is : 2.6016260162601625
# coco's min aspect is : 0.2484375


MAX_INSTANCE = 100
MEAN_IMAGE = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
STD_IMAGE = np.array([0.229, 0.224, 0.225])
IMG_HEIGHT = 800
IMG_WIDTH = 800

def mold_image(img):
    img = img.astype(np.float32) / 255.0
    img = img - MEAN_IMAGE
    # img = img / STD_IMAGE
    return img

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def load_gt(dataset, image_id, augmentation=True):
    """
    :param dataset:
    :param image_id:
    :param augmentation:
    :return:
        image:[height,width,3]
        num_gt:[80,]
        set_gt:[80,]
        https://github.com/cocodataset/cocoapi/issues/34
        bbox_gt:[obj_nums,4] x1,y1,x2,y2 0-index x2 y2在box外部
        class_ids:[obj_nums]
    """
    image = dataset.load_image(image_id)
    num_gt, set_gt, bbox_gt,class_ids = dataset.load_num_set_box_gt(image_id)
    if augmentation:
        import imgaug as ia
        import imgaug.augmenters as iaa
        # imgaug变换box格式与coco不一样
        # coco:800高600宽的图片右下角的点坐标为(800,600)在box外部 左闭右开
        # https://github.com/cocodataset/cocoapi/issues/34
        # TODO always test these code
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1,y1,x2-1.0,y2-1.0) for x1,y1,x2,y2 in bbox_gt],
                                      shape=image.shape[:2])
        seq = iaa.Sequential([iaa.Fliplr(0.5)])
        seq_det = seq.to_deterministic()
        image = seq_det.augment_image(image)
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        # bboxes 回到coco box的形式
        # bbox_gt = [np.array([bbox.x1,bbox.y1,bbox.x2+1.0,bbox.y2+1.0]) for bbox in bbs_aug.bounding_boxes]
        # bbox_gt = np.array(bbox_gt)
        bbox_gt = bbs_aug.to_xyxy_array(np.float32)
        bbox_gt[:, 2] += 1.0
        bbox_gt[:, 3] += 1.0

    return image, num_gt, set_gt, bbox_gt, class_ids


def data_generator(dataset, batch_size, shuffle=True):
    """
    通过dataset进行coco数据的读取
    batch_size, shuffle, augmentation
    :return: yield
    因为每个图像含有instance数量不同，故进行pad
    """
    b = 0 # batch item index
    image_index = -1
    error_count = 0
    image_ids = np.copy(dataset.image_ids)

    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]
            # (H,W,3)
            # (80,)
            # (80,)
            # (?,4)
            # (?,)
            image, num_gt, set_gt, bbox_gt, class_ids = load_gt(dataset,
                                                                image_id)

            # precess image
            image = resize(image, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)

            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_num_gt = np.zeros((batch_size,) + (80,), dtype=np.int32)
                batch_set_gt = np.zeros((batch_size,) + (80,), dtype=np.int32)
                batch_bbox_gt = np.zeros(
                    (batch_size, MAX_INSTANCE, 4), dtype=np.float32)
                batch_class_ids = np.zeros(
                    (batch_size, MAX_INSTANCE), dtype=np.int32)

            # 若超过100个instances没写
            # if bbox_gt.shape[0] > MAX_INSTANCE:
            #     ids = np.random.choice(
            #         np.arange(bbox_gt.shape[0]), MAX_INSTANCE, replace=False)
            batch_images[b] = mold_image(image.astype(np.float32))
            batch_num_gt[b] = num_gt
            batch_set_gt[b] = set_gt
            batch_bbox_gt[b, :bbox_gt.shape[0]] = bbox_gt
            batch_class_ids[b, :class_ids.shape[0]] = class_ids

            b += 1
            if b >= batch_size:
                inputs = [batch_images, batch_num_gt, batch_set_gt,
                          batch_bbox_gt, batch_class_ids]
                outputs = []
                # TODO test this code
                yield inputs, outputs
                b = 0


        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise



