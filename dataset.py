import glob
import numpy as np
import skimage.io
import skimage.transform
import xml.etree.ElementTree as ET
from distutils.version import LooseVersion




image_files = "./data/JPEGImages/"
annotations_files = "./data/Annotations/"
pkl_files = "./data/pkl/"
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

class_name2idx = {"aeroplane":0,"bicycle":1,"bird":2,"boat":3,"bottle":4,"bus":5,"car":6,"cat":7,
                 "chair":8,"cow":9,"diningtable":10,"dog":11,"horse":12,"motorbike":13,"person":14,
                 "pottedplant":15,"sheep":16,"sofa":17,"train":18,"tvmonitor":19}
class_idx2name = {item:k for k,item in class_name2idx.items()}
class_name = [x for x in class_name2idx.keys()]
class_num = 20


def load_image(img_file):
    image = skimage.io.imread(img_file)
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def mold_image(images):
    return images.astype(np.float32) - MEAN_PIXEL

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


def data_generator(batch_size, augmentation=None, shuffle=True):
    b = 0
    ix = 0
    img_files = glob.glob(image_files + '*.jpg')


    while True:
        if shuffle and ix == 0:
            np.random.shuffle(img_files)
        image_path = img_files[ix]
        image_basename = image_path.split('\\')[1]
        npz_path = pkl_files + image_basename.split('.')[0] + '.npy'
        img = load_image(image_path)
        out = np.load(npz_path)

        num_gt = out[0, :]
        set_gt = out[1, :]

        img = resize(img, (512,512), preserve_range=True)

        if b == 0:
            batch_images = np.zeros((batch_size,) + img.shape, dtype=np.float32)
            batch_num_gt = np.zeros((batch_size,) + (20,), dtype=np.int32)
            batch_set_gt = np.zeros((batch_size,) + (20,), dtype=np.int32)

        batch_images[b] = mold_image(img)
        batch_num_gt[b] = num_gt
        batch_set_gt[b] = set_gt
        b += 1
        ix = (ix + 1) % len(img_files)
        if b >= batch_size:
            inputs = [batch_images, batch_num_gt, batch_set_gt]
            outputs = []
            yield inputs, outputs
            b = 0



        # print('dd')

def transform_annotation2pkl():
    """
    :return: num_gt:(C)
             set_gt:(C)
    """
    img_files = glob.glob(image_files + '*.jpg')

    for i in range(len(img_files)):
        image_path = img_files[i]
        image_basename = image_path.split('\\')[1]
        xml_path = annotations_files + image_basename.split('.')[0] + '.xml'
        pkl_file = pkl_files + image_basename.split('.')[0]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        num_gt = np.zeros(20)
        set_gt = np.zeros(20)
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            obj_idx = class_name2idx[obj_name]
            num_gt[obj_idx] += 1

        # S
        set_gt[np.where(np.logical_and(num_gt >= 0, num_gt <= 4))] = 1
        # Shat
        set_gt[np.where(num_gt >= 5)] = 2
        # A
        idx = np.where(num_gt == 0)
        set_gt[idx] = -1
        ind = np.random.choice(idx[0], np.ceil(idx[0].shape[0] * 0.1).astype(np.int32))
        set_gt[list(ind)] = 0


        out = np.stack((num_gt, set_gt), axis=0).astype(np.int32)
        np.save(pkl_file, out)
        # print('dd')



if __name__ == '__main__':
    transform_annotation2pkl()