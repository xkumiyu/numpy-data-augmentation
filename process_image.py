import argparse
import os

import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate


def read_image(imagefile, dtype=np.float32):
    image = np.array(Image.open(imagefile), dtype=dtype)
    return image


def save_image(image, imagefile, data_format='channel_last'):
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save(imagefile)


def concat_images(images, rows, cols):
    _, h, w, _ = images.shape
    images = images.reshape((rows, cols, h, w, 3))
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape((rows * h, cols * w, 3))
    return images


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def subtract(image):
    image = image / 255
    return image


def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image


def center_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image


def scale_augmentation(image, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, crop_size)
    return image


def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image


def cutout(image_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    image[top:bottom, left:right, :].fill(mask_value)
    return image


def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random'):
    image = np.copy(image_origin)
    if np.random.rand() > p:
        return image
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Data Augmentation')
    parser.add_argument('infile')
    parser.add_argument('--outdir', '-o', default='./')
    parser.add_argument('--n_loop', '-n', type=int, default=1)
    parser.add_argument('--concat', '-c', action='store_true')
    args = parser.parse_args()

    processing_list = ['random_crop', 'horizontal_flip', 'vertical_flip',
                       'scale_augmentation', 'random_rotation', 'cutout',
                       'random_erasing']

    inimg = read_image(args.infile)
    inimg224 = resize(inimg, 224)
    if args.concat:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        def save_concat_image(outimg_name, func, *func_args):
            images = []
            for i in range(args.n_loop):
                images.append(func(*func_args))
            x = int(np.sqrt(args.n_loop))
            outimg = concat_images(np.array(images), x, x)
            save_image(outimg, os.path.join(args.outdir, outimg_name))

        save_concat_image('random_crop.jpg', random_crop, resize(inimg, 400), 224)
        save_concat_image('horizontal_flip.jpg', horizontal_flip, inimg224)
        save_concat_image('vertical_flip.jpg', vertical_flip, inimg224)
        save_concat_image('scale_augmentation.jpg', scale_augmentation, inimg, (256, 480), 224)
        save_concat_image('random_rotation.jpg', random_rotation, inimg224)
        save_concat_image('cutout.jpg', cutout, inimg224, inimg224.shape[0] // 2)
        save_concat_image('random_erasing.jpg', random_erasing, inimg224)

    else:
        for processing_name in processing_list:
            outdir = os.path.join(args.outdir, processing_name)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        for i in range(args.n_loop):
            save_image(
                random_crop(resize(inimg, 256), 224),
                os.path.join(args.outdir, 'random_crop', '{}.jpg'.format(i)))
            save_image(
                horizontal_flip(inimg224),
                os.path.join(args.outdir, 'horizontal_flip', '{}.jpg'.format(i)))
            save_image(
                vertical_flip(inimg224),
                os.path.join(args.outdir, 'vertical_flip', '{}.jpg'.format(i)))
            save_image(
                scale_augmentation(inimg, (256, 480), 224),
                os.path.join(args.outdir, 'scale_augmentation', '{}.jpg'.format(i)))
            save_image(
                random_rotation(inimg224),
                os.path.join(args.outdir, 'random_rotation', '{}.jpg'.format(i)))
            save_image(
                cutout(inimg224, inimg224.shape[0] // 2),
                os.path.join(args.outdir, 'cutout', '{}.jpg'.format(i)))
            save_image(
                random_erasing(inimg224),
                os.path.join(args.outdir, 'random_erasing', '{}.jpg'.format(i)))
