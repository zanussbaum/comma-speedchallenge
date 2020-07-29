import numpy as np
import re
import cv2 as cv
import tensorflow as tf
import glob
import pickle
import matplotlib.pyplot as plt

from random import shuffle


from skimage.util import random_noise as noise
from tqdm import tqdm

def read(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string,channels=3)
    return image

"""
Applies Gaussian Noise to image
"""
def random_noise(image):
    return noise(image.numpy()) 


"""
Random Hue Augmentation
"""
def random_hue(image):
    return tf.image.random_hue(image,.05) 


"""
Applies the transformations and saves the images
"""
def apply_transformations(image_path, speed):
    orig = read(image_path)
    augmented = [orig]
    if speed > 25.5:
        augmented.extend([random_noise(orig) for i in range(3)])

    return augmented


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def run():
    images = natural_sort(glob.glob('../frames/flow/*.jpg'))
    shuffle(images)
    assert len(images) == 20399, f"Number of frames {len(images)}"
    with open('../frames/mean_speeds', 'rb') as f:
        speeds = pickle.load(f)

    augmented_speeds = []
    augmented_images = []

    validation_images = []
    validation_speeds = []
    for i, img in enumerate(tqdm(images, desc="Augmenting")):
        augmented = apply_transformations(img, speeds[i])
        if (i+1) % 4 == 0:
            validation_images.extend(augmented)
            validation_speeds.extend([speeds[i]]*len(augmented))
        else:
            augmented_images.extend(augmented)
            augmented_speeds.extend([speeds[i]]*len(augmented))

    assert len(augmented_speeds) == len(augmented_images)
    assert len(validation_speeds) == len(validation_images)

    for i, img in enumerate(tqdm(augmented_images, desc="Saving")):
        tf.keras.preprocessing.image.save_img(f'../frames/augmented/frame_{i}.jpg', img)

    with open('../frames/augmented_speeds', 'wb') as f:
        pickle.dump(augmented_speeds, f)

    for i, img in enumerate(tqdm(validation_images, desc="Saving")):
        tf.keras.preprocessing.image.save_img(f'../frames/validation/frame_{i}.jpg', img)

    with open('../frames/validation_speeds', 'wb') as f:
        pickle.dump(validation_speeds, f)

if __name__ == '__main__':
    run()
    #images = sorted(glob.glob('../frames/augmented/*.jpg'))
    #with open('../frames/augmented_speeds', 'rb') as f:
    #    speeds = pickle.load(f)
    #assert len(images) == len(speeds)
    #
    #num_images = len(images)
    #num_added = 1
    #
    #for i, image in enumerate(tqdm(images, desc="Balancing")):
    #    if speeds[i] >= 26:
    #        orig = read(image)
    #        for j in range(2):
    #            tf.keras.preprocessing.image.save_img(
    #            f'../frames/augmented/frame_{num_images+num_added}.jpg',
    #                        random_noise(orig))
    #            num_added += 1
    #            speeds.append(speeds[i])

    #with open('../frames/augmented_speeds', 'wb') as f:
    #    pickle.dump(speeds, f)
    #assert len(speeds) == (len(images) + num_added - 1), f"Num speeds {len(speeds)}, Num Frames {len(images) + num_added - 1}"
