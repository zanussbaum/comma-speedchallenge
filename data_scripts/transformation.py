import skimage.transform
import cv2 as cv
import numpy as np

"""
Flips an image horizontally
"""
def flip(image):
    return cv.flip(image, 1)

"""
Scales an image by a random factor
"""
def scale(image):
    scales = [.5, .75, 1.5, 2]

    choice = np.random.choice(scales)

    transformed = cv.resize(image, None, fx=choice, fy=choice, interpolation=cv.INTER_LINEAR if choice > 1 else cv.INTER_AREA)


    return cv.resize(transformed, image.shape[:-1][::-1])
"""
Randomly crops and resizes an image
"""
def random_crop(image):
    height, width, _ = image.shape
    crop_width = int(width * .75)
    crop_height = int(height * .75)

    max_x = height - crop_height
    max_y = width - crop_width

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    cropped = image[y: y+crop_height, x: x+crop_width]

    resized = cv.resize(cropped, (width, height))

    return resized



if __name__ == '__main__':
    original_image = 'frames/flow/flow_0.jpg'
    image = cv.imread(original_image)

    cv.imshow('original', image)

    transformed = random_crop(image)

    cv.imshow('transformed', transformed)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
             break
    cv.destroyAllWindows()
