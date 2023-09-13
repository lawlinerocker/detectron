import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage

ia.seed(1)
image=imageio.imread("295fe953ea5ccc524c913e30a770900f.jpg")

images = np.array(
    [image for _ in range(32)], dtype=np.uint8)  # 32 means creat 32 enhanced images using following methods.
