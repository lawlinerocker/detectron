import json
import imgaug as ia
#import imgaug.augmenters as iaa
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmenters.meta import Sometimes
import imageio
# import cv2
import matplotlib.pyplot as plt
import os
json_data = {}
dic = {}
b=1
with open("via_region_data.json", "r") as f:
    content = json.load(f)
    for polygon in content:  # there are probably multiple polygons in the file? so this "for" must be somehow derived from the file content
        xx = content[polygon]["regions"]["0"]["shape_attributes"]["all_points_x"]  # "..." needs to be filled with some keys
        yy = content[polygon]["regions"]["0"]["shape_attributes"]["all_points_y"]
        keypoints = [ia.Keypoint(x=x, y=y) for x, y in zip(xx, yy)]
        ia.seed(1)
        
        image = imageio.imread(content[polygon]["filename"])


        for m in range(10):
            kps = KeypointsOnImage(keypoints,shape=image.shape)
            

            seq = iaa.Sequential([
                iaa.Sometimes(0.3, iaa.Affine(rotate=10,scale=(0.5, 0.7))),
                iaa.Sometimes(0.3, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(0.2, iaa.JpegCompression(compression=(70, 99))),
                iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Sometimes(0.2, iaa.MotionBlur(k=15, angle=[-45, 45])),
                iaa.Sometimes(0.2, iaa.MultiplyHue((0.5, 1.5))),
                iaa.Sometimes(0.2, iaa.MultiplySaturation((0.5, 1.5))),
                iaa.Sometimes(0.34, iaa.MultiplyHueAndSaturation((0.5, 1.5),
                                                                per_channel=True)),
                iaa.Sometimes(0.34, iaa.Grayscale(alpha=(0.0, 1.0))),
                iaa.Sometimes(0.2, iaa.ChangeColorTemperature((1100, 10000))),
                iaa.Sometimes(0.1, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(0.2, iaa.SigmoidContrast(gain=(3, 10),
                                                    cutoff=(0.4, 0.6))),
                iaa.Sometimes(0.1, iaa.CLAHE()),
                iaa.Sometimes(0.1, iaa.HistogramEqualization()),
                iaa.Sometimes(0.2, iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
                iaa.Sometimes(0.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                iaa.Sometimes(0.5, iaa.Fliplr(0.5))

            ],random_order=True) 


            image_aug, kps_aug = seq(image=image, keypoints=kps)
            allx=[]
            ally=[]
            for i in range(len(kps.keypoints)):
                before = kps.keypoints[i]
                after = kps_aug.keypoints[i]
                allx.append(int(after.x))
                ally.append(int(after.y))
            #    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            #     i, before.x, before.y, after.x, after.y)
            #    )ผ
            imagefile = "folder/genpic"+str(b)+".jpg"
            # imageio.imwrite("folder/genpic"+str(b)+".jpg", image_aug)
            imageio.imwrite(imagefile, image_aug)
            size_bytes = os.path.getsize(imagefile)
            print(size_bytes)
            dic = {"genpic"+str(b)+".jpg"+str(size_bytes):
                {"filename":"genpic"+str(b)+".jpg",
                "size":str(size_bytes),
                "file_attributes": {},
                "regions": {
                    "0":{
                        "shape_attributes":{
                            "name": "polygon",
                            "all_points_x":allx,
                            "all_points_y":ally
                        },
                        "region_attributes":{}
                    }
                }}}
            json_data.update(dic)
            b=b+1
            
with open('folder/output.json', 'w') as jsonFile:
    json.dump(json_data, jsonFile)
    jsonFile.close()