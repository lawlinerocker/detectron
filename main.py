import torch,matplotlib.pyplot as plt
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog,     DatasetCatalog
from detectron2.engine import DefaultTrainer

##
from detectron2.data import build_detection_train_loader
from detectron2.engine import HookBase
import detectron2.utils.comm as comm



from detectron2.structures import BoxMode
def main():
    
    class ValidationLoss(HookBase):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg.clone()
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
            self._loader = iter(build_detection_train_loader(self.cfg))
            
        def after_step(self):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)
                
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                    comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                    **loss_dict_reduced)
    def get_object_dicts(img_dir):
        json_file = os.path.join(img_dir,"via_region_data.json")
        #json_file = os.path.join(img_dir)
        with open(json_file) as f:
            imgs_anns = json.load(f)
    
        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
        
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            annos = v["regions"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts


    for d in ["train", "val"]:
        DatasetCatalog.register("reciept_" + d, lambda d=d: get_object_dicts("reciept/" + d))
        MetadataCatalog.get("reciept_" + d).set(thing_classes=["reciept"])
    object_metadata = MetadataCatalog.get("reciept_train")

    dataset_dicts = get_object_dicts('reciept/train')
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=object_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10,10),dpi=300)
        # plt.imshow(out.get_image()[:, :, ::-1])
        # plt.show()
        # cv2.imshow(out.get_image()[:, :, ::-1])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.VAL = ("reciept_train",)
    cfg.DATASETS.TRAIN = ("reciept_train",)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    #cfg.MODEL.DEVICE = "cpu"
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    val_loss=ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
  

    dataset_dicts = get_object_dicts("reciept/val")
    for d in random.sample(dataset_dicts, 5):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=object_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # plt.figure(figsize=(10,10),dpi=300)
        # plt.imshow(out.get_image()[:, :, ::-1])
        # plt.show()


if __name__=="__main__":
    main()


#train_path="reciept/train/via_region_data.json"



