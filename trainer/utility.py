
from detectron2.utils.visualizer import  Visualizer,ColorMode
from detectron2.data import  DatasetCatalog,MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_train_cfg(config_file_path,checkpoint_url,train_dataset_name,test_dataset_name,
                  base_lr,num_iters,device,output_dir,num_workers,num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_filename=model_zoo.get_config_file(config_path=config_file_path))
    cfg.DATASETS.TRAIN = (train_dataset_name)
    cfg.DATASETS.TEST = (test_dataset_name)
    #get the model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    #set number of workers
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.BASE_LR = base_lr
    #maximum number of iterations
    cfg.SOLVER.MAX_ITER = num_iters
    #get number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    #set the device
    cfg.MODEL.DEVICE = device

    #set the output directory
    cfg.OUTPUT_DIR = output_dir
    return  cfg