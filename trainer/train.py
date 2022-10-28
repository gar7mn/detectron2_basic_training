import os

from utility import *
import detectron2
import pickle
from detectron2.engine import  DefaultTrainer
from detectron2.data.datasets import register_coco_instances
#the config file paths can be gotten from the detectron2 model zoo on github
config_file_path = " " #path to yaml file eg: faster_rcnn_R_50
checkpoint_url = ""   #path to yaml file eg: faster_rcnn_R_50
device = "cuda"
#dataset name
train_dataset_name = "" #training dataset name in the quotes
test_dataset_name = "" #test dataset name in the quotes
#classes
num_classes = 1 #set to the number of different types of objects in your dataset
#set the image paths
train_im_path = r"" #path in quotes
test_im_path = r"" #path in quotes
#set the annotation paths
train_annotation_dir = r"" #path to json file where annotations are stored for the train dataset
test_annotation_dir = r"" #path to json file where annotations are stored for the train dataset
#cfg save directory
cfg_save_dir = r""
#set the output directory
output_dir = r"" # path to output in the  quotes
#set the number of workers
num_workers = 16 #make sure you have a good gpu if your going to use this many
#set the base lr
base_lr = 0.003
#set the nummber of iterations
num_iters = 2000
#register the datasets
register_coco_instances(name=train_dataset_name,metadata={},
                        json_file=train_annotation_dir,image_root=train_im_path) #metadata may be how you add the names to object predictions

register_coco_instances(name=test_dataset_name,metadata={},
                        json_file=test_annotation_dir,image_root=test_im_path)
def main():
    cfg = get_train_cfg(config_file_path=config_file_path,checkpoint_url=checkpoint_url,train_dataset_name=train_dataset_name,
                        test_dataset_name=test_dataset_name,base_lr=base_lr,num_iters=num_iters,device=device,
                        output_dir=output_dir,num_workers=num_workers,num_classes=num_classes)
    with open(cfg_save_dir,"wb") as sd:
        pickle.dump(cfg,sd,protocol=pickle.HIGHEST_PROTOCOL)
        #make output directories if they dont already exist
        os.makedirs(output_dir,exist_ok=True)
        #setup trainer and assign the config
        trainer = DefaultTrainer(cfg)
        #choose whether or not to resume the model
        trainer.resume_or_load(resume=False)
        trainer.train()

