import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from albumentations.core.transforms_interface import BasicTransform
import os

def is_running_on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def is_running_on_colab():
    return "COLAB_GPU" in os.environ

KAGGLE_STR = "/kaggle/working/Multi-Pix2Pix/" if is_running_on_kaggle() else ""
COLAB_STR = "/content/Multi-Pix2Pix/" if is_running_on_colab() else ""

PREFIX_STR = KAGGLE_STR + COLAB_STR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = PREFIX_STR + "image_dataset/landslide-self/Train"
VAL_DIR = PREFIX_STR + "image_dataset/landslide-self/Test"

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
VAL_BATCH_SIZE = 5
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = PREFIX_STR + "disc_LSM_v1.pth.tar"
CHECKPOINT_GEN = PREFIX_STR + "gen_LSM_v1.pth.tar"

RESIZE = None
if RESIZE != None:
    IMAGE_RESIZED = RESIZE * IMAGE_SIZE
else:
    IMAGE_RESIZED = IMAGE_SIZE
 
# Stlye loss
ALPHA = 1
BETA = 0.01

#################################################
IN_CHANNELS = 3
OUT_CHANNELS = 3
INTER_IMAGES = 4
BINARIZE = False
GRAYSCALE_ALL = False
#################################################

class ThresholdTransform(BasicTransform):
    def __init__(self, thr_255, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.thr = thr_255 / 255.0  # Convert threshold to [0, 1] range

    def apply(self, img, **params):
        # Binarize image based on the threshold
        return (img > self.thr).astype(img.dtype)

    @property
    def targets(self):
        # Specify that this transform applies to the 'mask'
        return {"image": self.apply, "mask": self.apply}

transform_only_input = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_inter = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask_binarize = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ThresholdTransform(thr_255=100),
        ToTensorV2(),
    ]
)