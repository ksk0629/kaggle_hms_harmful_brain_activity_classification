from dataclasses import dataclass

@dataclass
class Config:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    pretrained_model = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [400, 300]  # Input image size
    epochs = 13 # Training epochs
    batch_size = 64  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v:k for k, v in label2name.items()}
    
    def __init__(self,
                 seed: int=42,
                 pretrained_model: str="efficientnetv2_b2_imagenet",
                 image_size: tuple[int, int]=[400, 300],
                 epochs: int=13,
                 lr_mode: str="cos"):
        this.seed = seed
        this.pretrained_model = pretrained_model
        this.image_size = image_size
        this.epochs = epochs
        this.batch_size = batch_size
        this.lr_mode = lr_mode