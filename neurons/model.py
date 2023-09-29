from transformers import AutoModel, AutoModelForPreTraining, AutoConfig
from transformers import ConvNextFeatureExtractor
from transformers import ResNetConfig, ResNetForImageClassification
import torch

# Define the data preprocessor
def build_processor():
    return ConvNextFeatureExtractor(size=224,
                                    image_mean=[0.485, 0.456, 0.406],
                                    image_std=[0.229, 0.224, 0.225])

# Define the optimizer based on the model definition
def build_optimizer(model, lr=0.01):
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    return optimizer

# Define the model to train
def build_model():
    # Define the resnet18 model and model checkpoint path.
    resnet_config = ResNetConfig(
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        layer_type='basic',
        hidden_act='relu',
        downsample_in_first_stage=False,
        out_features=['stage4'],
        out_indices=[4],
        num_labels=10,
        label2id={'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9},
        id2label={0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    )
    model = ResNetForImageClassification(resnet_config)
    return model
