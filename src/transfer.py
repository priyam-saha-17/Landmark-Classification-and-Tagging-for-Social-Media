import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50, dropout = 0.2):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    # HINT: loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    
    for param in model_transfer.parameters():
        param.requires_grad = False
        
    #print(model_transfer)
    
    if "resnet18" in model_name:
        # Add the linear layer at the end with the appropriate number of classes
        # 1. get numbers of features extracted by the backbone
        num_ftrs  = model_transfer.fc.in_features

        # 2. Create a new linear layer with the appropriate number of inputs and
        #    outputs
        model_transfer.fc  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(256, 56),
                nn.BatchNorm1d(56),
                nn.ReLU(),
            
                nn.Linear(56, n_classes)
            )
    
    
    elif "vgg16" in model_name:
        # Add the linear layer at the end with the appropriate number of classes
        # 1. get numbers of features extracted by the backbone
        num_ftrs  = 25088      # for vgg16, no. of input features to the classifier is 25088 (can be observed by printing the model)

        # 2. Create a new linear layer with the appropriate number of inputs and
        #    outputs
        model_transfer.classifier  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),              
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(256, 56),
                nn.BatchNorm1d(56),
                nn.ReLU(),
            
                nn.Linear(56, n_classes)
            )

        
        
    elif "efficientnet_b0" in model_name:
        # Add the linear layer at the end with the appropriate number of classes
        # 1. get numbers of features extracted by the backbone
        num_ftrs  = 1280      # for vgg16, no. of input features to the classifier is 25088 (can be observed by printing the model)

        # 2. Create a new linear layer with the appropriate number of inputs and
        #    outputs
        model_transfer.classifier  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),              
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(256, 56),
                nn.BatchNorm1d(56),
                nn.ReLU(),
            
                nn.Linear(56, n_classes)
            )

    elif "mobilenet_v2" in model_name:
        # Add the linear layer at the end with the appropriate number of classes
        # 1. get numbers of features extracted by the backbone
        num_ftrs  = 1280      # for vgg16, no. of input features to the classifier is 25088 (can be observed by printing the model)

        # 2. Create a new linear layer with the appropriate number of inputs and
        #    outputs
        model_transfer.classifier  = nn.Sequential(
                nn.Linear(num_ftrs, 1024),              
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            
                nn.Linear(256, 56),
                nn.BatchNorm1d(56),
                nn.ReLU(),
            
                nn.Linear(56, n_classes)
            )
        
        
    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
