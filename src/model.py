import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        
        self.conv_block = nn.Sequential(
            
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 16),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=dropout),
            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 32),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=dropout),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=dropout),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 128),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=dropout),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(num_features = 256),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=dropout)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((7,7))
        
        self.mlp_block = nn.Sequential(
            nn.Linear(7*7*256, 1024),
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
            
            nn.Linear(56, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x = self.conv_block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flattening the feature map
        x = self.mlp_block(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
