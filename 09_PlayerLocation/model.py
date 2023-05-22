import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=False)
        # print(self.model)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.model.classifier = nn.Identity()

        # self.hitter_info_proj = nn.Sequential(
        #     nn.Linear(2, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        # )
        
        self.head1 = nn.Sequential(
            nn.Linear(1792, 1792),
            # nn.ReLU(),
            nn.Linear(1792, 1) # 434
        )
        self.head2 = nn.Sequential(
            nn.Linear(1792, 1792),
            # nn.ReLU(),
            nn.Linear(1792, 1) # 483
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.flatten(1)
        dx = self.head1(x)
        dy = self.head2(x)
        return dx, dy

if __name__ == "__main__":
    m = Model()