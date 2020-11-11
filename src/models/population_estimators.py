from torch import nn


class PopulationEstimator(nn.Module):
    def __init__(self, backbone, input_channel, num_classes=4) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_channel = input_channel
        self.num_classes = num_classes
        # 사람이 한명인지 두명인지 세명 이상인지
        self.classifier = nn.Sequential(nn.Linear(input_channel, num_classes),
                                        )  # 0,1,2,3
        # 사람이 있는지 유무만
        self.classifier2 = nn.Sequential(nn.Linear(input_channel, 1), )

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.classifier(x)  # labels, CrossEntropyLoss
        x2 = self.classifier2(x)  # existence, BCEWithLogitsLoss
        return x1, x2