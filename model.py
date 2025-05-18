# 6. 모델 정의 (예시로 ResNet18 사용, Dropout 추가)
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # pretrained=True 대신 weights 매개변수를 사용
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),  # Batch Normalization 추가
            nn.Dropout(0.5),  # 드롭아웃 추가 (30% 확률로 뉴런을 비활성화)
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# DEVICE가 정의되어 있다고 가정하고 모델을 해당 장치에 할당
model = CustomResNet(num_classes=len(classes)).to(DEVICE)
