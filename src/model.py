## Author: chenlin
## Date: 2026-11-5

# 使用ResNet518 CNN Backbone
import torch
import torch.nn as nn
import torchvision.models as models


class EyeStateClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()

        # 1. 加载 ResNet18 预训练模型
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )
        backbone = models.resnet18(weights=weights)

        # 2. 将第一层卷积修改为灰度图输入
        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 使用原 RGB 预训练权重初始化灰度卷积
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        backbone.conv1 = new_conv

        # 3. 去掉原来的 ImageNet 分类层
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # 4. 图像和辅助特征维度
        image_dim = 512
        metadata_dim = 9

        # 5. 融合分类头
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image, metadata):
        # image: [B, 1, 74, 74]
        # metadata: [B, 9]

        # 提取图像特征
        image_features = self.backbone(image)

        # [B, 512, 1, 1] -> [B, 512]
        image_features = torch.flatten(
            image_features, start_dim=1
        )

        # 拼接图像特征和辅助特征
        # [B, 512] + [B, 9] -> [B, 521]
        features = torch.cat(
            [image_features, metadata], dim=1
        )

        # 输出二分类 logit
        logits = self.classifier(features)

        return logits