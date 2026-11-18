## Author: chenlin
## Date: 2026-11-5

# 使用ResNet34 CNN Backbone
import torch
import torch.nn as nn
import torchvision.models as models


def adapt_state_dict(state_dict: dict) -> dict:
    """兼容旧版 checkpoint 的 backbone.N 键名,映射到当前 stem/layerN 结构。"""
    mapping = {
        "backbone.0.": "stem.0.",
        "backbone.1.": "stem.1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    }
    out = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            for old_prefix, new_prefix in mapping.items():
                if key.startswith(old_prefix):
                    key = new_prefix + key[len(old_prefix):]
                    break
            else:
                continue  # 旧结构中的 relu/maxpool 无参数,跳过
        out[key] = value
    return out


class SEBlock(nn.Module):
    """Squeeze-Excitation 通道注意力模块。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(self.pool(x).flatten(1))
        return x * weights.view(b, c, 1, 1)


class SpatialGate(nn.Module):
    """CBAM 风格空间门控:对每个空间位置生成 0-1 注意力权重。

    沿通道做 avg/max 池化得到 2 张空间图,7x7 卷积融合后 sigmoid。
    末层卷积零权重 + 正偏置初始化,起始门控接近恒等映射,避免扰动预训练特征。
    """

    def __init__(self, bias_init: float = 4.0, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate


class FiLMGenerator(nn.Module):
    """FiLM 条件化调制(Perez et al. 2018):metadata 生成逐通道仿射参数调制视觉特征。

    γ/β 零初始化 → 起始为恒等映射(γ=1, β=0),训练中逐渐分化,不扰动预训练特征。
    """

    def __init__(self, metadata_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Linear(hidden, channels)
        self.beta = nn.Linear(hidden, channels)
        for head in (self.gamma, self.beta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        h = self.trunk(metadata)
        gamma = (1.0 + self.gamma(h)).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        beta = self.beta(h).unsqueeze(-1).unsqueeze(-1)             # [B, C, 1, 1]
        return x * gamma + beta


class SpatialPyramidPool(nn.Module):
    """SPP-Net 风格空间金字塔池化(He et al. 2015)。

    对最后一层特征图按多组 bin(默认 1/2/4)做自适应平均池化后拼接,
    输出维度与输入分辨率无关,因此网络可接受任意尺寸输入。
    """

    def __init__(self, channels: int, levels: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.levels = levels
        self.output_dim = channels * sum(k * k for k in levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [
            torch.flatten(nn.functional.adaptive_avg_pool2d(x, (k, k)), 1)
            for k in self.levels
        ]
        return torch.cat(pooled, dim=1)


class EyeStateClassifier(nn.Module):
    def __init__(
        self,
        pretrained=True, # 是否加载预训练权重
        use_metadata=True,
        use_se=False,
        multi_scale=False,
        spatial_attn=False,
        film=False,
        use_spp=False,
    ):
        super().__init__()

        self.use_metadata = use_metadata
        self.use_se = use_se
        self.multi_scale = multi_scale
        self.spatial_attn = spatial_attn
        self.film = film and use_metadata
        self.use_spp = use_spp

        # 1. 加载 ResNet34 预训练模型
        weights = (
            models.ResNet34_Weights.DEFAULT
            if pretrained else None
        )
        backbone = models.resnet34(weights=weights)

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

        # 3. 拆分 backbone,保留各 stage 以支持多尺度特征融合
        children = list(backbone.children())
        self.stem = nn.Sequential(*children[:4])   # conv1/bn1/relu/maxpool
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

        # 4. SE 通道注意力,插在 layer3/layer4 之后
        if use_se:
            self.se3 = SEBlock(256)
            self.se4 = SEBlock(512)

        # 4.5 空间注意力,插在 layer2/layer3 之后(空间信息仍丰富处;layer4 过小且邻接 GAP 头,冗余)
        if spatial_attn:
            self.gate2 = SpatialGate()
            self.gate3 = SpatialGate()

        # 4.6 FiLM:metadata 深度注入,调制 layer3/layer4 输出(替代末端 concat 的浅层交互)
        if self.film:
            metadata_dim = 9
            self.film3 = FiLMGenerator(metadata_dim, 256)
            self.film4 = FiLMGenerator(metadata_dim, 512)

        # 5. 特征维度: SPP 优先(512×(1+4+16)=10752);多尺度融合取 layer2/3/4 的 GAP 拼接
        if use_spp:
            self.spp = SpatialPyramidPool(512)
            self.feature_dim = self.spp.output_dim
        else:
            self.feature_dim = 128 + 256 + 512 if multi_scale else 512

        # 6. 图像和辅助特征维度
        metadata_dim = 9 if use_metadata else 0

        # 7. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + metadata_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, metadata):
        # image: [B, 1, H, W]
        # metadata: [B, 9]

        features = self.stem(image)
        features = self.layer1(features)
        f2 = self.layer2(features)                # [B, 128, H/8, W/8]
        if self.spatial_attn:
            f2 = self.gate2(f2)
        f3 = self.layer3(f2)                      # [B, 256, H/16, W/16]
        if self.spatial_attn:
            f3 = self.gate3(f3)
        if self.use_se:
            f3 = self.se3(f3)
        if self.film:
            f3 = self.film3(f3, metadata)
        f4 = self.layer4(f3)                      # [B, 512, H/32, W/32]
        if self.use_se:
            f4 = self.se4(f4)
        if self.film:
            f4 = self.film4(f4, metadata)

        if self.use_spp:
            # SPP: 多组 bin 的空间金字塔池化,支持任意输入分辨率
            image_features = self.spp(f4)
        elif self.multi_scale:
            # 多尺度 GAP 拼接: 128 + 256 + 512 = 896
            pooled = [
                torch.flatten(nn.functional.adaptive_avg_pool2d(t, 1), 1)
                for t in (f2, f3, f4)
            ]
            image_features = torch.cat(pooled, dim=1)
        else:
            image_features = torch.flatten(
                nn.functional.adaptive_avg_pool2d(f4, 1), 1
            )

        if self.use_metadata:
            image_features = torch.cat(
                [image_features, metadata], dim=1
            )

        # 输出二分类 logit(单个,配合 BCEWithLogitsLoss)
        logits = self.classifier(image_features)

        return logits
