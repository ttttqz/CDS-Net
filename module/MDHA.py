import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from module.KAN import KANLinear


class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = KANLinear(num_channels, num_channels_reduced)
        self.fc2 = KANLinear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()

        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlpx = nn.Linear(patch_size, output_dim // 4)
        self.mlpy = nn.Linear(patch_size, output_dim // 4)
        self.mlp2 = nn.Linear(output_dim // 4, output_dim)
        self.mlp4 = nn.Linear(output_dim // 4, output_dim)
        self.norm2 = nn.LayerNorm(output_dim // 4)
        self.norm4 = nn.LayerNorm(output_dim // 4)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)


    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        # Local branch
        local_patches_x = x.unfold(2, P, P)
        local_patches_y = x.unfold(3, P, P)
        local_patches_x = local_patches_x.reshape(B, C, -1, P)
        local_patches_y = local_patches_y.reshape(B, C, -1, P)
        local_patches_x = local_patches_x.mean(dim=1)
        local_patches_y = local_patches_y.mean(dim=1)
        local_patches_x = self.mlpx(local_patches_x)
        local_patches_y = self.mlpy(local_patches_y)
        if P == 2:
            local_patches = torch.cat((local_patches_x, local_patches_y), dim=1)
            local_patches = self.norm2(local_patches)
            local_patches = self.mlp2(local_patches)
            local_attention = F.softmax(local_patches, dim=-1)
            local_out = local_patches * local_attention
            cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
            mask = cos_sim.clamp(0, 1)
            local_out = local_out * mask
            local_out = local_out @ self.top_down_transform
            local_out = local_out.reshape(B, H, W, self.output_dim)
            local_out = local_out.permute(0, 3, 1, 2)
            output = self.conv2(local_out)
            return output
        if P == 4:
            local_patches_y = TF.resize(local_patches_y, size=local_patches_x.shape[1:])
            local_patches = local_patches_x + local_patches_y
            local_patches = self.norm4(local_patches)
            local_patches = self.mlp4(local_patches)
            local_attention = F.softmax(local_patches, dim=-1)
            local_out = local_patches * local_attention
            cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
            mask = cos_sim.clamp(0, 1)
            local_out = local_out * mask
            local_out = local_out @ self.top_down_transform
            local_out = local_out.reshape(B, H // 2, W // 2, self.output_dim)
            local_out = local_out.permute(0, 3, 1, 2)
            local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
            output = self.conv4(local_out)
            return output


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class MDHA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)
        self.se = SELayer(filters, reduction_ratio=8)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.bn3 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.35)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.c2(x1)
        x2 = (x1 + x2 + x_skip) + (x_lga2 + x_lga4)
        x2 = self.drop(x2)
        x3 = self.c3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x = self.se(x3)

        return x

