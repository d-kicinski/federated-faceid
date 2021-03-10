from torch import nn

NoLayerNorm = nn.Identity
BatchNorm = nn.BatchNorm2d


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int = 32, **kwargs):
        super(GroupNorm, self).__init__(num_channels=num_channels, num_groups=num_groups, **kwargs)
