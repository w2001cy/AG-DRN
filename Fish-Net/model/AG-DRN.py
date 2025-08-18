import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import use_spectral_norm
from .basic_net import Conv2dBlock, ResBlocks, TransConv2dBlock
from .resample2d import resample_image
from torchvision.ops import DeformConv2d
from .DAT import DAT
from .DDA import DDA, make_laplace_pyramid


class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()

  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        nn.init.kaiming_normal_(m.attn.qkv.weight, mode='fan_in')
        nn.init.xavier_uniform_(m.ffn.project_out.weight)
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)


class InpaintGenerator(nn.Module):
  def __init__(self):
    super(InpaintGenerator, self).__init__()

    self.flow_column = FlowColumn()
    self.conv_column = InpaintNet()

  def forward(self, inputs):
      flow_map, flows = self.flow_column(inputs)
      pyramid_imgs, images_out = self.conv_column(inputs, flows)
      return pyramid_imgs, images_out


class InpaintNet(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintNet, self).__init__()

        cnum = 32

        # Initialize all network components
        self._init_encoder_layers(cnum)
        self._init_transformer_layers(cnum)
        self._init_decoder_layers(cnum)
        self._init_output_layers(cnum)
        self._init_prediction_heads(cnum)
        self._init_edge_adapters()

        if init_weights:
            self.init_weights()

    def _init_encoder_layers(self, cnum):
        """Initialize encoder (downsampling) layers."""
        self.dw_conv01 = nn.Sequential(
            nn.Conv2d(3, cnum, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            DDA(cnum)
        )

        self.dw_conv02 = nn.Sequential(
            nn.Conv2d(cnum, cnum * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            DDA(cnum * 2)
        )

        self.dw_conv03 = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            DDA(cnum * 4)
        )

        self.dw_conv04 = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dw_conv05 = nn.Sequential(
            nn.Conv2d(cnum * 8, cnum * 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dw_conv06 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def _init_transformer_layers(self, cnum):
        """Initialize transformer layers for different scales."""
        transformer_config = {
            'num_heads': 8,
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias'
        }

        # Downsampling transformers
        self.dw_tr04 = nn.Sequential(
            DAT(dim=cnum * 4, **transformer_config)
        )

        self.dw_tr05 = nn.Sequential(
            DAT(dim=cnum * 8, **transformer_config)
        )

        self.dw_tr06 = nn.Sequential(
            DAT(dim=cnum * 16, **transformer_config)
        )

        # Upsampling transformers
        self.up_tr04 = nn.Sequential(
            DAT(dim=cnum * 8, **transformer_config)
        )

        self.up_tr05 = nn.Sequential(
            DAT(dim=cnum * 16, **transformer_config)
        )

    def _init_decoder_layers(self, cnum):
        """Initialize decoder (upsampling) layers."""
        self.up_conv05 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv04 = nn.Sequential(
            nn.Conv2d(cnum * 32, cnum * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv03 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 4, 3, 1, 1),
            nn.ReLU(True),
            DDA(cnum * 4)
        )

        self.up_conv02 = nn.Sequential(
            nn.Conv2d(cnum * 8, cnum * 2, 3, 1, 1),
            nn.ReLU(True),
            DDA(cnum * 2)
        )

        self.up_conv01 = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum, 3, 1, 1),
            nn.ReLU(True),
            DDA(cnum)
        )

    def _init_output_layers(self, cnum):
        """Initialize output layers for different pyramid levels."""
        # Final decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnum, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Pyramid output layers (to RGB)
        self.torgb5 = nn.Sequential(
            nn.Conv2d(cnum * 32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.torgb4 = nn.Sequential(
            nn.Conv2d(cnum * 16, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.torgb3 = nn.Sequential(
            nn.Conv2d(cnum * 8, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.torgb2 = nn.Sequential(
            nn.Conv2d(cnum * 4, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.torgb1 = nn.Sequential(
            nn.Conv2d(cnum * 2, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _init_prediction_heads(self, cnum):
        """Initialize prediction heads for different feature levels."""
        self.pred_heads = nn.ModuleList([
            nn.Conv2d(cnum, 1, 1),  # for dw_conv01
            nn.Conv2d(cnum * 2, 1, 1),  # for dw_conv02
            nn.Conv2d(cnum * 4, 1, 1),  # for dw_conv03
            nn.Conv2d(cnum, 1, 1),  # for up_conv01
            nn.Conv2d(cnum * 2, 1, 1),  # for up_conv02
            nn.Conv2d(cnum * 4, 1, 1)  # for up_conv03
        ])

    def _init_edge_adapters(self):
        """Initialize edge feature adapters for different channel sizes."""
        self.edge_adapter = nn.ModuleDict({
            '32': nn.Conv2d(3, 32, 1),
            '64': nn.Conv2d(3, 64, 1),
            '128': nn.Conv2d(3, 128, 1)
        })

    def compute_edge_features(self, x):
        """Compute edge features using Laplace pyramid."""
        with torch.no_grad():
            pyr = make_laplace_pyramid(x, level=3, channels=3)
            return pyr[0]

    def _encode_features(self, img, edge_base):
        """Encode input image through downsampling path with edge features."""
        x = img

        # Level 1 (32 channels)
        x1 = self.dw_conv01[:-1](x)
        x1 = self.dw_conv01[-1](
            edge_feature=self.edge_adapter['32'](
                F.interpolate(edge_base, x1.shape[2:], mode='bilinear')),
            x=x1,
            pred=self.pred_heads[0](x1)
        )

        # Level 2 (64 channels)
        x2 = self.dw_conv02[:-1](x1)
        x2 = self.dw_conv02[-1](
            edge_feature=self.edge_adapter['64'](
                F.interpolate(edge_base, x2.shape[2:], mode='bilinear')),
            x=x2,
            pred=self.pred_heads[1](x2)
        )

        # Level 3 (128 channels)
        x3 = self.dw_conv03[:-1](x2)
        x3 = self.dw_conv03[-1](
            edge_feature=self.edge_adapter['128'](
                F.interpolate(edge_base, x3.shape[2:], mode='bilinear')),
            x=x3,
            pred=self.pred_heads[2](x3)
        )

        # Level 4-6 (deeper layers)
        x4 = self.dw_conv04(x3)
        x5 = self.dw_conv05(self.dw_tr05(x4))
        x6 = self.dw_conv06(self.dw_tr06(x5))

        return x1, x2, x3, x4, x5, x6

    def _apply_flow_resampling(self, features, flows):
        """Apply flow-based resampling to encoded features."""
        x1, x2, x3, x4, x5, x6 = features

        x5 = resample_image(x5, flows[4])
        x4 = resample_image(x4, flows[3])
        x3 = resample_image(x3, flows[2])
        x2 = resample_image(x2, flows[1])
        x1 = resample_image(x1, flows[0])

        return x1, x2, x3, x4, x5, x6

    def _decode_features(self, features, edge_base):
        """Decode features through upsampling path with skip connections."""
        x1, x2, x3, x4, x5, x6 = features

        # Upsampling path
        upx5 = self.up_conv05(
            F.interpolate(self.dw_tr06(x6), scale_factor=2, mode='bilinear', align_corners=True)
        )

        upx4 = self.up_conv04(
            F.interpolate(
                torch.cat([self.up_tr05(upx5), x5], dim=1),
                scale_factor=2, mode='bilinear', align_corners=True
            )
        )

        # Level 3 with edge features
        upx3 = self.up_conv03[0](
            F.interpolate(
                torch.cat([self.up_tr04(upx4), x4], dim=1),
                scale_factor=2, mode='bilinear'
            )
        )
        upx3 = self.up_conv03[1](upx3)
        upx3 = self.up_conv03[2](
            edge_feature=self.edge_adapter['128'](
                F.interpolate(edge_base, upx3.shape[2:])
            ),
            x=upx3,
            pred=self.pred_heads[5](upx3)
        )

        # Level 2 with edge features
        upx2 = self.up_conv02[0](
            F.interpolate(torch.cat([upx3, x3], dim=1), scale_factor=2, mode='bilinear')
        )
        upx2 = self.up_conv02[1](upx2)
        upx2 = self.up_conv02[2](
            edge_feature=self.edge_adapter['64'](
                F.interpolate(edge_base, upx2.shape[2:])
            ),
            x=upx2,
            pred=self.pred_heads[4](upx2)
        )

        # Level 1 with edge features
        upx1 = self.up_conv01[0](
            F.interpolate(torch.cat([upx2, x2], dim=1), scale_factor=2, mode='bilinear')
        )
        upx1 = self.up_conv01[1](upx1)
        upx1 = self.up_conv01[2](
            edge_feature=self.edge_adapter['32'](
                F.interpolate(edge_base, upx1.shape[2:])
            ),
            x=upx1,
            pred=self.pred_heads[3](upx1)
        )

        return upx1, upx2, upx3, upx4, upx5, x1, x2, x3, x4, x5

    def _generate_pyramid_outputs(self, decoder_features):
        """Generate multi-scale pyramid outputs."""
        upx1, upx2, upx3, upx4, upx5, x1, x2, x3, x4, x5 = decoder_features

        img5 = self.torgb5(torch.cat([upx5, x5], dim=1))
        img4 = self.torgb4(torch.cat([upx4, x4], dim=1))
        img3 = self.torgb3(torch.cat([upx3, x3], dim=1))
        img2 = self.torgb2(torch.cat([upx2, x2], dim=1))
        img1 = self.torgb1(torch.cat([upx1, x1], dim=1))

        # Final output
        output = self.decoder(
            F.interpolate(
                torch.cat([upx1, x1], dim=1),
                scale_factor=2, mode='bilinear', align_corners=True
            )
        )

        return [img1, img2, img3, img4, img5], output

    def forward(self, img, flows):
        """Forward pass of the InpaintNet."""
        # Compute edge features
        edge_base = self.compute_edge_features(img)

        # Encode features
        encoded_features = self._encode_features(img, edge_base)

        # Apply flow resampling
        resampled_features = self._apply_flow_resampling(encoded_features, flows)

        # Decode features
        decoder_features = self._decode_features(resampled_features, edge_base)

        # Generate outputs
        pyramid_imgs, output = self._generate_pyramid_outputs(decoder_features)

        return pyramid_imgs, output


class FlowColumn(nn.Module):
    """
    Flow estimation network using U-Net architecture with edge-aware features.
    Generates multi-scale optical flow maps for video inpainting tasks.
    """

    def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
                 norm='in', pad_type='reflect', use_sn=True):
        super(FlowColumn, self).__init__()

        self.base_dim = dim
        self.n_res = n_res

        # Initialize all network components
        self._init_encoder_layers(input_dim, dim, norm, activ, pad_type, use_sn)
        self._init_decoder_layers(dim, n_res, norm, activ, pad_type, use_sn)
        self._init_flow_output_layers(dim, pad_type)
        self._init_prediction_heads()
        self._init_edge_adapters()

    def _init_encoder_layers(self, input_dim, dim, norm, activ, pad_type, use_sn):
        """Initialize encoder (downsampling) layers."""
        # Level 1: Input -> dim//2 channels
        self.down_flow01 = nn.Sequential(
            Conv2dBlock(input_dim, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim // 2)
        )

        # Level 2: dim//2 -> dim//2 channels
        self.down_flow02 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim // 2)
        )

        # Level 3: dim//2 -> dim channels
        self.down_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim)
        )

        # Level 4: dim -> 2*dim channels
        self.down_flow04 = nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim * 2)
        )

        # Level 5: 2*dim -> 4*dim channels
        self.down_flow05 = nn.Sequential(
            Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim * 4)
        )

        # Level 6: 4*dim -> 8*dim channels (bottleneck)
        self.down_flow06 = nn.Sequential(
            Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DDA(dim * 8)
        )

    def _init_decoder_layers(self, dim, n_res, norm, activ, pad_type, use_sn):
        """Initialize decoder (upsampling) layers with skip connections."""
        # Update dim to bottleneck dimension
        dim = 8 * dim

        # Level 5: 8*base_dim -> 4*base_dim
        self.up_flow05 = nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim, dim // 2, 6, 2, 2, norm=norm, activation=activ)
        )

        # Level 4: 8*base_dim -> 2*base_dim (with skip connection)
        self.up_flow04 = nn.Sequential(
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 2, dim // 4, 6, 2, 2, norm=norm, activation=activ)
        )

        # Level 3: 4*base_dim -> base_dim (with skip connection)
        self.up_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 4, dim // 8, 6, 2, 2, norm=norm, activation=activ)
        )

        # Level 2: 2*base_dim -> base_dim//2 (with skip connection)
        self.up_flow02 = nn.Sequential(
            Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 8, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 8, dim // 16, 6, 2, 2, norm=norm, activation=activ)
        )

        # Level 1: base_dim -> base_dim//2 (with skip connection)
        self.up_flow01 = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 16, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm=norm, activation=activ)
        )

    def _init_flow_output_layers(self, dim, pad_type):
        """Initialize flow output layers and final location prediction."""
        # Update dim to bottleneck dimension
        dim = 8 * dim

        # Final location/flow map prediction
        self.location = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, 'in', 'lrelu', pad_type, use_sn=True),
            ResBlocks(self.n_res, dim // 16, 'in', 'lrelu', pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm='in', activation='lrelu'),
            Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none',
                        pad_type=pad_type, use_bias=False)
        )

        # Multi-scale flow output layers
        self.to_flow05 = Conv2dBlock(dim // 2, 2, 3, 1, 1, norm='none', activation='none',
                                     pad_type=pad_type, use_bias=False)
        self.to_flow04 = Conv2dBlock(dim // 4, 2, 3, 1, 1, norm='none', activation='none',
                                     pad_type=pad_type, use_bias=False)
        self.to_flow03 = Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none',
                                     pad_type=pad_type, use_bias=False)
        self.to_flow02 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none',
                                     pad_type=pad_type, use_bias=False)
        self.to_flow01 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none',
                                     pad_type=pad_type, use_bias=False)

    def _init_prediction_heads(self):
        """Initialize prediction heads for different feature levels."""
        # Channel dimensions for each level
        channel_dims = [32, 32, 64, 128, 256, 512]

        self.pred_heads = nn.ModuleList([
            nn.Conv2d(channels, 1, 1) for channels in channel_dims
        ])

    def _init_edge_adapters(self):
        """Initialize edge feature adapters for different channel sizes."""
        self.edge_adapter = nn.ModuleDict({
            '32': nn.Conv2d(3, 32, 1),
            '64': nn.Conv2d(3, 64, 1),
            '128': nn.Conv2d(3, 128, 1),
            '256': nn.Conv2d(3, 256, 1),
            '512': nn.Conv2d(3, 512, 1)
        })

    def compute_edge_features(self, x):
        """Compute edge features using Laplace pyramid."""
        with torch.no_grad():
            pyr = make_laplace_pyramid(x, level=3, channels=3)
            return pyr[0]

    def _encode_with_edge_features(self, inputs, edge_base):
        """Encode input through downsampling path with edge-aware DDA modules."""
        # Level 1: 32 channels
        f_x1 = self.down_flow01[:-1](inputs)
        f_x1 = self.down_flow01[-1](
            edge_feature=self.edge_adapter['32'](
                F.interpolate(edge_base, f_x1.shape[2:], mode='bilinear')
            ),
            x=f_x1,
            pred=self.pred_heads[0](f_x1)
        )

        # Level 2: 32 channels
        f_x2 = self.down_flow02[:-1](f_x1)
        f_x2 = self.down_flow02[-1](
            edge_feature=self.edge_adapter['32'](
                F.interpolate(edge_base, f_x2.shape[2:], mode='bilinear')
            ),
            x=f_x2,
            pred=self.pred_heads[1](f_x2)
        )

        # Level 3: 64 channels
        f_x3 = self.down_flow03[:-1](f_x2)
        f_x3 = self.down_flow03[-1](
            edge_feature=self.edge_adapter['64'](
                F.interpolate(edge_base, f_x3.shape[2:], mode='bilinear')
            ),
            x=f_x3,
            pred=self.pred_heads[2](f_x3)
        )

        # Level 4: 128 channels
        f_x4 = self.down_flow04[:-1](f_x3)
        f_x4 = self.down_flow04[-1](
            edge_feature=self.edge_adapter['128'](
                F.interpolate(edge_base, f_x4.shape[2:], mode='bilinear')
            ),
            x=f_x4,
            pred=self.pred_heads[3](f_x4)
        )

        # Level 5: 256 channels
        f_x5 = self.down_flow05[:-1](f_x4)
        f_x5 = self.down_flow05[-1](
            edge_feature=self.edge_adapter['256'](
                F.interpolate(edge_base, f_x5.shape[2:], mode='bilinear')
            ),
            x=f_x5,
            pred=self.pred_heads[4](f_x5)
        )

        # Level 6: 512 channels (bottleneck)
        f_x6 = self.down_flow06[:-1](f_x5)
        f_x6 = self.down_flow06[-1](
            edge_feature=self.edge_adapter['512'](
                F.interpolate(edge_base, f_x6.shape[2:], mode='bilinear')
            ),
            x=f_x6,
            pred=self.pred_heads[5](f_x6)
        )

        return f_x1, f_x2, f_x3, f_x4, f_x5, f_x6

    def _decode_with_skip_connections(self, encoded_features):
        """Decode features through upsampling path with skip connections."""
        f_x1, f_x2, f_x3, f_x4, f_x5, f_x6 = encoded_features

        # Upsampling with skip connections
        f_u5 = self.up_flow05(f_x6)
        f_u4 = self.up_flow04(torch.cat((f_u5, f_x5), 1))
        f_u3 = self.up_flow03(torch.cat((f_u4, f_x4), 1))
        f_u2 = self.up_flow02(torch.cat((f_u3, f_x3), 1))
        f_u1 = self.up_flow01(torch.cat((f_u2, f_x2), 1))

        return f_u1, f_u2, f_u3, f_u4, f_u5, f_x1

    def _generate_multi_scale_flows(self, decoder_features):
        """Generate flow maps at multiple scales."""
        f_u1, f_u2, f_u3, f_u4, f_u5, f_x1 = decoder_features

        # Generate final flow map
        flow_map = self.location(torch.cat((f_u1, f_x1), 1))

        # Generate multi-scale flow outputs
        flow05 = self.to_flow05(f_u5)
        flow04 = self.to_flow04(f_u4)
        flow03 = self.to_flow03(f_u3)
        flow02 = self.to_flow02(f_u2)
        flow01 = self.to_flow01(f_u1)

        return flow_map, [flow01, flow02, flow03, flow04, flow05]

    def forward(self, inputs):
        """
        Forward pass of FlowColumn network.

        Args:
            inputs: Input tensor of shape (B, C, H, W)

        Returns:
            flow_map: Final flow map tensor
            multi_scale_flows: List of flow maps at different scales
        """
        # Compute edge features for edge-aware processing
        edge_base = self.compute_edge_features(inputs)

        # Encode features with edge-aware DDA modules
        encoded_features = self._encode_with_edge_features(inputs, edge_base)

        # Decode features with skip connections
        decoder_features = self._decode_with_skip_connections(encoded_features)

        # Generate multi-scale flow outputs
        flow_map, multi_scale_flows = self._generate_multi_scale_flows(decoder_features)

        return flow_map, multi_scale_flows

class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()


  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x

