# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Transformer modules
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings if given."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)


    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
        #return (x + x.mean(dim=1, keepdim=True)) * 0.5  #以前的所有实验都用的这个
        # median = torch.median(x, dim=1, keepdim=True)[0]
        # return (x + median) * 0.5
class LayerNorm2d(nn.Module):
    """
    LayerNorm2d module from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Original Multi-Scale Deformable Attention Module.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        # self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # ffn
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.init import constant_, xavier_uniform_
#
# from .conv import Conv
# from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch
#
# __all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
#            'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'MedianCBS','CBS','ContextBroadcasting','MedianContextBroadcasting','ApproximateModeContextBroadcasting','ApproximateModeCBS')
#
#
# class TransformerEncoderLayer(nn.Module):
#     """Transformer Encoder."""
#
#     def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
#         super().__init__()
#         from ...utils.torch_utils import TORCH_1_9
#         if not TORCH_1_9:
#             raise ModuleNotFoundError(
#                 'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
#         self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
#         # Implementation of Feedforward model
#         self.fc1 = nn.Linear(c1, cm)
#         self.fc2 = nn.Linear(cm, c1)
#
#         self.norm1 = nn.LayerNorm(c1)
#         self.norm2 = nn.LayerNorm(c1)
#         self.dropout = nn.Dropout(dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.act = act
#         self.normalize_before = normalize_before
#
#     def with_pos_embed(self, tensor, pos=None):
#         """Add position embeddings if given."""
#         return tensor if pos is None else tensor + pos
#
#     def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
#         q = k = self.with_pos_embed(src, pos)
#         src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
#         src = src + self.dropout2(src2)
#         return self.norm2(src)
#
#     def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
#         return src + self.dropout2(src2)
#
#     def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
#         """Forward propagates the input through the encoder module."""
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)
#
#
# class AIFI(TransformerEncoderLayer):
#
#     def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
#         super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
#
#     def forward(self, x):
#         c, h, w = x.shape[1:]
#         pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
#         # flatten [B, C, H, W] to [B, HxW, C]
#         x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
#         return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
#
#     @staticmethod
#     def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
#         grid_w = torch.arange(int(w), dtype=torch.float32)
#         grid_h = torch.arange(int(h), dtype=torch.float32)
#         grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
#         assert embed_dim % 4 == 0, \
#             'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
#         pos_dim = embed_dim // 4
#         omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
#         omega = 1. / (temperature ** omega)
#
#         out_w = grid_w.flatten()[..., None] @ omega[None]
#         out_h = grid_h.flatten()[..., None] @ omega[None]
#
#         return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
#
#
# class TransformerLayer(nn.Module):
#     """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""
#
#     def __init__(self, c, num_heads):
#         """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
#         super().__init__()
#         self.q = nn.Linear(c, c, bias=False)
#         self.k = nn.Linear(c, c, bias=False)
#         self.v = nn.Linear(c, c, bias=False)
#         self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
#         self.fc1 = nn.Linear(c, c, bias=False)
#         self.fc2 = nn.Linear(c, c, bias=False)
#
#
#     def forward(self, x):
#         """Apply a transformer block to the input x and return the output."""
#         x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
#         return self.fc2(self.fc1(x)) + x
#
#
# class TransformerBlock(nn.Module):
#     """Vision Transformer https://arxiv.org/abs/2010.11929."""
#
#     def __init__(self, c1, c2, num_heads, num_layers):
#         """Initialize a Transformer module with position embedding and specified number of heads and layers."""
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
#         self.c2 = c2
#
#     def forward(self, x):
#         """Forward propagates the input through the bottleneck module."""
#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         p = x.flatten(2).permute(2, 0, 1)
#         return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
#
#
# # class MLPBlock(nn.Module):
# #
# #     def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
# #         super().__init__()
# #         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
# #         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
# #         self.act = act()
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return self.lin2(self.act(self.lin1(x)))
# class ContextBroadcasting(nn.Module):   #########11111111111111
#     def __init__(self):
#         super(ContextBroadcasting, self).__init__()
#
#     def forward(self, x):
#         # x: [batch_size, num_tokens, token_dim]
#         global_context = x.mean(dim=1, keepdim=True)  # 计算全局上下文，保持维度以便于广播
#         cb = (x + global_context) / 2  # 将全局上下文广播到每个令牌并取平均
#         return cb
#
# class MedianContextBroadcasting(nn.Module):         ###################2222222222222222222
#     def __init__(self):
#         super(MedianContextBroadcasting, self).__init__()
#
#     def forward(self, x):
#         # x: [batch_size, num_tokens, token_dim]
#         # 计算每个token的维度上的中位数，保持维度以便于广播
#         median_token = x.median(dim=1, keepdim=True)[0]  # 返回值是一个元组，第一个元素是中位数
#         # 将中位数广播到每个token并取平均
#         mcb = (x + median_token) / 2
#         return mcb
# class ApproximateModeContextBroadcasting(nn.Module):   ###############333333333333333333
#     def __init__(self, temperature=0.1):
#         super(ApproximateModeContextBroadcasting, self).__init__()
#         # Temperature用于调节Softmax的"锐度"
#         self.temperature = temperature
#
#     def forward(self, x):
#         B, N, D = x.shape
#         # 将数据reshape以便处理
#         x_flattened = x.view(B * N, D)
#         # 计算自相似性
#         similarity = torch.matmul(x_flattened, x_flattened.t()) / self.temperature
#         # 应用Softmax获取权重
#         weights = F.softmax(similarity, dim=1)
#         # 计算加权平均值作为众数的近似
#         mode_approx = torch.matmul(weights, x_flattened) / weights.sum(dim=1, keepdim=True)
#         # 将结果reshape回原始维度并广播到每个令牌
#         mode_approx = mode_approx.view(B, N, D)
#         enhanced = (x + mode_approx) / 2
#         return enhanced
# class CBS(nn.Module):            ###############4444444444444444
#     def __init__(self, dim):
#         super(CBS, self).__init__()
#         self.dim = dim
#         self.scale = nn.Parameter(torch.ones(self.dim))
#
#     def forward(self, x):
#         # print("Input x shape:", x.shape)  # 输出输入x的形状
#         mean_context = x.mean(dim=1, keepdim=True)  # [B, 1, D]
#         # print("Mean context shape:", mean_context.shape)  # 输出mean_context的形状
#         scale_reshaped = self.scale.view(1, 1, self.dim)
#         # print("Scale shape:", scale_reshaped.shape)  # 输出scale_reshaped的形状
#         scaled_context = mean_context * scale_reshaped
#         enhanced = (x + scaled_context) / 2
#         return enhanced
# class MedianCBS(nn.Module):         ################555555555555555
#     def __init__(self, dim):
#         super(MedianCBS, self).__init__()
#         self.dim = dim
#         self.scale = nn.Parameter(torch.ones(self.dim))  # 初始化缩放因子
#
#     def forward(self, x):
#         # x: [batch_size, num_tokens, token_dim]
#         median_context, _ = x.median(dim=1, keepdim=True)  # 计算中位数，[B, 1, D]
#         # 重新定义scale的维度以适配median_context
#         scale_reshaped = self.scale.view(1, 1, self.dim)  # 确保这里使用的dim与median_context的最后一个维度相同
#         # 应用scale
#         scaled_context = median_context * scale_reshaped
#         # 广播和结合
#         enhanced = (x + scaled_context) / 2
#         return enhanced
# class ApproximateModeCBS(nn.Module):          ####################66666666666666666666
#     def __init__(self, dim, num_clusters=10):
#         super(ApproximateModeCBS, self).__init__()
#         self.dim = dim
#         self.num_clusters = num_clusters
#         self.scale = nn.Parameter(torch.ones(dim))
#         self.centroids = nn.Parameter(torch.randn(num_clusters, dim))
#
#     def forward(self, x):
#         # x: [batch_size, num_tokens, token_dim]
#         B, N, D = x.shape
#
#         # 计算所有样本与每个聚类中心的距离
#         distances = torch.cdist(x.view(-1, D), self.centroids)  # [B*N, num_clusters]
#         # 找到每个样本最近的聚类中心
#         min_indices = distances.argmin(dim=1)  # [B*N]
#         # 选择与每个样本最接近的聚类中心作为近似众数
#         mode_approx = self.centroids[min_indices].view(B, N, D)
#
#         # 计算平均众数作为上下文
#         mean_mode_context = mode_approx.mean(dim=1, keepdim=True)  # [B, 1, D]
#         # 应用scale
#         scaled_context = mean_mode_context * self.scale.view(1, 1, D)
#         # 广播和结合
#         enhanced = (x + scaled_context) / 2
#         return enhanced
# class MLPBlock(nn.Module):
#
#     def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
#         super().__init__()
#         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
#         self.act = act()
#         self.cb = ApproximateModeCBS(mlp_dim)  # 调整dim参数以匹配实际的特征维度
#         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.lin1(x)
#         x = self.act(x)
#         x = self.cb(x)
#         x = self.lin2(x)
#         return x
#
# # MedianCBS 模块应用在所有线性层和激活函数之后
# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])
#         self.cb = ApproximateModeCBS(hidden_dim)  # Adjust dim parameter to match the actual feature dimension
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         x = self.cb(x)
#         # return (x + x.mean(dim=1, keepdim=True)) * 0.5
#         return x
#
# #MedianCBS 模块在第一个线性层和激活函数之后应用
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个线性层
# #         self.act = nn.ReLU()  # 激活函数
# #         self.cb = MedianCBS(hidden_dim)  # Adjust dim parameter to match the actual feature dimension
# #         self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([hidden_dim] + h, h + [output_dim])])
# #
# #     def forward(self, x):
# #         x = self.fc1(x)  # 第一个线性层
# #         x = self.act(x)  # 激活函数
# #         x = self.cb(x)  # MedianCBS
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
#
# class LayerNorm2d(nn.Module):
#     """
#     LayerNorm2d module from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
#     https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
#     """
#
#     def __init__(self, num_channels, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps
#
#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         return self.weight[:, None, None] * x + self.bias[:, None, None]
#
#
# class MSDeformAttn(nn.Module):
#     """
#     Original Multi-Scale Deformable Attention Module.
#     https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
#     """
#
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'
#
#         self.im2col_step = 64
#
#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points
#
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
#             1, self.n_levels, self.n_points, 1)
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)
#
#     def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
#         """
#         https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#         Args:
#             query (torch.Tensor): [bs, query_length, C]
#             refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area
#             value (torch.Tensor): [bs, value_length, C]
#             value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#             value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements
#
#         Returns:
#             output (Tensor): [bs, Length_{query}, C]
#         """
#         bs, len_q = query.shape[:2]
#         len_v = value.shape[1]
#         assert sum(s[0] * s[1] for s in value_shapes) == len_v
#
#         value = self.value_proj(value)
#         if value_mask is not None:
#             value = value.masked_fill(value_mask[..., None], float(0))
#         value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
#         attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
#         # N, Len_q, n_heads, n_levels, n_points, 2
#         num_points = refer_bbox.shape[-1]
#         if num_points == 2:
#             offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
#             add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#             sampling_locations = refer_bbox[:, :, None, :, None, :] + add
#         elif num_points == 4:
#             add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
#             sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
#         else:
#             raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
#         output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
#         return self.output_proj(output)
#
#
# class DeformableTransformerDecoderLayer(nn.Module):
#     """
#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#     https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
#     """
#
#     def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
#         super().__init__()
#
#         # self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#
#         # cross attention
#         self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.act = act
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)
#
#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos
#
#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         return self.norm3(tgt)
#
#     def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
#         # self attention
#         q = k = self.with_pos_embed(embed, query_pos)
#         tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
#                              attn_mask=attn_mask)[0].transpose(0, 1)
#         embed = embed + self.dropout1(tgt)
#         embed = self.norm1(embed)
#
#         # cross attention
#         tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
#                               padding_mask)
#         embed = embed + self.dropout2(tgt)
#         embed = self.norm2(embed)
#
#         # ffn
#         return self.forward_ffn(embed)
#
#
# class DeformableTransformerDecoder(nn.Module):
#     """
#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#     """
#
#     def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
#
#     def forward(
#             self,
#             embed,  # decoder embeddings
#             refer_bbox,  # anchor
#             feats,  # image features
#             shapes,  # feature shapes
#             bbox_head,
#             score_head,
#             pos_mlp,
#             attn_mask=None,
#             padding_mask=None):
#         output = embed
#         dec_bboxes = []
#         dec_cls = []
#         last_refined_bbox = None
#         refer_bbox = refer_bbox.sigmoid()
#         for i, layer in enumerate(self.layers):
#             output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
#
#             bbox = bbox_head[i](output)
#             refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
#
#             if self.training:
#                 dec_cls.append(score_head[i](output))
#                 if i == 0:
#                     dec_bboxes.append(refined_bbox)
#                 else:
#                     dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
#             elif i == self.eval_idx:
#                 dec_cls.append(score_head[i](output))
#                 dec_bboxes.append(refined_bbox)
#                 break
#
#             last_refined_bbox = refined_bbox
#             refer_bbox = refined_bbox.detach() if self.training else refined_bbox
#
#         return torch.stack(dec_bboxes), torch.stack(dec_cls)
