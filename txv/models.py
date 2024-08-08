from typing import List, Literal, Optional, Type, Union
import torch
from torch import nn
from einops import rearrange
from txv.utils import *



class PatchEmbeddings(nn.Module):
    def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            in_chans: int = 3 
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim        

    def forward(self,image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        assert H == self.image_size and W == self.image_size, f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"
        assert self.image_size % self.patch_size == 0, f"Input dimensions {self.image_size}x{self.image_size} not divisble for patch size {self.patch_size}"
        
        return rearrange(self.proj(image),'b c h w -> b (h w) c')

    
class Attention(nn.Module):
    def __init__(self,
                embed_dim: int = 768,
                num_heads: int = 12,
                qkv_bias: bool = False, 
                save_att: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.save_att = save_att
        self.issaveq = False
        self.issavek = False
        self.issavev = False
        

        self.attn_gradients = None
        self.attention_map = None
        
        self.input = None       
        self.q = None
        self.k = None
        self.v = None
        # self.output = None

    def save_attn(self, att: torch.Tensor) -> None:
        self.attention_map = att

    def get_attn(self) -> torch.Tensor:
        assert self.attention_map is not None, "Please do forward pass before extracting attention map"
        return self.attention_map
    
    def save_attgradients(self, grad: torch.Tensor) -> None:
        self.attn_gradients = grad

    def get_attgradients(self) -> torch.Tensor:
        assert self.attn_gradients is not None, "Please do backward pass before extracting attention gradients"
        return self.attn_gradients
    
    def save_q(self, q: torch.Tensor) -> None:
        self.q = q

    def get_q(self) -> torch.Tensor:
        assert self.q is not None, "Please do forward pass before extracting q"
        return self.q
    
    def save_k(self, k: torch.Tensor) -> None:
        self.k = k

    def get_k(self) -> torch.Tensor:
        assert self.k is not None, "Please do forward pass before extracting k"
        return self.k
    
    def save_v(self, v: torch.Tensor) -> None:
        self.v = v

    def get_v(self) -> torch.Tensor:    
        return self.v       
    
    def save_input(self, inp: torch.Tensor) -> None:
        self.input = inp

    def get_input(self) -> torch.Tensor:
        assert self.input is not None, "Please do forward pass before extracting input"
        return self.input
    
    # def save_output(self, out: torch.Tensor) -> None:
    #     self.output = out

    # def get_output(self) -> torch.Tensor:
    #     assert self.output is not None, "Please do forward pass before extracting output"
    #     return self.output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv,'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)
        
        # Save q,k,v for visualization
        if self.issaveq:
            self.save_q(q)
        if self.issavek:
            self.save_k(k)
        if self.issavev:
            self.save_v(v)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.softmax(attn)

        # Save attention map for visualization
        if self.save_att:
            self.save_attn(attn)
        attn.register_hook(self.save_attgradients)

        x = rearrange((attn @ v),'b h n d -> b n (h d)')
        x = self.proj(x)
        return x
        
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: int = 768*4,
        act_layer: Union[nn.GELU, nn.SiLU] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768, 
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            save_att: bool = False,
            save_qkv: bool = False,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, save_att, save_qkv) 
        self.norm2 = norm_layer(embed_dim)
        self.mlp = mlp_layer(embed_dim, int(embed_dim * mlp_ratio), act_layer)
        
        
        self.input = None
        self.out = None


    def save_input(self, inp: torch.Tensor) -> None:
        self.input = inp

    def get_input(self) -> torch.Tensor:
        assert self.input is not None, "Please do forward pass before extracting input"
        return self.input
    
    def save_out(self, out: torch.Tensor) -> None:
        self.out = out

    def get_out(self) -> torch.Tensor:
        assert self.out is not None, "Please do forward pass before extracting output"
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.save_input(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # self.save_out(x)
        return x



class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            arch: Literal['vit', 'dino', 'mae'] = 'vit',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            save_att: bool = False,
            save_qkv: bool = False,
            qkv_bias: bool = False,
            n_last_blocks: int = 1,
            avgpool_patchtokens: bool = False,
            norm_layer: Optional[nn.Module] = nn.LayerNorm,
            mlp_layer: Optional[nn.Module] = Mlp,) -> None:
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.arch = arch
        self.num_heads = num_heads
        self.patch_embed = PatchEmbeddings(img_size, patch_size, embed_dim, in_chans)
        self.pos_embed = nn.Parameter(torch.randn(1,\
                                                   self.patch_embed.num_patches + 1,\
                                                      self.patch_embed.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,self.patch_embed.embed_dim))
        
        
        
        self.blocks = nn.Sequential(*[Block(
                                        embed_dim,
                                        num_heads,
                                        mlp_ratio,
                                        qkv_bias,
                                        save_att,
                                        save_qkv,  
                                        norm_layer=norm_layer,
                                        mlp_layer=mlp_layer
                                        ) for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim) if self.arch != 'mae' else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if self.arch == 'mae' else nn.Identity()       
        if self.arch == 'dino':
            self.n_last_blocks = n_last_blocks
            self.avgpool_patchtokens = avgpool_patchtokens
            embed_dim = embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            
        self.head = nn.Linear(embed_dim, num_classes)
        self.last_block_output = None


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0],-1,-1), x), dim=1)
        x = x + self.pos_embed        
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.arch == 'dino' and len(self.blocks) - i <= self.n_last_blocks:
                output.append(self.norm(x))
        if self.arch != 'dino':
            output = self.norm(x)
        return output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        
        self.last_block_output = self.forward_features(image)
        x = self.last_block_output

        if self.arch == 'mae':
            x = x[:, 1:].mean(dim=1)
        elif self.arch == 'dino':
            output = torch.cat([_x[:, 0] for _x in x], dim=-1)
            if self.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(x[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                x = output.reshape(output.shape[0], -1)
            else:
                x = output.reshape(output.shape[0], -1)
            # x = torch.cat((x[:, 0].unsqueeze(-1), x[:, 1:, :].mean(axis=1).unsqueeze(-1)), dim=-1)
            # x = x.reshape(x.shape[0], -1)
        else:
            x = x[:, 0]
        
        x = self.fc_norm(x)
        x = self.head(x)
        return x
        



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.arch = 'distilled'

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed

        x = self.blocks(x)

        x = self.norm(x)

        x, x_dist = x[:, 0], x[:, 1]
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2
