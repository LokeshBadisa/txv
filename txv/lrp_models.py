from math import sqrt
from txv.lrp_modules import *
from einops import rearrange
from typing import Literal, Optional, Type, Union
import torch
import torch.nn as nn
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
        self.proj = Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim        

    def forward(self,image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        assert H == self.image_size and W == self.image_size, f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"
        assert self.image_size % self.patch_size == 0, f"Input dimensions {self.image_size}x{self.image_size} not divisble for patch size {self.patch_size}"
        
        return rearrange(self.proj(image),'b c h w -> b (h w) c')
    
    def relprop(self, cam, alpha, **kwargs):
        cam = rearrange(cam,'b (h w) c -> b c h w',h=int(sqrt(self.num_patches)))
        cam = self.proj.relprop(cam, alpha, **kwargs)

class Attention(nn.Module):
    def __init__(self,
                embed_dim: int = 768,
                num_heads: int = 12,
                qkv_bias: bool = False, 
                save_att: bool = True,
                save_qkv: bool = True,
                save_rel: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Linear(embed_dim, embed_dim * 3, qkv_bias)
        self.softmax = Softmax(dim=-1)
        self.proj = Linear(embed_dim, embed_dim)
        self.save_att = save_att
        self.save_qkv = save_qkv
        self.save_rel = save_rel
        self.issaveq = save_qkv
        self.issavek = save_qkv
        self.issavev = save_qkv

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')
        

        self.attn_gradients = None
        self.attention_map = None
        self.relevance = None
        
        self.input = None       
        self.q = None
        self.k = None
        self.v = None
        self.output = None

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
    
    def save_output(self, out: torch.Tensor) -> None:
        self.output = out

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Please do forward pass before extracting output"
        return self.output
    
    # def save_v_cam(self, cam):
    #     self.v_cam = cam

    # def get_v_cam(self):
    #     return self.v_cam
    
    def save_attn_cam(self, cam):
        self.relevance = cam

    def get_attn_cam(self):
        return self.relevance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv,'b n (qkv h d) -> qkv b h n d',qkv=3,h=self.num_heads)
        
        # Save q,k,v for visualization
        if self.issaveq:
            self.save_q(q)
        if self.issavek:
            self.save_k(k)
        if self.issavev:
            self.save_v(v)

        attn = self.matmul1([q,k]) * self.scale
        attn = self.softmax(attn)

        # Save attention map for visualization
        if self.save_att:
            self.save_attn(attn)
        attn.register_hook(self.save_attgradients)

        x = rearrange(self.matmul2([attn,v]),'b h n d -> b n (h d)')
        x = self.proj(x)
        return x
    
    def relprop(self, cam, alpha,**kwargs):
        cam = self.proj.relprop(cam, alpha,**kwargs)
        cam = rearrange(cam,'b n (h d) -> b h n d',h=self.num_heads)
        (cam1, cam_v)= self.matmul2.relprop(cam, alpha, **kwargs)
        cam1 /= 2
        cam_v /= 2

        # self.save_v_cam(cam_v)
        if self.save_rel:
            self.save_attn_cam(cam1)

        cam1 = self.softmax.relprop(cam1, alpha, **kwargs)
        (cam_q, cam_k) = self.matmul1.relprop(cam1, alpha, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, alpha, **kwargs)
    
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.mul = einsum('b n d, d -> b n d')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mul([x, self.gamma])
    
    def relprop(self, cam, alpha,**kwargs):
        cam = self.mul.relprop(cam, alpha,**kwargs)
        return cam

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: int = 768*4,
        act_layer: Union[GELU,SiLU] = GELU,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def relprop(self, cam, alpha,**kwargs):
        cam = self.fc2.relprop(cam, alpha,**kwargs)
        cam = self.act.relprop(cam, alpha,**kwargs)
        cam = self.fc1.relprop(cam, alpha,**kwargs)
        return cam
                                                                                         
class Block(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768, 
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            save_att: bool = False,
            save_qkv: bool = False,
            save_rel: bool = False,
            act_layer: nn.Module = GELU,
            norm_layer: nn.Module = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, save_att, save_qkv, save_rel) 
        self.norm2 = norm_layer(embed_dim)
        self.mlp = mlp_layer(embed_dim, int(embed_dim * mlp_ratio), act_layer)
        
        
        self.input = None
        self.out = None

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()


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
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        # self.save_out(x)
        return x
    
    def relprop(self, cam, alpha, **kwargs):
        cam = self.add2.relprop(cam, alpha,**kwargs)
        cam = self.mlp.relprop(cam, alpha,**kwargs)
        cam = self.norm2.relprop(cam, alpha,**kwargs)
        cam = self.add1.relprop(cam, alpha,**kwargs)
        cam = self.attn.relprop(cam, alpha,**kwargs)
        cam = self.norm1.relprop(cam, alpha,**kwargs)
        return cam
        
class LRPVisionTransformer(nn.Module):
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
            save_rel: bool = False,
            qkv_bias: bool = False,  
            n_last_blocks: Optional[int] = None,
            avgpool_patchtokens: bool = False,       
            norm_layer: Optional[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp) -> None:
        super().__init__()
        self.depth = depth
        assert arch in ['vit', 'dino', 'mae'], "Architecture must be either 'vit', 'dino' or 'mae'"
        self.arch = arch
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        
        self.patch_embed = PatchEmbeddings(img_size, patch_size, embed_dim, in_chans)
        self.pos_embed = nn.Parameter(torch.randn(1,\
                                                   self.patch_embed.num_patches + 1,\
                                                      self.patch_embed.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,self.patch_embed.embed_dim))
        

        self.blocks = Sequential(*[Block(
                                        embed_dim,
                                        num_heads,
                                        mlp_ratio,
                                        qkv_bias,
                                        save_att,
                                        save_qkv,
                                        save_rel,
                                        norm_layer=norm_layer,
                                        mlp_layer=mlp_layer
                                        ) for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim) if self.arch != 'mae' else Identity()
        self.fc_norm = norm_layer(embed_dim) if self.arch == 'mae' else Identity() 
        
        if self.arch == 'dino':
            self.n_last_blocks = n_last_blocks
            self.avgpool_patchtokens = avgpool_patchtokens
            embed_dim = embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            self.pools = [IndexSelect() for _ in range(n_last_blocks+int(avgpool_patchtokens))]
            self.head_dim = embed_dim
            if self.avgpool_patchtokens:
                self.pool2 = IndexSelect()             
                self.mean = Mean()
                self.pool3 = IndexSelect()
        else:
            if n_last_blocks is not None or avgpool_patchtokens:
                raise Exception("n_last_blocks and avgpool_patchtokens are only for DINO model")
            self.pool1 = IndexSelect()

        self.head = Linear(embed_dim, num_classes)
        self.last_block_output = None

        self.add = Add()
        self.cat = Cat()
        if self.arch == 'dino':
            self.cat2 = Cat()
            if self.avgpool_patchtokens:
                self.cat3 = Cat()
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.cat((self.cls_token.expand(x.shape[0],-1,-1), x), dim=1)
        x = self.add([x, self.pos_embed])
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.arch == 'dino' and len(self.blocks) - i <= self.n_last_blocks:
                output.append(self.norm(x))
        if self.arch != 'dino':
            output = self.norm(x) 
        return output
    
    def forward_features_relprop(self, cam, alpha, **kwargs):
        if self.arch != 'dino':
            cam = self.norm.relprop(cam, alpha, **kwargs)
            cam = self.blocks.relprop(cam, alpha, **kwargs) 
        else:
            propagating_cam = torch.zeros_like(cam[0],device=cam[0].device)
            for i, blk in enumerate(reversed(self.blocks)):
                if i < self.n_last_blocks:
                    propagating_cam = blk.relprop(propagating_cam+cam[-i-1], alpha, **kwargs)
                else:
                    propagating_cam = blk.relprop(propagating_cam, alpha, **kwargs)
            cam = propagating_cam
        # cam = self.add.relprop(cam, alpha, **kwargs)
        # cam = self.cat.relprop(cam, alpha, **kwargs)
        # cam = cam[:,1:]
        # cam = self.patch_embed.relprop(cam, alpha, **kwargs)
        return cam

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        
        x = self.forward_features(image)
        self.last_block_output = x
        
        if self.arch == 'mae':
            x = self.pool1(x, dim=1, indices=torch.arange(1,x.shape[1], device=x.device)).mean(dim=1)
        elif self.arch == 'dino':     
            new_output = self.cat2([self.pools[i](output, dim=1, \
                        indices=torch.tensor(0, device=x[0].device)) for i, output in enumerate(x)], dim=-1)
            
            if self.avgpool_patchtokens:
                x = torch.stack(x,dim=0)
                another = self.pool3(x,dim=0,indices=torch.tensor(self.n_last_blocks-1).to(x[-1].device))                
                output = self.cat3([new_output.squeeze(1).unsqueeze(-1), self.mean(self.pool2(another.squeeze(1),\
                                dim=1, indices=torch.arange(1,x[-1].shape[1], device=x[-1].device)),\
                                dim=1).unsqueeze(-1)], dim=-1)
                x = output.reshape(output.shape[0], -1)
            else:
                x = new_output.reshape(new_output.shape[0], -1)
        else:
            x = self.pool1(x,dim=1,indices= torch.tensor(0,device=x.device)).squeeze(1)
        
        x = self.fc_norm(x)        
        x = self.head(x)

        return x
    
    def forward_relprop(self, cam, alpha, **kwargs):
        cam = self.head.relprop(cam, alpha, **kwargs)
        cam = self.fc_norm.relprop(cam, alpha, **kwargs)
        if self.arch != 'dino':
            cam = self.pool1.relprop(cam, alpha, **kwargs)
        else:
            if self.avgpool_patchtokens:
                cam = cam.reshape(cam.shape[0],self.embed_dim,self.n_last_blocks+int(self.avgpool_patchtokens)) 
                cam = self.cat3.relprop(cam, alpha, **kwargs)
                cam1, cam2 = cam[0], cam[1].reshape(cam[1].shape[0],self.embed_dim)
                cam2 = self.mean.relprop(cam2, alpha, **kwargs)
                cam2 = self.pool2.relprop(cam2, alpha, **kwargs)
                cam2 = self.pool3.relprop(cam2, alpha, **kwargs)
                cam = cam1.squeeze(-1)               

            
            cam = self.cat2.relprop(cam.unsqueeze(1), alpha, **kwargs) 
            cams = []
            for i in range(self.n_last_blocks):
                cam1 = self.pools[i].relprop(cam[i], alpha, **kwargs)
                cams.append(cam1)
            
        if self.arch == 'dino':
            if self.avgpool_patchtokens:
                return torch.stack(cams)+ cam2
            else:
                return cams
        else:
            return cam
                   
    def relprop(self, cam, alpha, **kwargs):
        cam = self.forward_relprop(cam, alpha, **kwargs)
        cam = self.forward_features_relprop(cam, alpha, **kwargs)
        return cam
         
class LRPDistilledVisionTransformer(LRPVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else Identity()
        self.pool1 = IndexSelect()
        self.pool2 = IndexSelect()
        self.add2 = Add()
        self.arch = 'distilled'

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = self.add([x, self.pos_embed])

        x = self.blocks(x)
        self.last_block_output = x
        x = self.norm(x)

        # x, x_dist = x[:, 0], x[:, 1]
        x_dist = self.pool2(x,dim=1,indices= torch.tensor(1,device=x.device)).squeeze(1)
        x = self.pool1(x,dim=1,indices= torch.tensor(0,device=x.device)).squeeze(1)        
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return self.add2([x,x_dist])/ 2
    
    def relprop(self, cam, alpha, layer=0, **kwargs):
        cam = self.add2.relprop(cam, alpha,**kwargs)
        cam1 = self.head_dist.relprop(cam, alpha,**kwargs)
        cam2 = self.head.relprop(cam,alpha,**kwargs)
        cam1 = self.pool1.relprop(cam1, alpha,**kwargs)
        cam2 = self.pool2.relprop(cam2, alpha,**kwargs)
        cam = (cam1+cam2)/2
        cam = self.norm.relprop(cam, alpha,**kwargs)
        for block in reversed(self.blocks[layer:]):
            cam = block.relprop(cam, alpha, **kwargs)
        # cam = self.add.relprop(cam, alpha, **kwargs)
        # cam = cam[:,2:]
        # cam = self.patch_embed.relprop(cam, alpha, **kwargs)
        return cam
