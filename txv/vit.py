from txv.models import VisionTransformer, DistilledVisionTransformer
from txv.lrp_models import LRPVisionTransformer, LRPDistilledVisionTransformer
from txv.utils import load_weights, getweights, combine_weights

def vit_small_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=768, depth=8, num_heads=6, mlp_ratio=3, qkv_bias= False,**kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('vit_small_patch16_224')))
    return model

def vit_base_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias= True,**kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('vit_base_patch16_224')))
    return model

def vit_large_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias= True,**kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('vit_large_patch16_224')))
    return model

def deit_small_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias= True,**kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('deit_small_patch16_224'),key='model'))
    return model

def deit_base_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias= True,**kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('deit_base_patch16_224'),key='model'))
    return model

def deit_tiny_distilled_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPDistilledVisionTransformer if lrp else DistilledVisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('deit_tiny_distilled_patch16_224'),key='model'))
    return model

def deit_small_distilled_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPDistilledVisionTransformer if lrp else DistilledVisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('deit_small_distilled_patch16_224'),key='model'))
    return model

def deit_base_distilled_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPDistilledVisionTransformer if lrp else DistilledVisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('deit_base_distilled_patch16_224'),key='model'))
    return model

# def deit_base_patch16_384(pretrained : bool =True, lrp : bool =False, **kwargs):
#     wrapper = LRPVisionTransformer if lrp else VisionTransformer
#     model = wrapper(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,\
#               qkv_bias=True, **kwargs)
#     if pretrained:
#         model.load_state_dict(load_weights(getweights('deit_base_patch16_384'),key='model'))
#     return model

# def deit_base_distilled_patch16_384(pretrained : bool =True, lrp : bool =False, **kwargs):
#     wrapper = LRPDistilledVisionTransformer if lrp else DistilledVisionTransformer
#     model = wrapper(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,\
#               qkv_bias=True, **kwargs)
#     if pretrained:
#         model.load_state_dict(load_weights(getweights('deit_base_distilled_patch16_384'),key='model'))
#     return model

def vit_mae_base_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        arch='mae', **kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('vit_mae_base_patch16_224'),key='model'))
    return model

def vit_mae_large_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,\
        arch='mae', **kwargs)
    if pretrained:
        model.load_state_dict(load_weights(getweights('vit_mae_large_patch16_224'),key='model'))
    return model

def vit_dino_small_patch8_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,\
            arch='dino', n_last_blocks=4, avgpool_patchtokens=False, **kwargs)
    if pretrained:
        model.load_state_dict(combine_weights(getweights('vit_dino_small_patch8_224'),\
                                                ['', 'state_dict'],[False, True],8))
    return model

def vit_dino_small_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,\
              arch='dino', n_last_blocks=4, avgpool_patchtokens=False, **kwargs)
    if pretrained:
        model.load_state_dict(combine_weights(getweights('vit_dino_small_patch16_224'),\
                                                ['', 'state_dict'],[False, True],16))
    return model

def vit_dino_base_patch8_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer
    model = wrapper(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,qkv_bias=True,\
        arch='dino', n_last_blocks=1, avgpool_patchtokens=True, **kwargs)
    if pretrained:
        model.load_state_dict(combine_weights(getweights('vit_dino_base_patch8_224'),\
                                                ['', 'state_dict'],[False, True],8))
    return model

def vit_dino_base_patch16_224(pretrained : bool =True, lrp : bool =False, **kwargs):
    wrapper = LRPVisionTransformer if lrp else VisionTransformer    
    model = wrapper(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias= True,\
        arch='dino', n_last_blocks=1, avgpool_patchtokens=True, **kwargs)
    if pretrained:
        model.load_state_dict(combine_weights(getweights('vit_dino_base_patch16_224'),\
                                                ['', 'state_dict'],[False, True],16))
    return model
