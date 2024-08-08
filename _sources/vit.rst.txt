vit
===

Available models
----------------

ViT-based models:

* ``vit_small_patch16_224()``
* ``vit_base_patch16_224()``
* ``vit_large_patch16_224()``

DeiT-based models:

* ``deit_small_patch16_224()``
* ``deit_base_patch16_224()``
* ``deit_tiny_distilled_patch16_224()``
* ``deit_small_distilled_patch16_224()``
* ``deit_base_distilled_patch16_224()``

MAE-based models:

* ``vit_mae_base_patch16_224()``
* ``vit_mae_large_patch16_224()``

DINO-based models:

* ``vit_dino_small_patch8_224()``
* ``vit_dino_small_patch16_224()``
* ``vit_dino_base_patch8_224()``
* ``vit_dino_base_patch16_224()``

Initialization of above models:

.. function:: model(pretrained : bool = True, lrp : bool =False)
    
        Returns a ViT model.    
    
        :param pretrained: (bool) - If True, returns a model pre-trained on ImageNet.
        :param lrp: (bool) - If True, returns a model with Layer-wise Relevance Propagation (LRP) enabled.

        Example:
        ~~~~~~~~    
    
            >>> from txv.vit import vit_base_patch16_224
            >>> model = vit_base_patch16_224(pretrained=True)
            >>> model.eval()
