from typing import Literal
import torch
import warnings
import torch.nn.functional as F
from einops import einsum as eins
from txv.models import *
from txv.utils import *


class LRP:
    """
    Layer-wise Relevance Propagation(LRP).
    Link to the paper: `On Pixel-Wise Explanations for Non-Linear 
    Classifier Decisions by Layer-Wise Relevance Propagation 
    <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140>`_
    """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`    

        
        .. caution::
            The model must be an LRP model. You can use the LRP version of a model by passing ``lrp=True`` in the model function.
        """
        self.model = model
        assert hasattr(model, 'relprop'), "Use LRP models"

    def explain(self, 
                input: torch.Tensor, 
                index: int = None,
                alpha: float = 0.5,
                abm: bool = True,
                ) -> torch.Tensor:
        """
        Explain the model prediction using Layer-wise Relevance Propagation(LRP)
        
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        alpha : float, optional
            Alpha value for LRP, by default 0.5
        abm : bool, optional    
            Architecture based modification, by default True
        """
        output = self.model(input)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            index = [index]
        
        self.model.zero_grad()
        
        one_hot_vector = torch.zeros(output.shape).to(input.device)
        one_hot_vector[:,index] = 1
        cam = self.model.relprop(one_hot_vector, alpha)
        if not abm:
            return cam
        else:
            if self.model.arch == 'distilled':
                return cam[:,2:].mean(-1)
            else:
                return cam[:,1:].mean(-1)

class IntegratedGradients:
    """
    Link to paper: `Axiomatic Attribution for Deep Networks <https://arxiv.org/abs/1703.01365>`_
    """
    def __init__(self, model: torch.nn.Module):
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models for IG is not recommended as they have higher memory footprint")

    
    def explain(self, 
                input: torch.Tensor,
                index: int = None,
                steps: int = 20,
                baseline: torch.Tensor = None,
       )-> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        steps : int, optional
            Number of steps in Riemann approximation of integral, by default 20
        baseline : torch.Tensor, optional
            Baseline tensor, by default None(tensor of zeros)
        """
        
        if baseline is None:
            baseline = torch.zeros_like(input).to(input.device)
        assert baseline.shape == input.shape, "Baseline and input shapes must match"

        
        alphas = torch.linspace(0, 1, steps=steps+1)[1:].to(input.device)
        alphas = alphas.view(-1, 1, 1, 1)
        explanation = torch.zeros_like(input).to(input.device)

        for alpha in alphas:
            outs = baseline + alpha * (input - baseline)
            outs.requires_grad = True
            output = self.model(outs)
            accum_grad(output, self.model, index)
            gradients = outs.grad.squeeze()
            explanation += gradients
        explanation = explanation / steps
        explanation = explanation * (input - baseline)
        explanation = explanation.sum(dim=1)
        explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min())
        return explanation
            
class RawAttention:
    """
    Basic Attention Visualization. 
    This is a class-agnostic explanation method. Therefore, an index cannot be passed as an argument.
    """
    def __init__(self, model: torch.nn.Module):
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`
            
        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, input: torch.Tensor, layer: int = 0)-> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        layer : int, optional
            Layer number to visualize, by default 0. 0 :math:`\leq` layer :math:`\leq` model.depth - 1

        Returns
        -------
        Returns attention map of the specified layer. Dimensions are (batch_size, num_heads, num_tokens, num_tokens)
        

        .. note::
            Perform necessary post-processing operations to visualize the attention map. 
            Take proper care in choosing between [CLS] token and other tokens.
        """
        assert layer < self.model.depth and layer>=0,\
              "Layer number must be less than total layers and greater than 0"
        self.model.blocks[layer].attn.save_att = True
        self.model(input)
        att = self.model.blocks[layer].attn.get_attn()
        return att

class AttentionRollout:
    """
    Link to Paper: `Quantifying Attention Flow in Transformers <https://arxiv.org/abs/2005.00928>`_
    This is a class-agnostic explanation method. Therefore, an index cannot be passed as an argument.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, input: torch.Tensor,layer: int = 0, abm: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        layer : int, optional
            Layer number to start the computation of rollout, by default 0. 
            0 :math:`\leq` layer :math:`\leq` model.depth - 1
        abm : bool, optional
            Architecture based modification, by default True
        """
        for i in range(layer, self.model.depth):
            self.model.blocks[i].attn.save_att = True

        self.model(input)
        
        b = input.shape[0]
        if self.model.arch == 'distilled':
            num_patches = self.model.patch_embed.num_patches + 2
        else:
            num_patches = self.model.patch_embed.num_patches + 1

        explanation = torch.eye(num_patches).expand(b,-1,-1).to(input.device)
        
        for i in range(layer, self.model.depth):
            A = self.model.blocks[i].attn.get_attn().clamp(min=0).mean(dim=1)
            A = A + torch.eye(A.shape[1]).to(input.device)
            A = A / A.sum(dim=-1, keepdim=True)
            explanation = torch.bmm(A, explanation)
            
        if not abm:
            return explanation  
        else:
            if self.model.arch == 'mae':
                return explanation[:,1:,1:].mean(axis=1)
            elif self.model.arch == 'dino':
                return explanation[:,1:,1:].mean(axis=1) + explanation[:,0,1:]
            elif self.model.arch == 'distilled':
                return explanation[:,0,2:] + explanation[:,1,2:]
            else:
                return explanation[:,0,1:]

class GradSAM:
    """
    Link to Paper: `Grad-SAM: Explaining Transformers 
    via Gradient Self-Attention Maps <https://arxiv.org/abs/2204.11073>`_
    """
    def __init__(self, model: torch.nn.Module)-> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, input: torch.Tensor, index: int = None, abm: bool = True)-> torch.Tensor:
        """        
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        abm : bool, optional
            Architecture based modification, by default True
        """
        for i in range(self.model.depth):
            self.model.blocks[i].attn.save_att = True

        output = self.model(input)
        accum_grad(output, self.model, index)

        if self.model.arch == 'distilled':
            num_patches = self.model.patch_embed.num_patches + 2
        else:
            num_patches = self.model.patch_embed.num_patches + 1

        heads = self.model.blocks[0].attn.num_heads
        explanation = torch.zeros((input.shape[0],self.model.depth,heads,num_patches, num_patches)).to(input.device)

        for i in range(self.model.depth):
            att = self.model.blocks[i].attn.get_attn()
            attgrad = self.model.blocks[i].attn.get_attgradients()
            explanation[:,i] = att * F.relu(attgrad)

        explanation = explanation.mean(dim=[1,2]).clamp(min=0)
        explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min())
        
        if not abm:
            return explanation
        else:
            if self.model.arch == 'mae':
                return explanation[:,1:,1:].mean(axis=1)
            elif self.model.arch == 'dino':
                return explanation[:,1:,1:].mean(axis=1) + explanation[:,0,1:]
            elif self.model.arch == 'distilled':
                return explanation[:,0,2:] + explanation[:,1,2:]
            else:
                return explanation[:,0,1:]      

class BeyondAttention:
    """Link to Paper: `Transformer Interpretability 
    Beyond Attention Visualization <https://arxiv.org/abs/2012.09838>`_ """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. caution::
            The model must be an LRP model. You can use the LRP version of a model by passing ``lrp=True`` in the model function.
        """
        self.model = model
        assert hasattr(model, 'relprop'), "Use LRP models"

    def explain(self,
                input: torch.Tensor, 
                index: int = None, 
                alpha: float = 1.0,
                abm: bool = True,
                **kwargs)-> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        alpha : float, optional
            Alpha value for LRP, by default 1.0
        abm : bool, optional
            Architecture based modification, by default True
        """
        for i in range(self.model.depth):
            self.model.blocks[i].attn.save_rel = True

        output = self.model(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            
        self.model.zero_grad()
        one_hot_vector = torch.zeros_like(output).to(input.device)
        one_hot_vector[:,index] = 1
        accum_grad(output, self.model, index)
        self.model.relprop(one_hot_vector, alpha, **kwargs)

        cams = []

        for i in range(self.model.depth):
            A = self.model.blocks[i].attn.get_attn_cam() * self.model.blocks[i].attn.get_attgradients() 
            A = A.clamp(min=0).mean(dim=1)
            cams.append(A)

        C = compute_rollout_attention(cams)

        if not abm:
            return C
        else:
            if self.model.arch == 'mae':
                return C[:,1:,1:].mean(axis=1)
            elif self.model.arch == 'dino':
                return C[:,1:,1:].mean(axis=1) + C[:,0,1:]
            elif self.model.arch == 'distilled':
                return C[:,0,2:] + C[:,1,2:]
            else:
                return C[:,0,1:]

class GenericAttention:
    """
    Link to Paper: `Generic Attention-model Explainability 
    for Interpreting Bi-Modal and Encoder-Decoder 
    Transformers <https://arxiv.org/abs/2103.15679>`_
    """
    def __init__(self, model):
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, input: torch.Tensor, 
                      index: int = None, 
                      layer: int = 1, 
                      abm: bool = True,
                      **kwargs)-> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        layer : int, optional
            Layer number to start the computation of attention weights, by default 1
        abm : bool, optional
            Architecture based modification, by default True

        """
        for i in range(self.model.depth):
            if i < layer-1:
                continue
            self.model.blocks[i].attn.save_att = True
            self.model.blocks[i]

        output = self.model(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            
        self.model.zero_grad()
        one_hot_vector = torch.zeros_like(output).to(input.device)
        one_hot_vector[:,index] = 1
        accum_grad(output, self.model, index)

        b = input.shape[0]
        
        _, _, num_tokens, _ = self.model.blocks[-1].attn.get_attn().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(input.device)
        for nb, blk in enumerate(self.model.blocks):
            if nb < layer-1:
                continue
                
            cam = blk.attn.get_attn()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            grad = blk.attn.get_attgradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])

            cam = (grad*cam).mean(0).clamp(min=0)

            R = R + torch.matmul(cam, R)  
        
        if not abm:
            return R
        else:
            if self.model.arch == 'mae':
                return R[:, 1:, 1:].mean(axis=1)
            elif self.model.arch == 'distilled':
                return R[:,0,2:]
            else:
                return R[:, 0, 1:]
        
class TAM:
    """
    Link to paper: `Explaining Information Flow Inside 
    Vision Transformers Using Markov 
    Chain <https://openreview.net/forum?id=TT-cf6QSDaQ>`_
    """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.

            
        .. caution::
            This method is only supported for Vision Transformers- small, base and large.
        """
        self.model = model
        assert model.arch == 'vit', "TAM is only supported for ViT"
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, 
                input: torch.Tensor, 
                index: int = None,
                l_end: int = 0,                
                steps: int = 20,
                baseline: torch.Tensor = None,
                abm: bool = True,
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        l_end : int, optional
            Layer number to end the computation of attention weights. By default 0. Attention weights are computed from last_layer to l_end.
        steps : int, optional
            Number of steps in Riemann approximation of integral, by default 20
        baseline : torch.Tensor, optional
            Baseline tensor, by default None(tensor of zeros)
        """
        for i in range(self.model.depth):
            self.model.blocks[i].attn.save_att = True

        output = self.model(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)


        states = self.model.blocks[-1].attn.get_attn().mean(dim=1)[:,0,:].reshape(input.shape[0],1,-1)
        for i in reversed(range(l_end, self.model.depth-1)):
            attn = self.model.blocks[i].attn.get_attn().mean(1)

            states_ = states
            states = states.bmm(attn)
            states += states_
        
        alphas = torch.linspace(0, 1, steps).to(input.device)
        alphas = alphas.view(-1, 1, 1, 1)
        baseline = torch.zeros_like(input).to(input.device) if baseline is None else baseline
        outs = baseline + (alphas * (input - baseline))
        

        num_tokens = self.model.patch_embed.num_patches+1
        Wc = torch.zeros((self.model.num_heads, num_tokens, num_tokens)).to(input.device)
        for inp in outs:
            output = self.model(inp.unsqueeze(0))
            accum_grad(output,self.model, index)
            Wc += self.model.blocks[-1].attn.get_attgradients().squeeze()
        

        Wc = Wc/steps
        Wc = Wc.clamp(min=0)
        Wc = Wc.unsqueeze(0)
        
        Wc = Wc.mean(1)[:,0,:].reshape(input.shape[0],1,-1)
        
        Rc = Wc * states
    
        if not abm:
            return Rc
        else:
            return Rc[:,0,1:]

class BeyondIntuition:
    """
    Link to paper: `Beyond Intuition: Rethinking 
    Token Attributions inside Transformers 
    <https://openreview.net/forum?id=rm0zIzlhcX>`_
    """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            A model from :code:`txv.vit`

        
        .. tip::
            Use the model with :code:`lrp=False` as LRP models have higher memory footprint.
        """
        self.model = model
        if hasattr(model, 'relprop'):
            warnings.warn("Using LRP models is not recommended as they have higher memory footprint")

    def explain(self, 
                input: torch.Tensor, 
                method: Literal['head','token'] = 'head',
                index: int = None,
                layer: int = 0,   
                steps: int = 20, 
                baseline: torch.Tensor = None,
                abm: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input tensor
        method : Literal['head','token'], optional
            Type of attention map: head-wise or token-wise, by default 'head'
        index : int, optional
            Index of the class to explain, by default the predicted class is explained
        layer : int, optional
            Layer number to start the computation of attention weights, by default 0
        steps : int, optional
            Number of steps in Riemann approximation of integral, by default 20
        baseline : torch.Tensor, optional   
            Baseline tensor, by default None(tensor of zeros)
        abm : bool, optional
            Architecture based modification, by default True
        """

        assert method in ['head','token'], "Method must be either head or token"


        if self.model.arch == 'distilled':
            num_patches = self.model.patch_embed.num_patches + 2
        else:
            num_patches = self.model.patch_embed.num_patches + 1
        b = input.shape[0]
        P = torch.eye(num_patches, num_patches).expand(b, num_patches, num_patches).to(input.device)

        for i in range(layer,self.model.depth):
                self.model.blocks[i].attn.save_att = True

        # Head Wise
        if method == 'head':            
            
            for i,block in enumerate(self.model.blocks):
                if i < layer:
                    continue
                
                output = self.model(input)

                accum_grad(output,self.model, index)
                A = block.attn.get_attn()
                grad = block.attn.get_attgradients()
                Ih = torch.matmul(A.transpose(-1,-2), grad).abs().mean(dim=(-1,-2))
                Ih = Ih/torch.sum(Ih)
                A = eins(Ih, A, 'b h , b h i j -> b i j')
                P = P + torch.matmul(A, P)


        # Token Wise
        if method == 'token':
            
            for i in range(layer,self.model.depth):
                self.model.blocks[i].attn.issavev = True

            output = self.model(input)

            for i,block in enumerate(self.model.blocks):
                if i < layer:
                    continue
                
                z = block.get_input()
                vproj = rearrange(block.attn.get_v(), 'b h n d -> b n (h d)')
                
                vproj = torch.matmul(vproj , block.attn.proj.weight.t())                
                
                alpha = torch.linalg.norm(vproj, dim=-1)/torch.linalg.norm(z, dim=-1).squeeze()
                
                m = torch.diag_embed(alpha)
                A = block.attn.get_attn().mean(1)
                A = torch.matmul(A,m)
                P += torch.matmul(A, P)
            P = P.abs()


        alphas = torch.linspace(0, 1, steps=steps+1)[1:].to(input.device)
        alphas = alphas.view(-1, 1, 1, 1)
        baseline = torch.zeros_like(input).to(input.device) if baseline is None else baseline
        outs = baseline + alphas * (input - baseline)
        outs.requires_grad = True

        for i in range(layer,self.model.depth):
                self.model.blocks[i].attn.issavev = False

        Fc = torch.zeros((self.model.num_heads, num_patches, num_patches)).to(input.device)
        for inp in outs:
            output = self.model(inp.unsqueeze(0))
            accum_grad(output,self.model, index)
            Fc += self.model.blocks[-1].attn.get_attgradients().squeeze()
        # output = self.model(outs)

        # accum_grad(output,self.model, index)

        # print(self.model.blocks[-1].attn.get_attgradients().shape)
        # Fc = self.model.blocks[-1].attn.get_attgradients().sum(dim=0, keepdim=True)/steps
        Fc = Fc.unsqueeze(0)


        Fc = Fc/steps
        Fc = Fc.clamp(min=0)
        Fc = (Fc - Fc.min()) / (Fc.max() - Fc.min())  
        Fc = F.relu(Fc).mean(1)
        Fc = Fc * P

        if not abm:
            return Fc
        else:
            if self.model.arch == 'mae':
                return Fc[:,1:,1:].mean(axis=1)
            elif self.model.arch == 'dino':
                return Fc[:,1:,1:].mean(axis=1) + Fc[:,0,1:]
            elif self.model.arch == 'distilled':
                return Fc[:,0,2:]
            else:
                return Fc[:,0,1:]