import numpy as np
import torch
import torch.nn.functional as F
from  torch.nn.modules.upsampling import Upsample
from utils.constants import Constants as c
from captum.attr import DeepLift, GuidedGradCam, InputXGradient, IntegratedGradients, KernelShap, Lime, Saliency, NoiseTunnel, GuidedBackprop, Deconvolution, FeatureAblation, Occlusion, ShapleyValueSampling, ShapleyValues


def batch_wise_attributions(attr, x, target, batch_size, **kwargs):
    attributions = torch.zeros_like(x)
    num_samples = x.shape[0]

    for i in range(0, num_samples, batch_size):
        end = min(num_samples, i + batch_size)
        if i >= end:
            break
        x_batch = x[i:end]
        target_batch = target[i:end]
        attributions[i:end] = attr(x_batch,target=target_batch, **kwargs)
    return attributions


class DeepLIFTCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.am = DeepLift(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)
        attribution = self.am.attribute(x, target=target)
        return attribution
    

class SaliencyCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.am = Saliency(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.am.attribute(x, target=target)

        return attribution



class GuidedGradCAMCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

        layers = list(model.layers)
        gap_preceding_layer = None
        for i in range(len(layers)):
            l = layers[i]
            if l.__class__.__name__ == 'GlobalAveragePooling':
                gap_preceding_layer = layers[i-1]
                break
        if gap_preceding_layer is None:
            raise Exception('Not supported model - Model has to have a gap layer')


        self.ggc = GuidedGradCam(self.model, gap_preceding_layer)

    def attribute(self, inpt, target=None):
        
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.ggc.attribute(x, target=target)

        return attribution



class InputXGradientCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.ixg = InputXGradient(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.ixg.attribute(x, target=target)

        return attribution


class IntegratedGradientsCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    def attribute(self, inpt, target=None):
        
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.ig.attribute(x, target=target, internal_batch_size=64)

        return attribution



class KernelShapCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = KernelShap(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.explainer.attribute(x, target=target, perturbations_per_eval=12)

        return attribution



class LimeCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = Lime(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)

        x = (inpt.data).requires_grad_(True)

        attribution = self.explainer.attribute(x, target=target, perturbations_per_eval=12)

        return attribution



class RandomAttribution():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def attribute(self, inpt, target=None):
        return torch.rand(inpt.size())








class GradCAM(): # Own implementation
    def __init__(self, model):
        super().__init__()
        self.model = model

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        
        if inpt.is_cuda:
            target = target.cuda()

        if not hasattr(self.model, 'layers') and not hasattr(self.model.layers, 'gap'):
            print('Relevance computation not supported. Model has to have "layers" attribute and "layers" has to contain "gap" layer' )
            return torch.zeros_like(inpt)

        self.model.eval()
        x = (inpt.data).requires_grad_(True)

        # Forward pass until GAP
        for i in range(len(self.model.layers)):
            if i == len(self.model.layers) - 1:
                continue
            x = self.model.layers[i](x)

        target_fmaps = (x.data).requires_grad_(True)

        # GAP
        x2 = self.model.layers.gap(target_fmaps) # gap layer

        # out
        x2 = self.model.out(x2)

        mask = torch.zeros_like(x2)
        if x2.is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.reshape(-1,1), 1.)
        x2.backward(mask) # calc gradient of logit of prediction 

        # The gradients have to be calculated before the softmax activation    
        gradients = target_fmaps.grad.data # 

        #  Get importance vector by global average pool of feature maps 
        pooled_gradients = torch.mean(gradients, dim=[2])

        # Combine feature maps using importance weights
        activations = target_fmaps.data
        for i in range(pooled_gradients.size()[1]):
            activations[:,i,:] *= pooled_gradients[:,i].reshape(-1,1)

        # Sum up activations
        activations = torch.sum(activations, 1)

        # Apply ReLU
        coarse_heatmap = F.relu(activations)

        # Upsample
        scaling_factor = inpt.size()[2] / coarse_heatmap.size()[1]
        m = Upsample(scale_factor=scaling_factor, mode='linear', align_corners=True)

        attributions = m(coarse_heatmap.view(coarse_heatmap.size()[0],1,coarse_heatmap.size()[1]))

        return attributions


class GuidedBackpropCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = GuidedBackprop(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)
        attribution = self.explainer.attribute(x, target=target)
        return attribution


class DeconvolutionCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = Deconvolution(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)
        attribution = self.explainer.attribute(x, target=target)
        return attribution

class FeatureAblationCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.am = FeatureAblation(self.model)
        self.feature_mask = None

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)

        if self.feature_mask is None or self.feature_mask.shape != x.shape:
            self.feature_mask = torch.zeros_like(x).int()
            area_id = 0
            window = 5
            for i in range(self.feature_mask.shape[-1]):
                if (i % window) == 0:
                    area_id += 1
                self.feature_mask[:,:,i] = area_id

        attribution = self.am.attribute(x, target=target, feature_mask=self.feature_mask, perturbations_per_eval=12)
        return attribution


class OcclusionCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = Occlusion(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)
        attribution = self.explainer.attribute(x, target=target, sliding_window_shapes=(1,1))
        return attribution



class ShapleyValueSamplingCaptum():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.explainer = ShapleyValueSampling(self.model)

    def attribute(self, inpt, target=None):
        if target is None:
            target = torch.exp(self.model(inpt)).argmax(dim=1)
        x = (inpt.data).requires_grad_(True)
        attribution = self.explainer.attribute(x, target=target)
        return attribution

