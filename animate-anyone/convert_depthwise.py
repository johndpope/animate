import time
from os.path import join

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionImageVariationPipeline, StableDiffusionPipeline
import copy
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel, Transformer2DModel
from einops import rearrange
from xformers.ops import memory_efficient_attention
from models.motionmodule import get_motion_module
import tensorboard
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(17)

import pkg_resources

for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print("tensorboard_plugins:",entry_point.dist)

class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
    
class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
    

# VideoNet is a unet initialized from stable diffusion used to denoise video frames
class VideoNet(nn.Module):
    def __init__(self, sd_unet: UNet2DConditionModel, num_frames: int = 24, batch_size: int = 2):
        super(VideoNet, self).__init__()
        self.batch_size = batch_size

        # create a deep copy of the sd_unet
        self.unet = copy.deepcopy(sd_unet)

        # maintain a list of all the new ReferenceConditionedResNets and TemporalAttentionBlocks
        self.ref_cond_attn_blocks: List[ReferenceConditionedAttentionBlock] = []

        # replace attention blocks with ReferenceConditionedAttentionBlock
        down_blocks = self.unet.down_blocks
        mid_block = self.unet.mid_block
        up_blocks = self.unet.up_blocks

        for i in range(len(down_blocks)):
            if hasattr(down_blocks[i], "attentions"):
                attentions = down_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

        for i in range(len(mid_block.attentions)):
            mid_block.attentions[i] = ReferenceConditionedAttentionBlock(mid_block.attentions[i], num_frames)
            self.ref_cond_attn_blocks.append(mid_block.attentions[i])
        
        for i in range(len(up_blocks)):
            if hasattr(up_blocks[i], "attentions"):
                attentions = up_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

    # update_reference_embeddings updates all the reference embeddings in the unet
    def update_reference_embeddings(self, reference_embeddings):
        if len(reference_embeddings) != len(self.ref_cond_attn_blocks):
            print("[!] WARNING - amount of input reference embeddings does not match number of modules in VideoNet")

        for i in range(len(self.ref_cond_attn_blocks)):
            # update the reference conditioned blocks embedding
            self.ref_cond_attn_blocks[i].update_reference_tensor(reference_embeddings[i])

    # update_num_frames updates all temporal attention block frame number
    def update_num_frames(self, num_frames):
        for i in range(len(self.ref_cond_attn_blocks)):
            # update the number of frames
            self.ref_cond_attn_blocks[i].update_num_frames(num_frames)

    # update_skip_temporal_attn updates all the skip temporal attention attributes
    def update_skip_temporal_attn(self, skip_temporal_attn):
        for i in range(len(self.ref_cond_attn_blocks)):
            # update the skip_temporal_attn attribute
            self.ref_cond_attn_blocks[i].skip_temporal_attn = skip_temporal_attn

    # forward pass just passes pose + conditioning embeddings to unet and returns activations
    def forward(self, intial_noise, timesteps, reference_embeddings, clip_condition_embeddings, skip_temporal_attn=False):

        # update the reference tensors for the ReferenceConditionedResNet modules
        self.update_reference_embeddings(reference_embeddings)

        # update the skip temporal attention attribute
        self.update_skip_temporal_attn(skip_temporal_attn)

        # forward pass the pose + conditioning embeddings through the unet
        return self.unet(
            intial_noise,
            timesteps,
            encoder_hidden_states=clip_condition_embeddings,
        )[0]

# load_mm loads a motion module into video net
def load_mm(video_net: VideoNet, mm_state_dict):
    refactored_mm_state_dict = {}
    for key in mm_state_dict:
        key_split = key.split('.')
        
        # modify the key split to have the correct arguments (except first unet)
        key_split[2] = 'attentions'
        key_split.insert(4, 'tam')
        new_key = '.'.join(key_split)
        refactored_mm_state_dict[new_key] = mm_state_dict[key]

    # load the modified weights into video_net
    _, unexpected = video_net.unet.load_state_dict(refactored_mm_state_dict, strict=False)

    return


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SpatialAttentionModule is a spatial attention module between reference and input
class SpatialAttentionModule(nn.Module):
    def __init__(self, num_inp_channels: int, embed_dim: int = 40, num_heads: int = 8) -> None:
        super(SpatialAttentionModule, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # create input projection layers
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=num_inp_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

        # create multiheaded attention module
        self.to_q = nn.Linear(num_inp_channels, embed_dim)
        self.to_k = nn.Linear(num_inp_channels, embed_dim)
        self.to_v = nn.Linear(num_inp_channels, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # create output projection layer
        self.proj_out = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

    # forward passes the activation through a spatial attention module
    def forward(self, x, reference_tensor):
        # expand and concat x with reference embedding where x is [b*t,c,h,w]
        orig_w = x.shape[3]
        concat = torch.cat((x, reference_tensor), axis=3)
        h, w = concat.shape[2], concat.shape[3]

        # pass data through input projections
        proj_x = self.norm_in(concat)
        proj_x = self.proj_in(proj_x)

        # re-arrange data from (b*t,c,h,w) to correct groupings to [b*t,w*h,c]
        grouped_x = rearrange(proj_x, 'bt c h w -> bt (h w) c')
        reshaped_x = rearrange(x, 'bt c h w -> bt (h w) c')

        # compute self-attention on the concatenated data along w dimension
        q, k, v = self.to_q(reshaped_x), self.to_k(grouped_x), self.to_v(grouped_x)

        # split embeddings for multi-headed attention
        q = rearrange(q, 'bt (h w) (n d) -> bt (h w) n d', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        k = rearrange(k, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)
        v = rearrange(v, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)

        # run attention calculation
        attn_out = memory_efficient_attention(q, k, v)
        # reshape from multihead
        attn_out = rearrange(attn_out, 'bt (h w) n d -> bt (h w) (n d)', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        
        norm1_out = self.norm1(attn_out + reshaped_x)
        ffn_out = self.ffn(norm1_out)
        attn_out = self.norm2(norm1_out + ffn_out)

        # re-arrange data from (b*t,w*h,c) to (b*t,c,h,w)
        attn_out = rearrange(attn_out, 'bt (h w) c -> bt c h w', h=x.shape[2], w=x.shape[3])

        # pass output through out projection
        out = self.proj_out(attn_out)

        # return sliced out with x as adding residual before reshape would be the same as adding x
        return out + x


# TemporalAttentionModule is a temporal attention module
class TemporalAttentionModule(nn.Module):
    def __init__(self, num_inp_channels: int, num_frames: int, embed_dim: int = 40, num_heads: int = 8) -> None:
        super(TemporalAttentionModule, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # create input projection layers
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=num_inp_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

        # create multiheaded attention module
        self.to_q = nn.Linear(num_inp_channels, embed_dim)
        self.to_k = nn.Linear(num_inp_channels, embed_dim)
        self.to_v = nn.Linear(num_inp_channels, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # create output projection layer
        self.proj_out = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

    # forward performs temporal attention on the input (b*t,c,h,w)
    def forward(self, x):
        assert x.dim() == 4, "Input tensor x must be 4-dimensional (batch*time, channels, height, width)"
    
        h, w = x.shape[2], x.shape[3]

        # pass data through input projections
        proj_x = self.norm_in(x)
        proj_x = self.proj_in(proj_x)

        # re-arrange data from (b*t,c,h,w) to correct groupings to (b*t,w*h,c)
        grouped_x = rearrange(x, '(b t) c h w -> (b h w) t c', t=self.num_frames)

        # perform self-attention on the grouped_x
        q, k, v = self.to_q(grouped_x), self.to_k(grouped_x), self.to_v(grouped_x)
        attn_out = memory_efficient_attention(q, k, v)
        norm1_out = self.norm1(attn_out + grouped_x)
        ffn_out = self.ffn(norm1_out)
        attn_out = self.norm2(norm1_out + ffn_out)

        # rearrange out to be back into the grouped batch and timestep format
        attn_out = rearrange(attn_out, '(b h w) t c -> (b t) c h w', t=self.num_frames, h=h, w=w)

        # pass attention output through out projection
        attn_out = self.proj_out(attn_out)

        return attn_out + x

    # forward passes the activation through a spatial attention module
    def forward(self, x, reference_tensor):
        # expand and concat x with reference embedding where x is [b*t,c,h,w]
        orig_w = x.shape[3]
        concat = torch.cat((x, reference_tensor), axis=3)
        h, w = concat.shape[2], concat.shape[3]

        # pass data through input projections
        proj_x = self.norm_in(concat)
        proj_x = self.proj_in(proj_x)

        # re-arrange data from (b*t,c,h,w) to correct groupings to [b*t,w*h,c]
        grouped_x = rearrange(proj_x, 'bt c h w -> bt (h w) c')
        reshaped_x = rearrange(x, 'bt c h w -> bt (h w) c')

        # compute self-attention on the concatenated data along w dimension
        q, k, v = self.to_q(reshaped_x), self.to_k(grouped_x), self.to_v(grouped_x)

        # split embeddings for multi-headed attention
        q = rearrange(q, 'bt (h w) (n d) -> bt (h w) n d', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        k = rearrange(k, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)
        v = rearrange(v, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)

        # run attention calculation
        attn_out = memory_efficient_attention(q, k, v)
        # reshape from multihead
        attn_out = rearrange(attn_out, 'bt (h w) n d -> bt (h w) (n d)', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        
        norm1_out = self.norm1(attn_out + reshaped_x)
        ffn_out = self.ffn(norm1_out)
        attn_out = self.norm2(norm1_out + ffn_out)

        # re-arrange data from (b*t,w*h,c) to (b*t,c,h,w)
        attn_out = rearrange(attn_out, 'bt (h w) c -> bt c h w', h=x.shape[2], w=x.shape[3])

        # pass output through out projection
        out = self.proj_out(attn_out)

        # return sliced out with x as adding residual before reshape would be the same as adding x
        return out + x


# ReferenceConditionedAttentionBlock is an attention block which performs spatial and temporal attention
class ReferenceConditionedAttentionBlock(nn.Module):
    def __init__(self, cross_attn: Transformer2DModel, num_frames: int, skip_temporal_attn: bool = False):
        super(ReferenceConditionedAttentionBlock, self).__init__()


        assert isinstance(cross_attn, Transformer2DModel), "cross_attn must be an instance of Transformer2DModel"
        assert isinstance(num_frames, int) and num_frames > 0, "num_frames must be a positive integer"
        assert isinstance(skip_temporal_attn, bool), "skip_temporal_attn must be a boolean"

        # store configurations and submodules
        self.skip_temporal_attn = skip_temporal_attn
        self.num_frames = num_frames
        self.cross_attn = cross_attn

        # extract channel dimension from provided cross_attn and 
        num_channels = cross_attn.config.in_channels
        embed_dim = cross_attn.config.in_channels
        self.sam = SpatialAttentionModule(num_channels, embed_dim=embed_dim)
        self.tam = get_motion_module(num_channels,
                    motion_module_type='Vanilla', 
                    motion_module_kwargs={})

        # store the reference tensor used by this module (this must be updated before the forward pass)
        self.reference_tensor = None

    # update_reference_tensor updates the reference tensor for the module
    def update_reference_tensor(self, reference_tensor: torch.FloatTensor):
        self.reference_tensor = reference_tensor

    # update_num_frames updates the number of frames the temporal attention module is configured for
    def update_num_frames(self, num_frames: int):
        self.num_frames = num_frames

    # forward performs spatial attention, cross attention, and temporal attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # begin spatial attention

        # pass concat tensor through spatial attention module along w axis [bt,c,h,w]
        out = self.sam(hidden_states, self.reference_tensor)

        # begin cross attention
        out = self.cross_attn(out, encoder_hidden_states, timestep, added_cond_kwargs, class_labels,
                            cross_attention_kwargs, attention_mask, encoder_attention_mask, return_dict)[0]

        # begin temporal attention
        if self.skip_temporal_attn:
            return (out,)
        
        # reshape data from [bt c h w] to be [b c t h w]
        temporal_input = rearrange(out, '(b t) c h w -> b c t h w', t=self.num_frames)
        
        # pass the data through the temporal attention module
        temporal_output = self.tam(temporal_input, None, None)

        # reshape temporal output back from [b c t h w] to [bt c h w]
        temporal_output = rearrange(temporal_output, 'b c t h w -> (b t) c h w')

        return (temporal_output,)




class DepthwiseSeparableInflatedConv3d(nn.Module):
    def __init__(self, inflated_conv3d_layer):
        super().__init__()
        in_channels = inflated_conv3d_layer.in_channels
        out_channels = inflated_conv3d_layer.out_channels
        kernel_size = inflated_conv3d_layer.kernel_size
        stride = inflated_conv3d_layer.stride
        padding = inflated_conv3d_layer.padding

        self.depthwise_conv = InflatedConv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = InflatedConv3d(in_channels, out_channels, kernel_size=1)

        # Initialize the depthwise convolution weights from the original InflatedConv3d layer
        depthwise_conv_weights = inflated_conv3d_layer.weight.clone()
        depthwise_conv_weights = depthwise_conv_weights.view(in_channels, 1, *kernel_size)
        self.depthwise_conv.weight = nn.Parameter(depthwise_conv_weights)
        if inflated_conv3d_layer.bias is not None:
            self.depthwise_conv.bias = nn.Parameter(inflated_conv3d_layer.bias.clone())

        # Initialize the pointwise convolution weights from the original InflatedConv3d layer
        pointwise_conv_weights = torch.ones(out_channels, in_channels, 1, 1) / in_channels
        self.pointwise_conv.weight = nn.Parameter(pointwise_conv_weights)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

        
if __name__ == '__main__':
    num_frames = 8



    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")

    # reference_unet = UNet2DConditionModel.from_pretrained('/media/2TB/ani/animate-anyone/pretrained_models/sd-image-variations-diffusers',
    #     subfolder="unet",
    # ).to(dtype=torch.float16, device="cuda")
    # construct pipe from imag evariation diffuser
    pipe = StableDiffusionImageVariationPipeline.from_pretrained('/media/2TB/ani/animate-anyone/pretrained_models/sd-image-variations-diffusers', revision="v2.0", vae=vae).to(device)
    
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to("cuda")

        
  
    # load mm pretrained weights from animatediff
    load_mm(video_net, torch.load('/media/2TB/stable-diffusion-webui/extensions/sd-webui-animatediff/model/v3_sd15_mm.ckpt'))


    # Create an optimized model with DepthwiseSeparableInflatedConv3d layers
    optimized_model = copy.deepcopy(video_net)

    for name, module in video_net.named_modules():
        print(f" name:{name} layer:{module.__class__.__name__}")
        if isinstance(module, InflatedGroupNorm):
            print(f" Found InflatedGroupNorm at {name}")
        if isinstance(module, InflatedConv3d):
            print(f"🪨 Found InflatedConv3d at {name}")
            inflated_conv3d_layer = getattr(video_net, name)
            depthwise_separable_conv3d_layer = DepthwiseSeparableInflatedConv3d(inflated_conv3d_layer)
            setattr(optimized_model, name, depthwise_separable_conv3d_layer)
            print("replacing layer....")

    # Replace InflatedConv3d layers with DepthwiseSeparableInflatedConv3d layers
    # for name, module in optimized_model.named_modules():
    #     print(f"name:{name} layer:{module}")
  
            

    # Save the checkpoint of the optimized model (5.5 gb)
    # torch.save(video_net.state_dict(), "optimized.pth")



    # Step 2: Initialize the TensorBoard SummaryWriter
    # writer = SummaryWriter('runs/videonet_experiment')



    # Assuming you have already loaded your model as `video_net`
    # Step 3: Add model graph to TensorBoard
    # Note: You may need to pass a sample input to `add_graph` depending on your model structure
    # Here, `initial_noise` is a sample input tensor
    # Get the correct number of latent dimensions from the model's configuration
    # Get the correct number of latent dimensions from the model's configuration
    # num_channels_latent = 4  # This should be verified from the model's configuration

    # # Initial noise tensor should match the latent dimensions and the model's expected input size
    # initial_noise = torch.randn(1, num_channels_latent, 512, 512).to(device)

    # # Timestep tensor; the value might need to be adjusted based on how the diffusion model processes it
    # timesteps = torch.tensor([1]).to(device)

    # # Assuming reference_embeddings need to match the number of attention blocks in your VideoNet model
    # n = len(video_net.ref_cond_attn_blocks)
    # reference_embeddings = torch.randn(1, n, num_channels_latent, 512, 512).to(device)

    # # Clip condition embeddings should match the latent dimensions and expected size
    # clip_condition_embeddings = torch.randn(1, num_channels_latent, 512, 512).to(device)

    # # You need to ensure the shapes and types match what your VideoNet model expects
    # with torch.no_grad():
    #     writer.add_graph(video_net, (initial_noise, timesteps, reference_embeddings, clip_condition_embeddings))
