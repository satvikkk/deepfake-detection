import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import cv2
import re
from utils import resize
import numpy as np
from torch import einsum
from random import randint

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout = 0))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EfficientViT(nn.Module): 
    def __init__(self, config, channels=512, selected_efficient_net=0):
        super().__init__() 

        # Extract configuration parameters
        image_size = config['model']['image-size']
        patch_size = config['model']['patch-size']
        num_classes = config['model']['num-classes']
        dim = config['model']['dim']
        depth = config['model']['depth']
        heads = config['model']['heads']
        mlp_dim = config['model']['mlp-dim']
        emb_dim = config['model']['emb-dim']
        dim_head = config['model']['dim-head']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        # Check if image dimensions are divisible by patch size
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        
        # Store selected EfficientNet variant
        self.selected_efficient_net = selected_efficient_net

        # Load pre-trained EfficientNet model
        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
            # Load custom weights if specified
            checkpoint = torch.load("weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23", map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            
        # Freeze parameters of initial layers of EfficientNet
        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks)-3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
        # Calculate number of patches and patch dimensions
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        # Define learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(emb_dim, 1, dim))
        # Linear projection of patches to embedding space
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # Define learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Dropout layer
        self.dropout = nn.Dropout(emb_dropout)

        # SE Block after patch embedding
        self.se_block = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # IF RESULTS ARE NOT GOOD, WE CAN TRY TO IMPLEMENT SE BLOCKS BEFORE CLASSIFICATION HEAD INSTEAD
        

        # Transformer layers
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Identity layer to extract class token
        self.to_cls_token = nn.Identity()

        # MLP classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        # Extract patch size
        p = self.patch_size
        # Extract features from input image using EfficientNet
        x = self.efficient_net.extract_features(img) # 1280x7x7
        
        # Apply SE Block after patch embedding
        se_weights = self.se_block(x)
        x = x * se_weights
        
        # Rearrange features into patches
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        y = self.patch_to_embedding(y)
        
        # Expand class token and concatenate with patches
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        
        # Add positional embeddings and apply dropout
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.dropout(x)
        
        # Pass through Transformer layers
        x = self.transformer(x)
        # Extract class token
        x = self.to_cls_token(x[:, 0])
        
        # Pass through MLP classification head
        return self.mlp_head(x)
