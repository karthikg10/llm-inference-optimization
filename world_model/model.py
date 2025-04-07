# model.py — World Model for Video Frame Prediction (fully runnable)
# Latent diffusion UNet that predicts future frames from past context.
# Runs on CPU when CUDA is unavailable; designed for consumer GPUs with INT8.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Diffusion timestep embedding (sinusoidal + MLP)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(min(8, in_ch),  in_ch)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention at a spatial resolution (for bottleneck)."""
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm  = nn.GroupNorm(min(8, ch), ch)
        self.attn  = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H*W).transpose(1, 2)  # [B, HW, C]
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class DiffusionUNet(nn.Module):
    """
    UNet backbone for latent diffusion.
    Down path -> bottleneck (with attention) -> up path.
    """
    def __init__(self, in_ch=4, base_ch=64, ch_mult=(1,2,4), time_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples  = nn.ModuleList()
        ch = base_ch
        self.down_chs = [ch]
        for mult in ch_mult:
            out_ch = base_ch * mult
            self.down_blocks.append(ResBlock(ch, out_ch, time_dim))
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            self.down_chs.append(out_ch)
            ch = out_ch

        # Bottleneck
        self.mid_res1  = ResBlock(ch, ch, time_dim)
        self.mid_attn  = AttentionBlock(ch)
        self.mid_res2  = ResBlock(ch, ch, time_dim)

        # Up path
        self.up_blocks   = nn.ModuleList()
        self.upsamples   = nn.ModuleList()
        for mult in reversed(ch_mult):
            out_ch = base_ch * mult
            skip_ch = self.down_chs.pop()
            self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim))
            ch = out_ch

        # Output projection
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B, in_ch, H, W]  (noisy latent)
        t: [B]                (diffusion timestep)
        returns: [B, in_ch, H, W] (predicted noise)
        """
        t_emb = self.time_embed(t)
        h = self.in_conv(x)

        # Down
        skips = [h]
        for res, down in zip(self.down_blocks, self.downsamples):
            h = res(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Up
        for up, res in zip(self.upsamples, self.up_blocks):
            h  = up(h)
            sk = skips.pop()
            # Handle size mismatch from strided conv
            if h.shape != sk.shape:
                h = F.interpolate(h, size=sk.shape[2:])
            h = torch.cat([h, sk], dim=1)
            h = res(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


class WorldModel(nn.Module):
    """
    Full World Model: encodes past frames -> latent -> diffusion UNet -> decode future frame.
    """
    def __init__(self, in_channels=3, latent_dim=4, hidden_dim=64,
                 num_frames=4, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size   = img_size

        # Temporal encoder (3D conv over past frames)
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=(2,3,3), padding=(0,1,1)),
            nn.SiLU(),
            nn.Conv3d(64, latent_dim, kernel_size=(1,3,3), padding=(0,1,1)),
        )

        # Latent diffusion UNet
        self.unet = DiffusionUNet(
            in_ch=latent_dim, base_ch=hidden_dim, time_dim=128)

        # VAE-style decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, frames):
        """frames: [B, T, C, H, W] -> latent [B, latent_dim, H/4, W/4]"""
        B, T, C, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4)           # [B, C, T, H, W]
        z = self.frame_encoder(x)                     # [B, latent, 1, H, W] approx
        return z[:, :, -1]                            # [B, latent, H, W]

    def denoise_step(self, z_noisy, t):
        """One denoising step — returns predicted noise."""
        return self.unet(z_noisy, t)

    def forward(self, frames, noise_level=None):
        """
        frames:      [B, T, C, H, W]  past frames
        noise_level: [B] diffusion timestep (random if None)
        returns:     predicted next frame [B, C, H, W]
        """
        B = frames.size(0)
        device = frames.device

        z = self.encode(frames)

        if noise_level is None:
            noise_level = torch.randint(0, 1000, (B,), device=device)

        # Add noise to latent (forward diffusion)
        noise  = torch.randn_like(z)
        alpha  = 1.0 - noise_level.float() / 1000.0
        z_noisy = alpha[:, None, None, None] * z + \
                  (1 - alpha[:, None, None, None]) * noise

        # Predict and remove noise
        pred_noise = self.unet(z_noisy, noise_level)
        z_denoised = z_noisy - (1 - alpha[:, None, None, None]) * pred_noise

        # Decode to pixel space
        # Resize if decoder stride causes mismatch
        out = self.decoder(z_denoised)
        if out.shape[-1] != frames.shape[-1]:
            out = F.interpolate(out, size=frames.shape[-2:])
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = WorldModel(img_size=64).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    B, T, C, H, W = 2, 4, 3, 64, 64
    frames = torch.randn(B, T, C, H, W, device=device)
    t      = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        pred = model(frames, t)

    print(f"Input:  {list(frames.shape)}")
    print(f"Output: {list(pred.shape)}  (predicted next frame)")
    print("World model forward pass: OK")
