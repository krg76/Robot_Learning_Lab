# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

#from scripts.bc import BCConvMLPPolicy
from scripts.unet import DiffusionPolicyUNet

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler

from scripts.dataset import DiffusionDatasetBoth, load_dir_episodes
from scripts.unet import DiffusionPolicyUNet

import torch

@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None

class UniversalPolicy:
    """
    Students implement:
      - reset()
      - step(obs) -> PolicyOut

    obs (from RobotEnv) will typically include:
      obs["joint_positions"] : (D,)
      obs["base_rgb"]        : (H,W,3) uint8
      obs["wrist_rgb"]       : (H,W,3) uint8
    """
    def __init__(self):
        # TODO: load model, init buffers, etc.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(
            "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab4/asset/policy/DiT_state_cosine_final.pth",#"asset/checkpoints/bcconv_final.pt",
            map_location=self.device,
            weights_only=False
        )
        self.stats = ckpt["stats"]
        self.model = DiffusionPolicyUNet(
            obs_low_dim=ckpt["meta"]["obs_dim"],
            action_dim=ckpt["meta"]["action_dim"],
            obs_horizon=ckpt["meta"]["obs_horizon"],
            #num_diffusion_steps=ckpt["meta"]["num_diffusion_steps"],#args["num_diffusion_steps"],
            image_type="both",
            #eta=0.0,
            #schedular_type="linear"
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.obs_horizon = ckpt["meta"]["obs_horizon"]
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

        self.num_diffusion_steps = ckpt["meta"]["num_diffusion_steps"]
        self.pred_horizon = ckpt["meta"]["pred_horizon"]
        self.action_dim = ckpt["meta"]["action_dim"]

        self.noise_scheduler = DDPMScheduler(   
            num_train_timesteps=self.num_diffusion_steps,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        PolicyOut(action = np.deg2rad(np.array([-0.2,-45.4,-0.3,21.8,0.4,67.2,0.6])))
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []
        
    def step(self, obs: Dict[str, Any]) -> PolicyOut:

      def _center_crop_hwc(img: np.ndarray, crop_h: int = 96, crop_w: int = 96) -> np.ndarray:
          """
          img: (H, W, 3) numpy array
          return: (3, crop_h, crop_w) numpy array (CHW)
          """
          H, W, C = img.shape
          assert C == 3, f"Expected 3 channels, got {C}"

          if H < crop_h or W < crop_w:
              raise ValueError(f"Image too small to crop: {(H, W)} < {(crop_h, crop_w)}")

          top = (H - crop_h) // 2
          left = (W - crop_w) // 2

          cropped = img[top:top + crop_h, left:left + crop_w, :]  # (96,96,3)
          cropped = np.transpose(cropped, (2, 0, 1))               # (3,96,96)
          return cropped
    
      joints = np.asarray(obs["joint_positions"], dtype=np.float32)
      img = np.asarray(obs["base_rgb"], dtype=np.float32)
      wimg = np.asarray(obs["wrist_rgb"], dtype=np.float32)

      img = _center_crop_hwc(img, 96, 96)
      wimg = _center_crop_hwc(wimg, 96, 96)

      img_mean = self.stats["img_ext"]["mean"].reshape(3, 1, 1).cpu().numpy()
      img_std  = self.stats["img_ext"]["std"].reshape(3, 1, 1).cpu().numpy()

      wimg_mean = self.stats["img_wst"]["mean"].reshape(3, 1, 1).cpu().numpy()
      wimg_std  = self.stats["img_wst"]["std"].reshape(3, 1, 1).cpu().numpy()

      joints = (joints - self.stats["obs"]["mean"].cpu().numpy())/self.stats["obs"]["std"].cpu().numpy()
      img = (img - img_mean)/img_std
      wimg = (wimg - wimg_mean)/wimg_std

      self.state_buffer.append(joints.copy())
      self.img_buffer.append(img)
      self.wimg_buffer.append(wimg)
      
      if len(self.state_buffer) > self.obs_horizon:
        self.state_buffer = self.state_buffer[-self.obs_horizon:]
      if len(self.img_buffer) > self.obs_horizon:
        self.img_buffer = self.img_buffer[-self.obs_horizon:]
      if len(self.wimg_buffer) > self.obs_horizon:
        self.wimg_buffer = self.wimg_buffer[-self.obs_horizon:]
  
      state_seq = np.stack(self.state_buffer, axis=0).astype(np.float32)  # (T,D)
      img_seq   = np.stack(self.img_buffer, axis=0).astype(np.float32)    # (T,3,96,96)
      wimg_seq  = np.stack(self.wimg_buffer, axis=0).astype(np.float32)   # (T,3,96,96)

      state_t = torch.from_numpy(state_seq).unsqueeze(0).to(self.device)  # (1,T,D)
      img_t   = torch.from_numpy(img_seq).unsqueeze(0).to(self.device)    # (1,T,3,96,96)
      wimg_t  = torch.from_numpy(wimg_seq).unsqueeze(0).to(self.device)   # (1,T,3,96,96)

      #start from here
      #print()
      B = 1
      rand_act = torch.rand((B,self.pred_horizon,self.action_dim)).to(self.device)
      self.noise_scheduler.set_timesteps(self.num_diffusion_steps)
      x_curr = rand_act

      for k in self.noise_scheduler.timesteps:
        noise_pred = self.model(x_curr, k.to(self.device), state_t, img_t, wimg_t)
        x_curr = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=x_curr).prev_sample
      action = x_curr

      #print("qewrbuhwgerhi;ou")

      #action = self.model(state_t, img_t, wimg_t)
      action = action.detach().cpu().numpy()
      range_ = self.stats["act"]["max"].cpu().numpy() - self.stats["act"]["min"].cpu().numpy()
      action = (action + 1) * range_ / 2 + self.stats["act"]["min"].cpu().numpy()


      # normalize shape to (K, 8)
      action = np.asarray(action, dtype=np.float32)
      if action.ndim == 1:          # (8,)
          action = action[None, :]  # (1,8)
      elif action.ndim == 2:        # (1,8) or (K,8)
          pass
      elif action.ndim == 3:        # (1,K,8)
          action = action[0]        # (K,8)
      else:
          raise ValueError(f"Unexpected action shape: {action.shape}")

      return action