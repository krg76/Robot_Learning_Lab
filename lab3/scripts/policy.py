# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

import torch

from scripts.bc import BCConvMLPPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def crop(img):
    H,W,_ = img.shape
    img_2 = img[H//2-48:H//2+48,W//2-48:W//2+48,:]
    img_3 = np.moveaxis(img_2, 2, 0)
    return np.expand_dims(img_3, axis=(0,1))
    #return torch.permute(img_2, (2, 0, 1))


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
        pre_trained_info = torch.load("asset/checkpoints/bcconv_final.pt", weights_only=False)

        model_kwargs = pre_trained_info["model_kwargs"]

        self.model = BCConvMLPPolicy(**model_kwargs).to(device)

        self.model.load_state_dict(pre_trained_info["model_state_dict"])

        self.stats = pre_trained_info["stats"]

        self.state_buff = []

        self.img_buff = []

        self.wimg_buff = []


    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        # return PolicyOut(action = np.deg2rad([-0.2,-45.4,-0.3,21.8,0.4,67.2,0.6,1]))

        return

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)

        self.state_buff.append(joints)
        joints = self.state_buff.pop(0)

        base_rgb = np.asarray(obs["base_rgb"], dtype=np.float32)

        self.img_buff.append(base_rgb)
        base_rgb = self.img_buff.pop(0)

        wrist_rgb = np.asarray(obs["wrist_rgb"], dtype=np.float32)

        self.wimg_buff.append(wrist_rgb)
        wrist_rgb = self.wimg_buff.pop(0)
        
        s_mean = self.stats["s_mean"]
        s_std = self.stats["s_std"]
        img_mean = self.stats["img_mean"]
        img_std = self.stats["img_std"]
        wimg_mean = self.stats["wimg_mean"]
        wimg_std = self.stats["wimg_std"]

        a_mean = self.stats["a_mean"]
        a_std = self.stats["a_std"]

        joints_norm = (joints - s_mean)/s_std
        joints_norm = np.expand_dims(joints_norm, axis=(0,1))

        base_rgb_norm = (base_rgb - img_mean)/img_std

        base_rgb_norm = crop(base_rgb_norm)

        wrist_rgb_norm = (wrist_rgb - wimg_mean)/wimg_std

        wrist_rgb_norm = crop(wrist_rgb_norm)

        #print(base_rgb_norm.shape, wrist_rgb_norm.shape)

        action = self.model(torch.from_numpy(joints_norm).to(device),torch.from_numpy(base_rgb_norm).to(device),torch.from_numpy(wrist_rgb_norm).to(device))

        action_unnorm = (action.detach().cpu().numpy() * a_std) + a_mean 
        #print(action_unnorm)
        #print("stats",a_mean,a_std)

        #act = action_unnorm[0,0]

        #action_unnorm = np.asarray(obs["joint_positions"], dtype=np.float32)

        #print(obs)

        #action_unnorm[-2] += 0.1

        #action_unnorm[-1] = 0

        # TODO: replace with your model inference
        #action = joints.copy()  # safe default: hold

        return PolicyOut(action=action_unnorm, info=None)