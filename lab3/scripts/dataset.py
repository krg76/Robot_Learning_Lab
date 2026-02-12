import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def save_stats_npz(path, stats: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **stats)

def load_stats_npz(path) -> dict:
    d = np.load(path)
    return {k: d[k].astype(np.float32) for k in d.files}

def get_or_compute_stats(stats_path, train_eps, normalize=True):
    """
    If normalize=False -> returns None.
    If stats file exists -> load.
    Else -> compute, save, return.
    """
    if not normalize:
        return None

    if os.path.exists(stats_path):
        print(f"[stats] loading from {stats_path}")
        return load_stats_npz(stats_path)

    print(f"[stats] not found, computing and saving to {stats_path}")
    stats = compute_stats_from_preloaded(train_eps)
    save_stats_npz(stats_path, stats)
    return stats

# -----------------------
# Utils
# -----------------------
def split_files(dir_path, test_ratio=0.2, seed=42):
    fs = sorted([f for f in os.listdir(dir_path) if f.endswith(".npy")])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fs))
    k = int(len(fs) * (1 - test_ratio))
    tr = [fs[i] for i in idx[:k]]
    te = [fs[i] for i in idx[k:]]
    return tr, te


# @torch.no_grad()
# def resize_uint8_hwc_to_float_chw(img_hwc_uint8, size=(96, 96)):
#     """
#     img_hwc_uint8: (H,W,3) uint8
#     returns: (3,h,w) float32 in [0,1]
#     """
#     x = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0  # (3,H,W)
#     x = x.unsqueeze(0)  # (1,3,H,W)
#     x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
#     return x.squeeze(0)  # (3,h,w)

def compute_stats_from_preloaded(train_eps, eps=1e-8):
    s_all = np.concatenate([ep["state"] for ep in train_eps], axis=0)
    a_all = np.concatenate([ep["action"] for ep in train_eps], axis=0)

    s_mean = s_all.mean(axis=0).astype(np.float32)
    s_std  = np.maximum(s_all.std(axis=0).astype(np.float32), eps)
    a_mean = a_all.mean(axis=0).astype(np.float32)
    a_std  = np.maximum(a_all.std(axis=0).astype(np.float32), eps)

    img_means, img_sqs = [], []
    w_means, w_sqs = [], []

    for ep in tqdm(train_eps, desc="Computing image stats"):
        img = ep["image"].astype(np.float32) / 255.0   # (T,3,H,W)
        wst = ep["wrist"].astype(np.float32) / 255.0

        img_means.append(img.mean(axis=(0, 2, 3)))
        img_sqs.append((img ** 2).mean(axis=(0, 2, 3)))

        w_means.append(wst.mean(axis=(0, 2, 3)))
        w_sqs.append((wst ** 2).mean(axis=(0, 2, 3)))

    img_mean = np.mean(img_means, axis=0).astype(np.float32)
    img_sq   = np.mean(img_sqs, axis=0).astype(np.float32)
    img_std  = np.maximum(np.sqrt(np.maximum(img_sq - img_mean ** 2, 0.0)).astype(np.float32), eps)

    w_mean = np.mean(w_means, axis=0).astype(np.float32)
    w_sq   = np.mean(w_sqs, axis=0).astype(np.float32)
    w_std  = np.maximum(np.sqrt(np.maximum(w_sq - w_mean ** 2, 0.0)).astype(np.float32), eps)

    return {
        "s_mean": s_mean, "s_std": s_std,
        "a_mean": a_mean, "a_std": a_std,
        "img_mean": img_mean, "img_std": img_std,
        "wimg_mean": w_mean, "wimg_std": w_std,
    }


def preload_episodes(dir_path, files, include_images=True):
    """
    Loads each episode ONCE.
    Returns list of dict episodes with numpy arrays.
    Images are kept in original HWC uint8 format.
    """
    episodes = []

    for f in tqdm(files, desc="Preloading episodes"):
        ep = np.load(os.path.join(dir_path, f), allow_pickle=True)  # (T,) dicts
        T = len(ep)

        state  = np.stack([ep[t]["state"]  for t in range(T)], axis=0).astype(np.float32)
        action = np.stack([ep[t]["action"] for t in range(T)], axis=0).astype(np.float32)
        lang   = str(ep[0]["language_instruction"])

        out = {
            "state": state,          # (T, obs_dim)
            "action": action,        # (T, act_dim)
            "lang": lang,
        }

        if include_images:
            # Keep raw HWC uint8 images
            img = np.stack([ep[t]["image"] for t in range(T)], axis=0)       # (T,H,W,3)
            wst = np.stack([ep[t]["wrist_image"] for t in range(T)], axis=0) # (T,H,W,3)

            # move channel to dim=1
            img = img.transpose(0, 3, 1, 2)   # (T,3,H,W)
            wst = wst.transpose(0, 3, 1, 2)   # (T,3,H,W)

            out["image"] = img
            out["wrist"] = wst

        episodes.append(out)

    return episodes

# -----------------------
# Dataset
# -----------------------
class HorizonDataset(Dataset):
    def __init__(self, episodes, obs_h, pred_h, stats=None, include_images=True, normalize=True):
        """
        episodes: list of preloaded dicts
        """
        self.episodes = episodes
        self.obs_h = int(obs_h)
        self.pred_h = int(pred_h)
        self.stats = stats
        self.include_images = include_images
        self.normalize = normalize

        # build global index: (episode_idx, t_start)
        self.idxs = []
        for i, ep in enumerate(self.episodes):
            T = ep["state"].shape[0]
            t_min = self.obs_h - 1
            t_max = T - self.pred_h
            if t_max >= t_min:
                self.idxs += [(i, t) for t in range(t_min, t_max + 1)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i, t = self.idxs[k]
        ep = self.episodes[i]

        obs = slice(t - self.obs_h + 1, t + 1)
        fut = slice(t, t + self.pred_h)

        obs_state = ep["state"][obs].copy()     # (Hobs,obs_dim)
        pred_act  = ep["action"][fut].copy()    # (Hpred,act_dim)

        if self.normalize and self.stats is not None:
            obs_state = (obs_state - self.stats["s_mean"]) / self.stats["s_std"]
            pred_act  = (pred_act  - self.stats["a_mean"]) / self.stats["a_std"]

        out = {
            "obs_state": torch.from_numpy(obs_state),
            "pred_action": torch.from_numpy(pred_act),
            "language_instruction": ep["lang"],
            "ep_id": torch.tensor(i, dtype=torch.int32),
            "t": torch.tensor(t, dtype=torch.int32),
        }

        if self.include_images:
            img = ep["image"][obs].astype(np.float32) / 255.0   # (Hobs,3,H,W)
            wst = ep["wrist"][obs].astype(np.float32) / 255.0

            if self.normalize and self.stats is not None:
                img = (img - self.stats["img_mean"][None, :, None, None]) / self.stats["img_std"][None, :, None, None]
                wst = (wst - self.stats["wimg_mean"][None, :, None, None]) / self.stats["wimg_std"][None, :, None, None]

            out["obs_image"] = torch.from_numpy(img)
            out["obs_wrist_image"] = torch.from_numpy(wst)

        return out


def collate_keep_str(batch):
    out = {}
    for k in batch[0]:
        v0 = batch[0][k]
        if isinstance(v0, str):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], 0)
    return out


def make_loaders(
    dir_path,
    obs_h=2,
    pred_h=16,
    batch_size=64,
    test_ratio=0.2,
    seed=42,
    include_images=True,
    normalize=True,
    # IMPORTANT when preloading: keep workers low to avoid RAM duplication
    num_workers=0,
    pin_memory=True,
):
    tr_files, te_files = split_files(dir_path, test_ratio, seed)

    # preload
    train_eps = preload_episodes(dir_path, tr_files, include_images=include_images)
    test_eps  = preload_episodes(dir_path, te_files, include_images=include_images)

    stats_path = "asset/stats.npz"
    stats = get_or_compute_stats(stats_path, train_eps, normalize=normalize)

    tr_ds = HorizonDataset(train_eps, obs_h, pred_h, stats=stats, include_images=include_images, normalize=normalize)
    te_ds = HorizonDataset(test_eps, obs_h, pred_h, stats=stats, include_images=include_images, normalize=normalize)

    tr_ld = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, collate_fn=collate_keep_str
    )
    te_ld = DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, collate_fn=collate_keep_str
    )
    return tr_ld, te_ld, stats
