import os
import json
import csv
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import CFG

CAMERA_VISIBILITY_OCCLUSION = {
    "Camera_rightside": {
        "visible_bones": [
            "ear.r", "eye.r",
            "hip_b.r", "hip_f.r", "leg_b.r", "leg_f.r",
            "shin_b.r", "shin_f.r", "thigh_b.r", "thigh_f.r",
            "foot_b.r", "foot_f.r", "claws_b.r", "claws_f.r",
            "helper_shin_f.r", "helper_foot_b.r"
        ]
    },
    "Camera_leftside": {
        "visible_bones": [
            "ear.l", "eye.l",
            "hip_b.l", "hip_f.l", "leg_b.l", "leg_f.l",
            "shin_b.l", "shin_f.l", "thigh_b.l", "thigh_f.l",
            "foot_b.l", "foot_f.l", "claws_b.l", "claws_f.l",
            "helper_shin_f.l", "helper_foot_b.l"
        ]
    },
    "Camera_Top": {
        "visible_bones": [
            "head", "nose", "mouth", "tongue_1", "tongue_2", "tongue_3", "tongue_4",
            "ear.l", "ear.r", "eye.l", "eye.r",
            "leg_b.l", "leg_b.r", "leg_f.l", "leg_f.r",
            "shin_b.l", "shin_b.r", "shin_f.l", "shin_f.r",
            "hip_b.l", "hip_b.r", "hip_f.l", "hip_f.r",
            "thigh_b.l", "thigh_b.r", "thigh_f.l", "thigh_f.r",
            "spine_03", "spine_04", "spine_05", "spine_base",
            "neck", "root_bone"
        ]
    },
    "Camera_headside": {
        "visible_bones": [
            "head", "nose", "mouth", "tongue_1", "tongue_2", "tongue_3", "tongue_4",
            "ear.l", "ear.r",
            "leg_b.l", "leg_b.r", "shin_b.l", "shin_b.r",
            "hip_b.l", "hip_b.r", "thigh_b.l", "thigh_b.r",
            "foot_b.l", "foot_b.r", "claws_b.l", "claws_b.r",
            "helper_foot_b.l", "helper_foot_b.r",
            "spine_03", "spine_04", "spine_05", "spine_base",
            "neck", "root_bone"
        ]
    },
    "Camera_backside": {
        "visible_bones": [
            "ear.l", "ear.r", "eye.l", "eye.r",
            "foot_f.l", "foot_f.r", "claws_f.l", "claws_f.r",
            "leg_f.l", "leg_f.r", "shin_f.l", "shin_f.r",
            "hip_f.l", "hip_f.r", "thigh_f.l", "thigh_f.r",
            "spine_03", "spine_04", "spine_05", "spine_base",
            "tail_01", "tail_02", "tail_03", "tail_04",
            "neck", "root_bone"
        ]
    }
}


def load_bone_order(path=None):
    path = path or CFG.paths.bone_order
    with open(path) as f:
        bones = json.load(f)
    return bones, {b: i for i, b in enumerate(bones)}


def find_all_csvs(root_dir):
    csv_files = []
    for breed_dir in sorted(Path(root_dir).iterdir()):
        if not breed_dir.is_dir():
            continue
        csv_dir = breed_dir / "2D_pose"
        if csv_dir.exists():
            for csv_file in sorted(csv_dir.glob("coordinates_2d_*.csv")):
                csv_files.append(csv_file)
    return csv_files


def parse_csv_to_samples(csv_path, bones):
    samples = {}
    bone_to_idx = {b: i for i, b in enumerate(bones)}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['Frame'])
                camera = row['Camera'].strip()
                focal = int(row['Focal Length'])
                bone = row['Bone'].strip()
                x = float(row['X'])
                y = float(row['Y'])
                
                if bone not in bone_to_idx:
                    continue
                
                key = (frame, camera, focal)
                if key not in samples:
                    samples[key] = {}
                samples[key][bone] = (x, y)
            except (ValueError, KeyError):
                continue
    
    return samples


def find_image_file(breed_dir, full_name, focal, camera, frame):
    image_dir = breed_dir / f"output_{full_name}" / f"focal_{focal}" / camera
    
    if not image_dir.exists():
        return None
    
    pattern = f"render_{full_name}_{camera}_f{focal}_frame{frame:04d}.png"
    image_path = image_dir / pattern
    if image_path.exists():
        return image_path
    
    for img in image_dir.glob(f"*frame{frame:04d}.png"):
        return img
    
    return None


def build_dataset_index(root_dir, bones):

    csv_files = find_all_csvs(root_dir)
    samples_by_breed_action = {}
    bone_to_idx = {b: i for i, b in enumerate(bones)}
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in tqdm(csv_files, desc="Building index"):
        breed_dir = csv_file.parent.parent
        breed = breed_dir.name.replace('output_', '')  #  'G_Retriever', 'JR_terrier', 'Akita'
        
        full_name = csv_file.stem.replace("coordinates_2d_", "")  # 'G_Retriever_Albedo_Attack_F'
        
        # (case-insensitive)
        breed_lower = breed.lower()
        full_name_lower = full_name.lower()
        
        if full_name_lower.startswith(breed_lower + '_'):
            # Remove breed prefix using original casing
            remainder = full_name[len(breed) + 1:]  #  preserve original case
            #  'Albedo_Attack_Bite'
        else:
            remainder = full_name  #  (shouldn't happen)
        
        # Step 4: Split remainder into texture and action
        parts = remainder.split('_')
        texture = parts[0] if len(parts) > 0 else 'Unknown'           # 'Albedo'
        action = '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'  # 'Attack_Bite'
        
        key = (breed, action)
        if key not in samples_by_breed_action:
            samples_by_breed_action[key] = []
        
        csv_samples = parse_csv_to_samples(csv_file, bones)
        
        for (frame, camera, focal), bone_dict in csv_samples.items():
            image_path = find_image_file(breed_dir, full_name, focal, camera, frame)
            
            if image_path is None:
                continue
            
            keypoints = np.zeros((len(bones), 2), dtype=np.float32)
            visibility = np.ones(len(bones), dtype=np.float32)
            
            for bone_name, (x, y) in bone_dict.items():
                if bone_name in bone_to_idx:
                    idx = bone_to_idx[bone_name]
                    keypoints[idx, 0] = x
                    keypoints[idx, 1] = y
            
            #  invisible (label space bounds)
            label_w, label_h = 1920, 1080
            for i in range(len(bones)):
                x, y = keypoints[i]
                if x < 0 or x > label_w or y < 0 or y > label_h:
                    visibility[i] = 0
            
            samples_by_breed_action[key].append({
                'image_path': str(image_path),
                'keypoints': keypoints,  # (1920x1080)
                'visibility': visibility,
                'camera': camera,
                'focal': focal,
                'frame': frame,
                'action': action,
                'breed': breed,
                'texture': texture,
            })
    
    total_samples = sum(len(s) for s in samples_by_breed_action.values())
    print(f"Built index: {total_samples} samples from {len(samples_by_breed_action)} unique (breed, action) pairs")
    
    return samples_by_breed_action


def _build_and_split_all_indices(root_dir, bones, cfg):

    cache_dir = Path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    index_train = cache_dir / "index_train.pkl"
    index_val = cache_dir / "index_val.pkl"
    index_test = cache_dir / "index_test.pkl"
    
    if index_train.exists() and index_val.exists() and index_test.exists():
        print("All pkl files already exist, skipping rebuild")
        return
    
    print("Building index once for all splits...")
    samples_by_breed_action = build_dataset_index(root_dir, bones)
    
    breed_actions = sorted(samples_by_breed_action.keys())
    seed = cfg.train.seed
    np.random.seed(seed)
    shuffled_ba = breed_actions.copy()
    np.random.shuffle(shuffled_ba)
    
    n_train = int(len(breed_actions) * cfg.train.split_ratios[0])
    n_val = int(len(breed_actions) * cfg.train.split_ratios[1])
    
    print(f"Total (breed, action) pairs: {len(breed_actions)}")
    print(f"  Train pairs: {n_train}")
    print(f"  Val pairs: {n_val}")
    print(f"  Test pairs: {len(breed_actions) - n_train - n_val}")
    
    # Split
    train_ba = shuffled_ba[:n_train]
    val_ba = shuffled_ba[n_train:n_train + n_val]
    test_ba = shuffled_ba[n_train + n_val:]
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for ba in train_ba:
        train_samples.extend(samples_by_breed_action[ba])
    for ba in val_ba:
        val_samples.extend(samples_by_breed_action[ba])
    for ba in test_ba:
        test_samples.extend(samples_by_breed_action[ba])
    
    print("\nSaving all three pkl files...")
    with open(index_train, 'wb') as f:
        pickle.dump(train_samples, f)
    print(f"{index_train} ({len(train_samples):,} samples)")
    
    with open(index_val, 'wb') as f:
        pickle.dump(val_samples, f)
    print(f"{index_val} ({len(val_samples):,} samples)")
    
    with open(index_test, 'wb') as f:
        pickle.dump(test_samples, f)
    print(f"{index_test} ({len(test_samples):,} samples)")


class DogPose2D(Dataset):
    
    def __init__(self, split="train", cfg=CFG):
        self.cfg = cfg
        self.split = split
        
        bones, bone_to_idx = load_bone_order(cfg.paths.bone_order)
        self.bones = bones
        self.bone_to_idx = bone_to_idx
        self.N = len(bones)
        
        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])
        
        cache_dir = Path(cfg.paths.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_cache = cache_dir / f"index_{split}.pkl"
        
        if index_cache.exists():
            print(f"Loading cached index: {index_cache}")
            with open(index_cache, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            _build_and_split_all_indices(cfg.paths.root, bones, cfg)
            
            with open(index_cache, 'rb') as f:
                self.samples = pickle.load(f)
        
        print(f"✓ Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = cv2.imread(sample['image_path'])
        if img is None:
            img = np.zeros((540, 960, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        keypoints = sample['keypoints'].copy().astype(np.float32)  # [N, 2] in label space (1920x1080)
        visibility = sample['visibility'].copy().astype(np.float32)

        keypoints_img = keypoints.copy()
        keypoints_img[:, 0] *= (960.0 / 1920.0)
        keypoints_img[:, 1] *= (540.0 / 1080.0)
        
        x_min, y_min = keypoints_img.min(axis=0)
        x_max, y_max = keypoints_img.max(axis=0)
        
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        margin = self.cfg.aug.bbox_margin
        x_min = max(0, x_min - w * (margin - 1) / 2)
        y_min = max(0, y_min - h * (margin - 1) / 2)
        x_max = min(960, x_max + w * (margin - 1) / 2)
        y_max = min(540, y_max + h * (margin - 1) / 2)
        
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        # Safety
        x_max = max(x_max, x_min + 1)
        y_max = max(y_max, y_min + 1)
        
        img_crop = img[y_min:y_max, x_min:x_max]
        keypoints_img[:, 0] -= x_min
        keypoints_img[:, 1] -= y_min
        
        crop_h, crop_w = img_crop.shape[:2]
        if crop_h == 0 or crop_w == 0:
            # Fallback if crop is empty
            img_crop = img
            keypoints_img = keypoints.copy()
            keypoints_img[:, 0] *= (960.0 / 1920.0)
            keypoints_img[:, 1] *= (540.0 / 1080.0)
            crop_h, crop_w = 540, 960
        
        img_resized = cv2.resize(img_crop, (self.cfg.geom.input_w, self.cfg.geom.input_h))
        img = img_resized
        
        keypoints_img[:, 0] = (keypoints_img[:, 0] / crop_w) * self.cfg.geom.input_w
        keypoints_img[:, 1] = (keypoints_img[:, 1] / crop_h) * self.cfg.geom.input_h
        
        keypoints_img[:, 0] /= self.cfg.geom.input_w
        keypoints_img[:, 1] /= self.cfg.geom.input_h
        keypoints_img = np.clip(keypoints_img, 0, 1)
        
        img = (img - self.img_mean) / self.img_std
        img = img.transpose(2, 0, 1) 

        camera = sample['camera']
        # suffixes like .002, .001
        camera_base = camera.split('.')[0] 
        
        visibility = np.zeros(self.N, dtype=np.float32)
        # Set only visible bones to 1.0
        if camera_base in CAMERA_VISIBILITY_OCCLUSION:
            visible_bones = CAMERA_VISIBILITY_OCCLUSION[camera_base]["visible_bones"]
            for bone_name in visible_bones:
                if bone_name in self.bone_to_idx:
                    bone_idx = self.bone_to_idx[bone_name]
                    visibility[bone_idx] = 1.0
        
        return {
            'image': torch.from_numpy(img).float(),
            'keypoints': torch.from_numpy(keypoints_img).float(),
            'visibility': torch.from_numpy(visibility).float(),
            'camera': camera,
        }


def make_loader(split, cfg, batch_size=None, num_workers=None, shuffle=None):

    batch_size = batch_size or cfg.train.batch_size
    num_workers = num_workers or cfg.train.num_workers
    shuffle = shuffle if shuffle is not None else (split == "train")
    
    ds = DogPose2D(split, cfg=cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,     
        persistent_workers=True,
        pin_memory=True,
        drop_last=(split == "train")
    )
    return loader, ds


def make_target_heatmaps(keypoints, visibility, cfg=CFG, sigma=None):
    g = cfg.geom
    sigma = cfg.target.sigma if sigma is None else sigma
    B, N, _ = keypoints.shape
    device = keypoints.device
    tw = visibility.clone()
    xs = keypoints[..., 0] * (g.hm_w - 1)
    ys = keypoints[..., 1] * (g.hm_h - 1)
    yy = torch.arange(g.hm_h, device=device).view(1, 1, g.hm_h, 1)
    xx = torch.arange(g.hm_w, device=device).view(1, 1, 1, g.hm_w)
    cx = xs.view(B, N, 1, 1); cy = ys.view(B, N, 1, 1)
    gmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    #####
    gmap = gmap * 10.0 
    
    return gmap * tw.view(B, N, 1, 1), tw



if __name__ == "__main__":

    train_loader, train_ds = make_loader("train", CFG)
