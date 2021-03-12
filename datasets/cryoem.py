import os, sys
import math, random
import numpy as np
import torch
import json

class CryoEmDataset(object):
    
    def __init__(self, root_dir, subsample=0, suffix=None):

        self.white_back = True
        # Read metadata
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
            required_keys = ['near', 'far']
            if not np.all([(k in self.meta_dict) for k in required_keys]):
                raise IOError('Missing required meta data')
        
        # Construct loaded filename
        rgbs_name, rays_name = 'rgbs' + '_train', 'rays' + '_train'

        if subsample != 0:
            rgbs_name += f'_x{subsample}'
            rays_name += f'_x{subsample}'

        # add suffix
        rgbs_name += '.npy'
        rays_name += '.npy'

        print("Loading cryoem data:", root_dir)
        self.rays = np.load(os.path.join(root_dir, rays_name)) # [N, H, W, ro+rd, 3]
        N, H, W, _, _ = self.rays.shape
        self.rays = self.rays.reshape(N,H,W,6)
        
        # RGB files may not exist considering exhibit set
        rgb_path = os.path.join(root_dir, rgbs_name)
        if os.path.exists(rgb_path):
            self.rgbs = np.load(rgb_path).astype(np.float32) # [N, H, W]
        else:
            self.rgbs = np.zeros((1,), dtype=np.float32) # fake rgbs

        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()

        print("Dataset loaded:", self.rays.shape, self.rgbs.shape)
        
        if suffix == '_train':
            ids = torch.arange(self.rays.shape[0], dtype=torch.float32) # [N,]
            ids = torch.reshape(ids, [-1, 1, 1, 1]).repeat(1,H,W,1)
            for i in range(self.rays.shape[0]):
                assert torch.all(ids[i] == i)
            # ids = torch.repeat(ids,[1,H,W,1]) # [N, 1, 1, 1]
            # ids = np.tile(ids, (1,)+self.rays.shape[1:-1]+(1,)) # [N, H, W, 6]
        near = self.meta_dict['near'] * torch.ones(N,H,W,1)
        far = self.meta_dict['far'] * torch.ones(N,H,W,1)

        if suffix == '_train':
            self.rays = torch.cat([self.rays, near, far, ids], -1) # [N, H, W, ro+rd, 3+id]
            print('Done, add camera ids', self.rays.shape)
        else:
            self.rays = torch.cat([self.rays, near, far], -1) # [N, H, W, ro+rd, 3+id]
        
        # # Cast to tensors
        # self.rays = torch.from_numpy(self.rays).float()
        # self.rgbs = torch.from_numpy(self.rgbs).float()

        # Basic attributes
        self.height = self.rays.shape[1]
        self.width = self.rays.shape[2]

        self.image_count = self.rays.shape[0]
        self.image_step = self.height * self.width
        
        self.rays = self.rays.reshape([-1, self.rays.shape[-1]])
        
        self.rgbs = self.rgbs.reshape([-1, 1]).repeat(1,3)
        print("Dataset reshaped:", self.rays.shape, self.rgbs.shape)

    def num_images(self):
        return self.image_count
        
    def height_width(self):
        return self.height, self.width
                
    def near_far(self):
        return self.meta_dict['near'], self.meta_dict['far']

class BatchCryoEmDataset(CryoEmDataset, torch.utils.data.Dataset):

    def __init__(self, root_dir, split='train', subsample=0):
        torch.utils.data.Dataset.__init__(self)
        CryoEmDataset.__init__(self, root_dir, subsample, '_train' if split == 'train' else '_test')
        # CryoEmDataset.__init__(self, root_dir, subsample, '_train')
        self.split = split

    def __len__(self):
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return self.image_count * self.height * self.width

    def __getitem__(self, i):

        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.rays[i, :8],
                      'ts': self.rays[i, 8].long(),
                      'rgbs': self.rgbs[i]}
        elif self.split == 'val':
            index = random.randint(0,self.image_count-1)
            start = index * self.height * self.width
            end = (index+1) * self.height * self.width
            sample = {'rays': self.rays[start:end,:8],
                      'ts': (index * torch.ones(len(self.rays[start:end]), 1)).long(),
                      'rgbs': self.rgbs[start:end]}
        
        return sample
        
class IterativeCryoEmDataset(CryoEmDataset, torch.utils.data.IterableDataset):

    def __init__(self, root_dir, batch_size, subsample=0, testset=False, no_cam_id=False, device=torch.device("cpu")):
        torch.utils.data.IterableDataset.__init__(self)
        CryoEmDataset.__init__(self, root_dir, subsample, '_test' if testset else '_train', no_cam_id, device)

        self.batch_size = batch_size
        self.current_iter = 0
        self.image_start = -1

    def __len__(self):
        return self.image_count * self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Prohibit multiple workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise ValueError("Error IterativeCryoEMDataset does not support multi-processing")

        # Randomize index
        if self.image_start < 0:
            self.image_start = np.random.choice(self.image_count) * self.image_step

        indx = self.image_start + np.random.choice(self.image_step)
        self.current_iter += 1
        
        # If jump to the next batch
        if self.current_iter >= self.batch_size:
            self.image_start = -1
            self.current_iter = 0

        return dict(rays = self.rays[indx], target_s = self.rgbs[indx])

# Containing only rays for rendering, no rgb groundtruth
class ExhibitCryoEmDataset(CryoEmDataset, torch.utils.data.Dataset):

    def __init__(self, root_dir, subsample=0, device=torch.device("cpu")):
        torch.utils.data.Dataset.__init__(self)
        CryoEmDataset.__init__(self, root_dir, subsample, '_exhibit', True, device)

    def __len__(self):
        return self.image_count * self.height * self.width

    def __getitem__(self, i):
        return dict(rays = self.rays[i])

def load_cryoem(basedir, no_batch, batch_size, subset, subsample=0, no_cam_id=False,
                max_train_vis=10, max_test_vis=10, device=torch.device("cpu")):

    if not os.path.isdir(basedir):
        raise ValueError("No such directory containing dataset:", basedir)

    # Switch to subset
    subset_dir = os.path.join(basedir, str(subset))
    if os.path.exists(subset_dir):
        basedir = subset_dir
        print("Switch to subset:", basedir)
    else:
        print("No subset detected!")
    
    train_set = BatchCryoEmDataset(basedir, subsample=subsample, testset=False, 
                                   no_cam_id=no_cam_id, device=device)
    test_set = BatchCryoEmDataset(basedir, subsample=subsample, testset=True, 
                                  no_cam_id=True, device=device)

    H, W = train_set.height_width()
    near, far = train_set.near_far()
    extras = {}

    def pick_rays(dataset, max_vis):
#         count = dataset.num_images()
#         pick_skip = max(1, count // (max_vis-1))
#         pick_stop = min(count, pick_skip * max_vis)
#         frame_index = list(range(0, pick_stop, pick_skip))
        
        frame_indx = np.linspace(0, dataset.num_images() - 1, max_vis, dtype=np.int32)
        print("Rendered image index:", frame_indx)

        pick_indx = np.array([np.arange(i*H*W, (i+1)*H*W) for i in frame_indx])
        pick_indx = pick_indx.reshape(-1)
        
        return torch.utils.data.Subset(dataset, pick_indx)

    print("Picking rendering set ...")
    
    train_render = pick_rays(train_set, max_train_vis)
    test_render = pick_rays(test_set, max_test_vis)    
    render_sets = {'train': train_render, 'test': test_render}

    try:
        exhibit_render = ExhibitCryoEmDataset(basedir, subsample=subsample, device=device)
        render_sets['exhibit'] = exhibit_render
    except FileNotFoundError:
        print("No exhibit set!")

    if no_batch:
        train_set = IterativeCryoEmDataset(basedir, batch_size, subsample=subsample, 
                                           testset=False, no_cam_id=no_cam_id, device=device)

    extras['data_device'] = device
    extras['num_train_images'] = train_set.num_images()
    extras['num_test_images'] = test_set.num_images()
    extras['num_per_image_pixels'] = H * W
    
    print("Done, loading cryoem data.")

    return train_set, test_set, render_sets, (near, far), (H, W), extras
