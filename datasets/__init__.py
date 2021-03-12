from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset
from .cryoem import BatchCryoEmDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                'cryoem':BatchCryoEmDataset}