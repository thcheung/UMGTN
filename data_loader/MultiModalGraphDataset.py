import os
import json
import shutil
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from transformers import AutoTokenizer, ViTFeatureExtractor, BertTokenizer, CLIPProcessor, RobertaTokenizer
from tqdm import tqdm
from utils import preprocess, image_transforms
from PIL import Image
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiData(Data):
    def is_node_attr(self, key):
        if key in ['image', 'image_key','source']:
            return False
        else:
            return super().is_node_attr(key)

    def is_edge_attr(self, key):
        if key in ['image', 'image_key','source']:
            return False
        else:
            return super().is_edge_attr(key)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['image', 'image_key','source']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class MultiModalGraphDataset(Dataset):
    def __init__(self, root, image_dir, split, classes, language='en', max_length=64, transform=None, pre_transform=None):

        self.split = split
        self.filename = "{}.json".format(split)

        self.classes = classes
        self.language = language
        self.root = root
        self.image_dir = image_dir
        self.max_length = max_length
        if self.language == "cn":
            self.max_nodes = 100 if self.split == "train" else 80
        elif self.language == "en":
            self.max_nodes = 80 if self.split == "train" else 60

        self.textTokenizer = self._get_tokenizer()
        self.image_transforms = image_transforms()
        self.imageTokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        super(MultiModalGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        data_len = (len(self.data))
        return [f'data_{self.split}_{i}.pt' for i in range(data_len)]

    def download(self):
        download_path = self.raw_dir
        os.makedirs(download_path,exist_ok=True)
        file_path = os.path.join(self.root, self.filename)
        shutil.copy(file_path,download_path)

    def process(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        for index, tweet in (enumerate(tqdm(self.data))):
            tweet["nodes"] = tweet["nodes"][:self.max_nodes]
            tweet["edges"] = tweet["edges"][:self.max_nodes-1]
            tweet_id = tweet['id']
            
            node_feats = self._get_node_features(tweet["nodes"])
            edge_index = self._get_adjacency_info(tweet["edges"])
            
            label = self._get_labels(tweet['label'])

            image, image_mask = self._get_image_features(tweet["image"])

            data = MultiData(x=node_feats,
                        edge_index = edge_index,
                        y=label,
                        image=image,
                        image_mask = image_mask,
                        id=tweet_id,
                        )

            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{self.split}_{index}.pt'))

    def _get_tokenizer(self):
        if self.language == 'en':
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.language == 'cn':
            return AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    def _get_node_features(self, nodes):
        texts = [preprocess(node['text']) for node in nodes]
        encoded_input = self.textTokenizer.batch_encode_plus(
            texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')
        all_node_feats = torch.stack([
            encoded_input["input_ids"], encoded_input["attention_mask"]], dim=-1)
        return all_node_feats

    def _get_edge_features(self, edge_len):
        return torch.ones(edge_len, 1)

    def _get_adjacency_info(self, edges):
        edge_indices = []
        
        for edge in edges:
            i = edge['from']
            j = edge['to']
            edge_indices += [[i, i]]
            edge_indices += [[i, j]]
            edge_indices += [[j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices

    def _get_labels(self, label):
        label = self.classes.index(label)
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int32)

    def _get_image_features(self, image_id):

        image_path = os.path.join(self.image_dir, image_id)

        if image_id == None or image_id == '' or not os.path.exists(image_path):
            image = torch.zeros(3, 224, 224)
            image_mask = torch.ones(1)
       
        else:
            image = Image.open(image_path).convert("RGB")
            image = self.imageTokenizer(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
            image_mask = torch.zeros(1)

        return image, image_mask

    def len(self):
        return len(self.data)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir,
                          f'data_{self.split}_{idx}.pt'))
    
        return data