from torch_geometric.nn import TransformerConv,global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, ViTModel, BertModel, CLIPVisionModel,RobertaModel
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch.nn.utils.rnn import pad_sequence
from utils import length_to_mask, pad_tensor
from torch_geometric.utils import to_dense_adj, add_self_loops


class MultiModalGraphModel(nn.Module):
    def __init__(self, hidden_dim=768, label_dim=3, dropout_rate=0.1, language='en'):
        super(MultiModalGraphModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language
        self.label_dim = label_dim

        self.textEncoder = self.get_text_model()
        self.freeze_textEncoder()

        self.imageEncoder = self.get_image_model()
        self.freeze_imageEncoder()

        self.transformer_source = TransformerEncoder(TransformerEncoderLayer(hidden_dim, 2, hidden_dim, batch_first=True), 1)
        self.transformer_graph = TransformerEncoder(TransformerEncoderLayer(hidden_dim, 2, hidden_dim, batch_first=True), 1)
        self.transformer_multi = TransformerEncoder(TransformerEncoderLayer(hidden_dim, 2, hidden_dim, batch_first=True), 1)

        self.token_type_embeddings = nn.Embedding(3, 768)

        self.dropout = nn.Dropout(p=dropout_rate)
     
        self.pooler_t = nn.Linear(hidden_dim,hidden_dim)
        self.pooler_i = nn.Linear(hidden_dim,hidden_dim)
        self.pooler_x = nn.Linear(hidden_dim,hidden_dim)

        self.pooler_source = nn.Linear(hidden_dim,hidden_dim)
        self.pooler_graph = nn.Linear(hidden_dim,hidden_dim)
        self.pooler_multi = nn.Linear(hidden_dim,hidden_dim)

        self.fc_source = nn.Linear(hidden_dim, label_dim)
        self.fc_graph = nn.Linear(hidden_dim, label_dim)
        self.fc_multi = nn.Linear(hidden_dim, label_dim)

    def get_text_model(self):
        if self.language == 'en':
            return AutoModel.from_pretrained("bert-base-uncased")
        elif self.language == 'cn':
            return AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")

    def get_image_model(self):
            return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def freeze_textEncoder(self):
        for name, param in list(self.textEncoder.named_parameters()):
            if self.language == 'en':
                if 'pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            elif self.language == 'cn':
                if 'pooler' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def freeze_imageEncoder(self):
        for name, param in list(self.imageEncoder.named_parameters()):
            if name.startswith('pooler'):
                continue
            else:
                param.requires_grad = False

    def forward(self, data):
        
        x,  batch_index = data.x, data.batch

        image, image_mask = data.image, data.image_mask

        # passing the source-reply graph features
        x_id = x[:, :, 0].int()
        x_mask = x[:, :, 1]
        x = self.textEncoder(input_ids=x_id, attention_mask=x_mask)

        # obtaining the reply graph features
        x = x['last_hidden_state']

        x , _ = pad_tensor(x, batch_index)
        x_mask , _ = pad_tensor(x_mask, batch_index)
        graph_mask = to_dense_adj(edge_index=data.edge_index,batch=batch_index)

        graph_mask = graph_mask.repeat(2,1,1)

        t = x[:,0]
        t_mask = x_mask[:,0]
        t_mask = ~(t_mask.int().bool())

        x = x[:,:,0]
        x_mask = x_mask[:,:,0]
        x_mask = ~(x_mask.int().bool())

        # passing the image features
        i = self.imageEncoder(pixel_values=image,output_hidden_states=True)

        i = i['last_hidden_state']
        i_mask = image_mask.unsqueeze(1).expand(-1,i.size(1))

        t = torch.tanh(self.pooler_t(t))
        i = torch.tanh(self.pooler_i(i))
        x = torch.tanh(self.pooler_x(x))

        x0  = torch.cat([t,i],dim=1)
        x_mask0 = torch.cat([t_mask,i_mask],dim=1)
        x0 = self.transformer_source(x0,src_key_padding_mask=x_mask0)

        x1 = torch.cat([x],dim=1)
        x_mask1 = torch.cat([x_mask],dim=1)
        x1 = self.transformer_graph(x1,mask=graph_mask,src_key_padding_mask=x_mask1)

        x2 = torch.cat([x0, x1],dim=1)
        x_mask2 = torch.cat([x_mask0,x_mask1],dim=1)
        x2 = self.transformer_multi(x2,src_key_padding_mask=x_mask2)

        x0 = torch.tanh(self.pooler_multi(x0[:,0]))
        x1 = torch.tanh(self.pooler_multi(x1[:,0]))
        x2 = torch.tanh(self.pooler_multi(x2[:,0]))

        return self.fc_multi(x0) , self.fc_multi(x1), self.fc_multi(x2)