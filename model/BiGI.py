import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.GNN2 import GNN2
from model.AttDGI import AttDGI
from model.myDGI import myDGI
from model.pretrained_ml100k import pretrained, genreEncoder, genreDecoder

class BiGI(nn.Module):
    def __init__(self, opt):
        super(BiGI, self).__init__()
        self.opt=opt
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            self.DGI = AttDGI(opt) # Since pytorch is not support sparse matrix well
        else :
            self.DGI = myDGI(opt) # Since pytorch is not support sparse matrix well
        self.dropout = opt["dropout"]

        # load pretrained user feature and item feature
        # load pretrained model
        # --------------------------------------------------------------------------------------------
        pretrained_filter = pretrained(genreEncoder(), genreDecoder())
        pretrained_filter.load_state_dict(torch.load('./load_pt/pretrained_filter.pt'))
        with open('./load_pt/user_feature_init', 'rb') as fp:
            init_user_feature = pickle.load(fp) # type: list
        user_feature = []
        for i in range(len(init_user_feature)):
            output = pretrained_filter(torch.tensor(init_user_feature[i], dtype=torch.float).unsqueeze(0))
            user_feature.append(output) # list of tensors
        user_feature = torch.stack(user_feature) 
        user_feature = torch.squeeze(user_feature, 1) # (943, 18)

        with open('./load_pt/item_feature_init', 'rb') as fp:
            init_item_feature = pickle.load(fp) # type: list

        item_key = set()
        with open('./load_pt/interaction_list', 'rb') as fp:
            interaction_list = pickle.load(fp)
        for key in interaction_list:
            item_key.add(key[1])
        item_key = sorted(item_key)
        item_key = list(item_key)

        item_feature = []
        for key in item_key:
            item_feature.append(torch.tensor(init_item_feature[key-1])) # list of tensors
        item_feature = torch.stack(item_feature) # (1650, 18)    
        user_embedding_parameter = nn.Parameter(user_feature)
        item_embedding_parameter = nn.Parameter(item_feature)    
        # --------------------------------------------------------------------------------------------
        self.user_embedding = nn.Embedding(opt["number_user"], opt["feature_dim"])
        # self.user_embedding.weight = user_embedding_parameter
        self.item_embedding = nn.Embedding(opt["number_item"], opt["feature_dim"])
        # self.item_embedding.weight = item_embedding_parameter
        self.user_embed = nn.Linear(opt['feature_dim'], opt["hidden_dim"])
        self.item_embed = nn.Linear(opt['feature_dim'], opt["hidden_dim"])   
        self.user_embed_fake = nn.Linear(opt['feature_dim'], opt["hidden_dim"])
        self.item_embed_fake = nn.Linear(opt['feature_dim'], opt["hidden_dim"])              
        self.item_index = torch.arange(0, self.opt["number_item"], 1)
        self.user_index = torch.arange(0, self.opt["number_user"], 1)
        if self.opt["cuda"]:
            self.item_index = self.item_index.cuda()
            self.user_index = self.user_index.cuda()

    def score_predict(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        # out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        # out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj, fake):
        if fake:
            ufea = self.user_embed_fake(ufea)
            vfea = self.item_embed_fake(vfea)
        else:
            ufea = self.user_embed(ufea)
            vfea = self.item_embed(vfea)
        # ufea = self.user_embed(ufea)
        # vfea = self.item_embed(vfea)
        learn_user,learn_item = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_user,learn_item
