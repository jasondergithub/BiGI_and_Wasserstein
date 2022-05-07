import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GAT import GAT

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class Extract_Overall(nn.Module):
    def __init__(self, opt):
        super(Extract_Overall, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.weight_matrix = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
    def forward(self, feature, adj):
        h = self.weight_matrix(feature)
        output = torch.mm(adj.to_dense(), h)
        output =self.relu(output)
        finalOutput = torch.mean(output, 0)
        return finalOutput

class Transformer_discriminator(nn.Module):
    def __init__(self, d_model):
        super(Transformer_discriminator, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dropout=0.3)
        self.Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.lin = nn.Linear(d_model, 1)
    
    def forward(self, concat_vector): #concat_vector: [128, 64]
        concat_vector = torch.unsqueeze(concat_vector, 0)
        output = self.Encoder(concat_vector)
        output = torch.squeeze(output, 0)
        output = self.lin(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_in,n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        S = S.expand_as(node) # batch * hidden_dim
        score = torch.squeeze(self.f_k(node, S),1) # batch
        if s_bias is not None:
            score += s_bias

        return score

class myDGI(nn.Module):
    def __init__(self, opt):
        super(myDGI, self).__init__()
        self.opt = opt
        self.read = AvgReadout()
        self.extract = Extract_Overall(opt)
        self.att = GAT(opt)
        self.sigm = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        self.lin1 = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.lin2 = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])          
        self.lin = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.lin_sub = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.linear_dis = nn.Linear(opt["hidden_dim"], 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    #
    def forward(self, user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out, UV_adj, VU_adj, CUV_adj, CVU_adj, user_One, item_One,
        UV_rated, VU_rated, relation_UV_adj, relation_VU_adj,
        msk=None, samp_bias1=None, samp_bias2=None):        

        S_u_One = self.read(user_hidden_out, msk)  # hidden_dim
        S_i_One = self.read(item_hidden_out, msk)  # hidden_dim
        Global_item_cor2_user = self.extract(item_hidden_out, UV_rated)
        Global_user_cor2_item = self.extract(user_hidden_out, VU_rated)   
        g = self.lin1(torch.cat((S_u_One, Global_item_cor2_user)).unsqueeze(0))
        h = self.lin2(torch.cat((Global_user_cor2_item, S_i_One)).unsqueeze(0))    
        S_Two = g + h     
        S_Two = torch.div(S_Two, 2)
        S_Two = self.sigm(S_Two)  # hidden_dim  need modify   
        S_Two = self.lin(S_Two) # 1 * hidden_dim

        real_user, real_item = self.att(user_hidden_out, item_hidden_out, UV_adj, VU_adj)
        fake_user, fake_item = self.att(fake_user_hidden_out, fake_item_hidden_out, CUV_adj, CVU_adj)

        real_user_index_feature_Two = torch.index_select(real_user, 0, user_One)
        real_item_index_feature_Two = torch.index_select(real_item, 0, item_One)
        fake_user_index_feature_Two = torch.index_select(fake_user, 0, user_One)
        fake_item_index_feature_Two = torch.index_select(fake_item, 0, item_One)
        real_sub_Two = self.lin_sub(torch.cat((real_user_index_feature_Two, real_item_index_feature_Two),dim = 1))
        # real_sub_Two = self.sigm(real_sub_Two)

        fake_sub_Two = self.lin_sub(torch.cat((fake_user_index_feature_Two, fake_item_index_feature_Two),dim = 1))
        # fake_sub_Two = self.relu(fake_sub_Two)

        # real_sub_prob = self.disc(S_Two, real_sub_Two)
        # fake_sub_prob = self.disc(S_Two, fake_sub_Two)
        mixup_real = torch.add(S_Two, real_sub_Two)
        # mixup_real = self.relu(mixup_real)
        mixup_fake = torch.add(S_Two, fake_sub_Two)
        # mixup_fake = self.relu(mixup_fake)

        # real_sub_prob = self.linear_dis(mixup_real)
        # fake_sub_prob = self.linear_dis(mixup_fake)

        # prob = torch.cat((real_sub_prob, fake_sub_prob))
        # label = torch.cat((torch.ones_like(real_sub_prob), torch.zeros_like(fake_sub_prob)))

        # return prob, label

        return mixup_real, mixup_fake
