import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.BiGI import BiGI
from model.Discriminator import Mixup_discriminator

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class DGITrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BiGI(opt)
        self.discriminator = Mixup_discriminator(opt)
        self.linear_dis = nn.Linear(opt["hidden_dim"], 1)
        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.linear_dis.cuda()
            self.discriminator.cuda()
            self.criterion.cuda()
        self.optimizer_D = torch_utils.get_optimizer(opt['optim'], self.discriminator.parameters(), opt['lr'])
        self.optimizer_G = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.rankingLoss = nn.MarginRankingLoss(margin=opt["margin"])
        self.epoch_dis_loss = []
        self.epoch_rec_loss = []
        self.epoch_dgi_loss = []

    def unpack_batch_predict(self, batch, cuda):
        batch = batch[0]
        if cuda:
            user_index = batch.cuda()
        else:
            user_index = batch
        return user_index

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        return user_index, item_index, negative_item_index

    def unpack_batch_DGI(self, batch, cuda):
        if cuda:
            user_index = batch[0].cuda()
            item_index = batch[1].cuda()
            negative_item_index = batch[2].cuda()
            User_index_One = batch[3].cuda()
            Item_index_One = batch[4].cuda()
            real_user_index_id_Two = batch[5].cuda()
            fake_user_index_id_Two = batch[6].cuda()
            real_item_index_id_Two = batch[7].cuda()
            fake_item_index_id_Two = batch[8].cuda()
        else:
            user_index = batch[0]
            item_index = batch[1]
            negative_item_index = batch[2]
            User_index_One = batch[3]
            Item_index_One = batch[4]
            real_user_index_id_Two = batch[5]
            fake_user_index_id_Two = batch[6]
            real_item_index_id_Two = batch[7]
            fake_item_index_id_Two = batch[8]
        return user_index, item_index, negative_item_index, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two

    def predict(self, batch):
        User_One = self.unpack_batch_predict(batch, self.opt["cuda"])  # 1

        Item_feature = torch.index_select(self.item_hidden_out, 0, self.model.item_index) # item_num * hidden_dim
        User_feature = torch.index_select(self.user_hidden_out, 0, User_One) # User_num * hidden_dim
        User_feature = User_feature.unsqueeze(1)
        User_feature = User_feature.repeat(1, self.opt["number_item"], 1)
        Item_feature = Item_feature.unsqueeze(0)
        Item_feature = Item_feature.repeat(User_feature.size()[0], 1, 1)
        Feature = torch.cat((User_feature, Item_feature),
                            dim=-1)
        output = self.model.score_predict(Feature)
        output_list, recommendation_list = output.sort(descending=True)
        return recommendation_list.cpu().numpy()

    def feature_corruption(self):
        user_index = torch.randperm(self.opt["number_user"], device=self.model.user_index.device)
        item_index = torch.randperm(self.opt["number_item"], device=self.model.user_index.device)
        user_feature = self.model.user_embedding(user_index)
        item_feature = self.model.item_embedding(item_index)
        # user_feature = self.model.user_embed_fake(user_feature)
        # item_feature = self.model.item_embed_fake(item_feature)
        return user_feature, item_feature

    def update_bipartite(self, UV_adj, VU_adj, adj,fake = 0):
        if fake:
            user_feature, item_feature = self.feature_corruption()
            user_feature = user_feature.detach()
            item_feature = item_feature.detach()                     
        else :
            user_feature = self.model.user_embedding(self.model.user_index)
            item_feature = self.model.item_embedding(self.model.item_index)
            # user_feature = self.model.user_embed(user_feature)
            # item_feature = self.model.item_embed(item_feature)
        self.user_hidden_out, self.item_hidden_out = self.model(user_feature, item_feature, UV_adj, VU_adj, adj, fake)
        
    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct(self, UV, VU, UV_rated, VU_rated, relation_UV_adj, relation_VU_adj, adj, CUV, CVU, fake_adj, batch, i):
        self.discriminator.train()
        self.optimizer_D.zero_grad()

        virtual_user_hidden_out = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (self.opt["number_user"], self.opt["hidden_dim"]))))
        virtual_item_hidden_out = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (self.opt["number_item"], self.opt["hidden_dim"]))))
        virtual_fake_user_hidden_out = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (self.opt["number_user"], self.opt["hidden_dim"]))))
        virtual_fake_item_hidden_out = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (self.opt["number_item"], self.opt["hidden_dim"]))))
        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            user_One, item_One, neg_item_One, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two  = self.unpack_batch_DGI(batch, self.opt[
                "cuda"])
        else :
            user_One, item_One, neg_item_One = self.unpack_batch(batch, self.opt[
                "cuda"])

        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            real_user_index_id_Three = self.my_index_select(user_hidden_out, real_user_index_id_Two)
            real_item_index_id_Three = self.my_index_select(item_hidden_out, real_item_index_id_Two)
            fake_user_index_id_Three = self.my_index_select(fake_user_hidden_out, fake_user_index_id_Two)
            fake_item_index_id_Three = self.my_index_select(fake_item_hidden_out, fake_item_index_id_Two)

            real_user_index_feature_Two = self.my_index_select(user_hidden_out, User_index_One)
            real_item_index_feature_Two = self.my_index_select(item_hidden_out, Item_index_One)
            fake_user_index_feature_Two = self.my_index_select(fake_user_hidden_out, User_index_One)
            fake_item_index_feature_Two = self.my_index_select(fake_item_hidden_out, Item_index_One)

            mixup_real, mixup_fake = self.model.DGI(user_hidden_out, item_hidden_out, real_user_index_feature_Two, real_item_index_feature_Two, fake_user_index_feature_Two, fake_item_index_feature_Two,real_item_index_id_Three,real_user_index_id_Three,fake_item_index_id_Three,fake_user_index_id_Three)

            dgi_loss = self.criterion(Prob, Label)
            loss = (1 - self.opt["lambda"])*reconstruct_loss + self.opt["lambda"] * dgi_loss
            self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
            self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())


        else :
            virtual_real, virtual_fake = self.model.DGI(virtual_user_hidden_out, virtual_item_hidden_out, virtual_fake_user_hidden_out,
                                            virtual_fake_item_hidden_out, UV, VU, CUV, CVU, user_One, item_One, UV_rated, VU_rated,
                                            relation_UV_adj, relation_VU_adj)
            virtual_real = virtual_real.detach()
            virtual_fake = virtual_fake.detach()
            real_label = torch.ones_like(virtual_real)
            fake_label = torch.zeros_like(virtual_fake)

            discriminator_loss = self.criterion(self.discriminator(virtual_real), real_label) + self.criterion(self.discriminator(virtual_fake), fake_label)
            discriminator_loss.backward()
            self.optimizer_D.step()
            
            for p in self.discriminator.parameters():
                p.data.clamp(-0.01, 0.01)   
            totalLoss = discriminator_loss.item()
            self.epoch_dis_loss.append(totalLoss)
            
            if i%5==0:
                self.model.train()
                self.optimizer_G.zero_grad()

                self.update_bipartite(CUV, CVU, fake_adj, fake = 1)
                fake_user_hidden_out = self.user_hidden_out
                fake_item_hidden_out = self.item_hidden_out

                self.update_bipartite(UV, VU, adj)
                user_hidden_out = self.user_hidden_out
                item_hidden_out = self.item_hidden_out

                user_feature_Two = self.my_index_select(user_hidden_out, user_One)
                item_feature_Two = self.my_index_select(item_hidden_out, item_One)
                neg_item_feature_Two = self.my_index_select(item_hidden_out, neg_item_One)

                pos_One = self.model.score(torch.cat((user_feature_Two, item_feature_Two), dim=1))
                neg_One = self.model.score(torch.cat((user_feature_Two, neg_item_feature_Two), dim=1))    

                if self.opt["wiki"]:
                    Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One))).cuda()
                    pre = torch.cat((pos_One, neg_One))
                    reconstruct_loss = self.criterion(pre, Label)
                else:
                    reconstruct_loss = self.rankingLoss(pos_One, neg_One, torch.tensor([1]).cuda())  
                
                mixup_real, mixup_fake = self.model.DGI(self.user_hidden_out, self.item_hidden_out, fake_user_hidden_out,
                                        fake_item_hidden_out, UV, VU, CUV, CVU, user_One, item_One, UV_rated, VU_rated,
                                        relation_UV_adj, relation_VU_adj)
                
                real_sub_prob = self.discriminator(mixup_real)
                fake_sub_prob = self.discriminator(mixup_fake)
                
                dgi_loss = -torch.mean(self.discriminator(real_sub_prob)) + torch.mean(self.discriminator(fake_sub_prob))
                generator_loss = (1 - self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * dgi_loss
                generator_loss.backward()
                self.optimizer_G.step()
                self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
                self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())
                totalLoss = discriminator_loss.item() + generator_loss.item()
            
            return totalLoss