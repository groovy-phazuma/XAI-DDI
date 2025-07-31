#!/usr/bin/env python3
"""
Created on 2025-07-31 (Thu) 17:42:29

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_INIT_EPS = 2.0
gamma = 12.0


class GCNModel(nn.Module):

    def __init__(self, args, n_entities, n_relations, entity_pre_embed=None, relation_pre_embed=None,
                 structure_pre_embed=None):

        super(GCNModel, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.structure_dim = args.structure_dim
        self.pre_entity_dim = args.pre_entity_dim

        self.fusion_type = args.feature_fusion
        self.multi_type = args.multi_type

        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.ddi_l2loss_lambda = args.DDI_l2loss_lambda

        self.hidden_dim = args.entity_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / self.hidden_dim

        # fusion type
        if self.fusion_type == 'concat':

            self.layer1_f = nn.Sequential(nn.Linear(self.structure_dim + self.entity_dim, self.entity_dim),
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer2_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer3_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))

        elif self.fusion_type == 'sum':

            self.W_s = nn.Linear(self.structure_dim, self.entity_dim)
            self.W_e = nn.Linear(self.entity_dim, self.entity_dim)

        elif self.fusion_type == 'init_double':

            self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
            self.druglayer_KG = nn.Linear(self.entity_dim, self.entity_dim)

            self.add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.cross_add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.activate = nn.ReLU()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            if self.entity_dim == 300:
                self.fc1 = nn.Sequential(nn.Linear(72 * 72 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 400:  # NOTE: add
                self.fc1 = nn.Sequential(nn.Linear(97 * 97 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 200:  # NOTE: add
                self.fc1 = nn.Sequential(nn.Linear(47 * 47 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 256:
                self.fc1 = nn.Sequential(nn.Linear(61 * 61 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 128:
                self.fc1 = nn.Sequential(nn.Linear(29 * 29 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 100:
                self.fc1 = nn.Sequential(nn.Linear(22 * 22 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 32:
                self.fc1 = nn.Sequential(nn.Linear(5 * 5 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 64:
                self.fc1 = nn.Sequential(nn.Linear(13 * 13 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))

            self.fc2_global = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_global_reverse = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_cross = nn.Sequential(
                nn.Linear(self.entity_dim * 4, self.entity_dim),
                nn.ReLU(True))

        if (self.use_pretrain == 1) and (structure_pre_embed is not None):

            self.n_approved_drug = structure_pre_embed.shape[0]
            self.structure_pre_embed = structure_pre_embed

            if self.fusion_type in ['init_double', 'sum', 'concat']:
                self.pre_entity_embed = entity_pre_embed

        if self.fusion_type in ['double', 'init_double']:
            self.all_embedding_dim = (self.entity_dim * 3 + self.structure_dim + self.entity_dim) * 2

        elif self.fusion_type in ['sum', 'concat']:
            self.all_embedding_dim = self.entity_dim * 2
        self.layer1 = nn.Sequential(nn.Linear(self.all_embedding_dim, args.n_hidden_1), nn.BatchNorm1d(args.n_hidden_1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(args.n_hidden_1, args.n_hidden_2), nn.BatchNorm1d(args.n_hidden_2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(args.n_hidden_2, args.out_dim))

    def generate_fusion_feature(self, embedding_pre, embedding_after, batch_data, epoch):
        # we focus on approved drug
        global embedding_data
        global embedding_data_reverse

        entity_embed_pre = self.pre_entity_embed[:self.n_approved_drug, :]

        if self.fusion_type == 'concat':

            x = torch.cat([self.structure_pre_embed, entity_embed_pre], dim=1)
            x = self.layer1_f(x)
            x = self.layer2_f(x)
            x = self.layer3_f(x)

            return x

        elif self.fusion_type == 'sum':

            structure = self.W_s(self.structure_pre_embed)
            entity = self.W_e(entity_embed_pre)
            add_structure_entity = structure + entity

            return add_structure_entity

        elif self.fusion_type == 'init_double':

            structure = self.druglayer_structure(self.structure_pre_embed)

            entity = self.druglayer_KG(entity_embed_pre)

            structure_embed_reshape = structure.unsqueeze(-1)  # batch_size * embed_dim * 1
            entity_embed_reshape = entity.unsqueeze(-1)  # batch_size * embed_dim * 1

            entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim e.g. torch.Size([2322, 400, 400])

            entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_global = entity_matrix.view(entity_matrix.size(0), -1)  # torch.Size([2322, 160000])

            entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

            entity_matrix_reshape = entity_matrix.unsqueeze(1)

            for i, data in enumerate(batch_data):

                entity_matrix_reshape = entity_matrix_reshape.to('cuda')
                entity_data = entity_matrix_reshape.index_select(0, data[0].to('cuda'))

                out = self.conv1(entity_data)
                out = self.conv2(out)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)

                if i == 0:
                    embedding_data = out
                else:
                    embedding_data = torch.cat((embedding_data, out), 0)

            global_local_before = torch.cat((embedding_data, entity_global), 1)
            cross_embedding_pre = self.fc2_global(global_local_before)

            # another reverse part

            entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)

            for i, data in enumerate(batch_data):

                entity_matrix_reshape_reverse = entity_matrix_reshape_reverse.to('cuda')
                entity_reverse = entity_matrix_reshape_reverse.index_select(0, data[0].to('cuda'))

                out = self.conv1(entity_reverse)  # [1, 8, 198, 198]

                out = self.conv2(out)  # [1, 8, 97, 97]

                out = out.view(out.size(0), -1)  #[1, 75272]

                out = self.fc1(out)

                if i == 0:
                    embedding_data_reverse = out
                else:
                    embedding_data_reverse = torch.cat((embedding_data_reverse, out), 0)

            global_local_before_reverse = torch.cat((embedding_data_reverse, entity_global_reverse), 1)
            cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)

            out3 = self.activate(self.multi_drug(structure * entity))

            out_concat = torch.cat(
                (self.structure_pre_embed, entity_embed_pre, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)

            return out_concat

    def train_DDI_data(self, mode, g, train_data, embedding_pre, embedding_after, batch_data, epoch):

        all_embed = self.generate_fusion_feature(embedding_pre, embedding_after, batch_data, epoch)

        drug1_embed = all_embed[train_data[:, 0]]
        drug2_embed = all_embed[train_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def test_DDI_data(self, mode, g, test_data, embedding_pre, embedding_after, batch_data, epoch):

        all_embed = self.generate_fusion_feature(embedding_pre, embedding_after, batch_data, epoch)
        drug1_embed = all_embed[test_data[:, 0]]
        drug2_embed = all_embed[test_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.multi_type != 'False':
            pred = F.softmax(x, dim=1)
        else:
            pred = torch.sigmoid(x)
        return pred, all_embed

    def forward(self, mode, *input):

        if mode == 'calc_ddi_loss':
            return self.train_DDI_data(mode, *input)
        if mode == 'predict':
            return self.test_DDI_data(mode, *input)
        if mode == 'feature_fusion':
            return self.generate_fusion_feature(*input)
