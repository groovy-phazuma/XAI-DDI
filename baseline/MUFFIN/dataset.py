#!/usr/bin/env python3
"""
Created on 2025-07-31 (Thu) 17:35:08

@author: I.Azuma
"""
# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# loading data
class DataLoaderSKGDDI(object):

    def __init__(self, args):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain

        self.ddi_batch_size = args.DDI_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.multi_type = args.multi_type

        self.entity_dim = args.entity_dim

        data_dir = os.path.join(args.data_dir, args.data_name)
        if self.multi_type == 'True':
            print('multi_type is True')
            train_file = os.path.join(data_dir, 'multi_ddi_sift.txt')
        else:
            print('multi_type is False')
            train_file = os.path.join(data_dir, 'DDI_pos_neg.txt')

        """
        if args.data_name == 'DRKG':
            kg_file = os.path.join(data_dir, "train.tsv")
        else:
            kg_file = os.path.join(data_dir, "kg2id.txt")
        """
        kg_file = args.kg_file

        self.DDI_train_data_X, self.DDI_train_data_Y, self.DDI_test_data_X, self.DDI_test_data_Y = self.load_DDI_data(
            train_file)

        self.statistic_ddi_data()

        triples = self.read_triple(kg_file)
        self.construct_triples(triples)

        self.print_info()

        self.train_graph = None
        if self.use_pretrain == 1:
            self.load_pretrained_data()

    def load_DDI_data(self, filename):

        train_X_data = []
        train_Y_data = []
        test_X_data = []
        test_Y_data = []

        traindf = pd.read_csv(filename, delimiter='\t', header=None)
        data = traindf.values
        DDI = data[:, 0:2]
        # 1123100,2
        Y = data[:, 2]
        label = np.array(list(map(int, Y)))

        print(DDI.shape)
        print(label.shape)

        kfold = KFold(n_splits=5, shuffle=True, random_state=3)

        for train, test in kfold.split(DDI, label):
            train_X_data.append(DDI[train])
            train_Y_data.append(label[train])
            test_X_data.append(DDI[test])
            test_Y_data.append(label[test])

        """
        train_X = np.array(train_X_data)
        train_Y = np.array(train_Y_data)
        test_X = np.array(test_X_data)
        test_Y = np.array(test_Y_data)
        """

        print('Loading DDI data down!')

        return train_X_data, train_Y_data, test_X_data, test_Y_data

    # 5-fold train data length
    def statistic_ddi_data(self):
        data = []
        for i in range(len(self.DDI_train_data_X)):
            data.append(len(self.DDI_train_data_X[i]))
        self.n_ddi_train = data

    def read_triple(self, path, mode='train', skip_first_line=False, format=[0, 1, 2]):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split('\t')
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                try:
                    heads.append(int(h))
                    tails.append(int(t))
                    rels.append(int(r))
                except ValueError:
                    print("For User Defined Dataset, both node ids and relation ids in the triplets should be int "
                          "other than {}\t{}\t{}".format(h, r, t))
                    raise
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return heads, rels, tails

    # load kg triple
    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_triples(self, kg_data):
        print("construct kg...")
        src, rel, dst = kg_data

        src, rel, dst = np.concatenate((src, dst)), np.concatenate((rel, rel)), np.concatenate((dst, src))
        self.kg_triple = np.array(sorted(zip(src, rel, dst)))
        # print(self.kg_triple.shape)

        self.n_relations = max(rel) + 1
        self.n_entities = max(max(src), max(dst)) + 1

        self.kg_train_data = self.kg_triple
        self.n_kg_train = len(self.kg_train_data)

        print('construct kg down!')

    def print_info(self):
        print('n_entities:         %d' % self.n_entities)
        print('n_relations:        %d' % self.n_relations)
        print('n_kg_train:         %d' % self.n_kg_train)
        print('n_ddi_train:         %s' % self.n_ddi_train)

    def load_pretrained_data(self):

        if self.data_name == 'DrugBank':

            # load pretrained KG information

            #transE_entity_path = 'embedding_data/entityVector_400.npz'
            #transE_relation_path = 'embedding_data/relationVector_400.npz'
            transE_entity_path = self.args.entity_embedding_file
            transE_relation_path = self.args.relation_embedding_file

            transE_entity_data = np.load(transE_entity_path)
            transE_relation_data = np.load(transE_relation_path)
            #transE_entity_data = transE_entity_data['embed']
            #transE_relation_data = transE_relation_data['embed']

            # load pretrained Structure information
            masking_entity_path = self.args.graph_embedding_file
            masking_entity_data = np.load(masking_entity_path)

        else:
            # change name by yourself.

            # if self.entity_dim == 300:
            #     transE_entity_path = 'data/DRKG/TransE_l2_DRKG_0/DRKG_TransE_l2_entity_300.npy'
            #     transE_relation_path = 'data/DRKG/TransE_l2_DRKG_0/DRKG_TransE_l2_relation_300.npy'
            # elif self.entity_dim == 256:
            #     transE_entity_path = 'ckpts/TransE_l2_DRKG_19/DRKG_TransE_l2_entity.npy'
            #     transE_relation_path = 'ckpts/TransE_l2_DRKG_19/DRKG_TransE_l2_relation.npy'
            # elif self.entity_dim == 100:
            #     # 128 negative sample
            #     transE_entity_path = 'data/DRKG/DRKG_TransE_l2_entity.npy'
            #     transE_relation_path = 'data/DRKG/DRKG_TransE_l2_relation.npy'
            # elif self.entity_dim == 128:
            #     transE_entity_path = 'data/DRKG/DRKG_TransE_l2_entity_128.npy'
            #     transE_relation_path = 'data/DRKG/DRKG_TransE_l2_relation_128.npy'
            # elif self.entity_dim == 32:
            #     transE_entity_path = 'data/DRKG/32/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy'
            #     transE_relation_path = 'data/DRKG/32/TransE_l2_DRKG_0/DRKG_TransE_l2_relation.npy'
            # elif self.entity_dim == 64:
            #     transE_entity_path = 'data/DRKG/64/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy'
            #     transE_relation_path = 'data/DRKG/64/TransE_l2_DRKG_0/DRKG_TransE_l2_relation.npy'

            transE_entity_path = self.args.entity_embedding_file
            transE_relation_path = self.args.relation_embedding_file

            transE_entity_data = np.load(transE_entity_path)
            transE_relation_data = np.load(transE_relation_path)

            masking_entity_path = self.args.graph_embedding_file
            masking_entity_data = np.load(masking_entity_path)

        # apply pretrained data

        self.entity_pre_embed = transE_entity_data
        self.relation_pre_embed = transE_relation_data

        self.structure_pre_embed = masking_entity_data

        self.n_approved_drug = self.structure_pre_embed.shape[0]

        print('loading pretrain data down!')
