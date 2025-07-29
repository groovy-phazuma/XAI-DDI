#!/usr/bin/env python3
"""
Created on 2025-07-29 (Tue) 18:43:50

@author: I.Azuma
"""
# %%
import pandas as pd

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def generate_entity_relation_id_file(
        delimiter='\t', 
        smilesfile='./data/DRKG/drugname_smiles.txt',
        combine_file='./data/DRKG/train_without_ddi_raw.tsv', 
        drkg_file='./data/DRKG/drkg.tsv', 
        entity_file='./data/DRKG/entities.tsv', 
        relation_file='./data/DRKG/relations.tsv', 
        triple_file='./data/DRKG/train.tsv'
        ):
    entity_map = {}
    rel_map = {}
    train = []

    infile = open(smilesfile, 'r')
    approved_drug_list = set()
    for line in infile:
        drug = line.replace('\n', '').replace('\r', '').split('\t')
        # approved_drug_list.add('Compound::' + drug[0])
        # drug_id = _get_id(entity_map, 'Compound::' + drug[0])
        approved_drug_list.add(drug[0])
        drug_id = _get_id(entity_map, drug[0])
        print(drug, drug_id)

    df = pd.read_csv(combine_file, sep="\t", header=None)
    triples = df.values.tolist()

    for i in range(len(triples)):
        src, rel, dst = triples[i][0], triples[i][1], triples[i][2]
        src_id = _get_id(entity_map, src)
        dst_id = _get_id(entity_map, dst)
        rel_id = _get_id(rel_map, rel)
        train_id = "{}{}{}{}{}\n".format(src_id, delimiter, rel_id, delimiter, dst_id)
        print(train_id)
        train.append(train_id)

    entities = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(entity_map.items(), key=lambda x: x[1])]
    with open(entity_file, "w+") as f:
        f.writelines(entities)
    n_entities = len(entities)

    relations = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(rel_map.items(), key=lambda x: x[1])]
    with open(relation_file, "w+") as f:
        f.writelines(relations)
    n_relations = len(relations)

    with open(triple_file, "w+") as f:
        f.writelines(train)
    n_kg = len(train)

    # the code down from here is just extract DDI pairs from DRKG and transfer it into id form.
    df_2 = pd.read_csv(drkg_file, sep="\t", header=None)
    triples2 = df_2.values.tolist()

    ddi_name = []
    ddi = []
    ddi_name_list = set()
    for i in range(len(triples2)):
        src, rel, dst = triples2[i][0], triples2[i][1], triples2[i][2]
        if rel in ['DRUGBANK::ddi-interactor-in::Compound:Compound', 'Hetionet::CrC::Compound:Compound']:
            # 存储有SMILES的DDI信息
            if src in approved_drug_list and dst in approved_drug_list:
                ddi_pair_single = "{}{}{}\n".format(src, '\t', dst)
                ddi_pair_single_reverse = "{}{}{}\n".format(dst, '\t', src)
                # print(ddi_pair_single)
                
                if ddi_pair_single not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single)
                    ddi_name.append(ddi_pair_single)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id = "{}{}{}\n".format(drug_id_1, delimiter, drug_id_2)
                    ddi.append(ddi_id)
                    
                if ddi_pair_single_reverse not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single_reverse)
                    ddi_name.append(ddi_pair_single_reverse)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id_reverse = "{}{}{}\n".format(drug_id_2, delimiter, drug_id_1)
                    ddi.append(ddi_id_reverse)
