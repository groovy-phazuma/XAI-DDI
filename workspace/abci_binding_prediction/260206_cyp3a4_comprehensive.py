# -*- coding: utf-8 -*-
"""
Created on 2026-02-06 (Fri) 13:53:21

CYP familyに対して、drugbankに登録されている化合物との結合親和性を回収するスクリプト

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
import os, subprocess, yaml, json
from pathlib import Path
from tqdm import tqdm

# --- 設定 ---
BASE_DIR = "/home/aah18044co/github/XAI-DDI"
WORK_DIR = f"{BASE_DIR}/workspace/abci_binding_prediction"
INPUT_YAML_DIR = f"{WORK_DIR}/results/260206_cyp3a4/inputs"
OUTPUT_DIR = f"{WORK_DIR}/results/260206_cyp3a4/outputs"


def create_yaml(hgnc_symbol, protein_seq, drug_id, smiles, output_dir):
    """
    IDを単純な固定文字列に変更し、
    ファイル名に情報を込めることで型エラーとパースエラーを回避する
    """
    # ファイル名には具体的な名前を使用
    job_name = f"{hgnc_symbol}_{drug_id}"
    
    data_config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",  # Boltz内部エラー回避のため固定文字
                    "sequence": protein_seq
                }
            },
            {
                "ligand": {
                    "id": "B",  # Boltz内部エラー回避のため固定文字
                    "smiles": smiles
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "B"  # ligandのidに対応させる
                }
            }
        ]
    }
    
    file_path = os.path.join(output_dir, f"{job_name}.yaml")
    with open(file_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    
    return job_name

def run_comprehensive_predict(fasta_map, drug_map):
    os.makedirs(INPUT_YAML_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 全組み合わせのYAML生成
    job_list = []
    print(f"Generating YAMLs for {len(fasta_map)} proteins x {len(drug_map)} drugs...")
    for hgnc, seq in tqdm(fasta_map.items()):
        for d_id, smi in drug_map.items():
            # すでに存在する場合はスキップ
            job_name = f"{hgnc}_{d_id}"
            if Path(f"{INPUT_YAML_DIR}/{job_name}.yaml").exists():
                job_list.append({"job_id": job_name, "hgnc": hgnc, "drug_id": d_id})
                continue
            # 配列が空でないか、SMILESが有効かチェック
            if not seq or len(seq) < 10 or pd.isna(smi): 
                continue
            
            job_name = create_yaml(hgnc, seq, d_id, smi, INPUT_YAML_DIR)
            job_list.append({"job_id": job_name, "hgnc": hgnc, "drug_id": d_id})

    # マッピング表の保存
    pd.DataFrame(job_list).to_csv(f"{OUTPUT_DIR}/job_mapping.csv", index=False)

    # 2. Boltz実行
    cmd = [
        "boltz", "predict", INPUT_YAML_DIR, 
        "--out_dir", OUTPUT_DIR, 
        "--use_msa_server", 
        "--accelerator", "gpu", 
        "--devices", "1",
        "--num_workers", "1" 
    ]
    
    print("\n🚀 Starting Boltz Prediction...")
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Prediction successfully completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Boltz failed with error: {e}")

if __name__ == "__main__":
    # データ読み込み
    print("Loading data...")
    fasta_dict = pd.read_pickle(f'{BASE_DIR}/dataset/target_proteins/cyp_fasta_dict_29.pkl')
    info_df = pd.read_csv(f'{BASE_DIR}/dataset/drugbank/drug_smiles.csv')
    smiles_dict = dict(zip(info_df['drug_id'], info_df['smiles']))

    # --- テスト実行の設定 ---
    target_hgncs = ['CYP3A4']
    test_fasta = {k: fasta_dict[k] for k in target_hgncs}
    
    target_drug_ids = list(smiles_dict.keys())
    test_drugs = {k: smiles_dict[k] for k in target_drug_ids} 
    
    # 実行
    run_comprehensive_predict(test_fasta, test_drugs)
