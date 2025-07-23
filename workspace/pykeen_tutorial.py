#!/usr/bin/env python3
"""
Created on 2025-07-23 (Wed) 22:30:43

Tutorial for PyKEEN
This script is a basic setup to explore the PyKEEN library for knowledge graph embeddings.

@author: I.Azuma
"""

# %%
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from pykeen.trackers import CSVResultTracker

# %%
# Train a knowledge graph embedding model using PyKEEN
result = pipeline(
    model='TransE',  # or TransE_l2
    dataset='nations',  # or specify own triples
    training_kwargs=dict(
        batch_size=2048,            # DGL-KE: --batch_size
        num_epochs=10000
    ),
    model_kwargs=dict(
        # DGL-KE: --hidden_dim
        # For TransE, DistMult, ComplEx etc., PyKEEN uses `embedding_dim`
        embedding_dim=200,
        # gamma (specific to TransE family)
        # DGL-KE: --gamma
        scoring_fct_norm=2,         # DGL-KE: TransE_l1 uses 1, TransE_l2 uses 2
    ),
    loss='MarginRankingLoss',  # Loss consistent with DGL-KE (common for TransE)
    optimizer='Adagrad',  # Optimizer used in DGL-KE
    optimizer_kwargs=dict(
        lr=0.01,  # DGL-KE: --lr
    ),
    negative_sampler='basic',  # Uniform negative sampling same as DGL-KE
    negative_sampler_kwargs=dict(
        num_negs_per_pos=128,  # DGL-KE: --neg_sample_size
    ),
    # Enable validation with stopper (explicit requirement in PyKEEN unlike DGL-KE)
    stopper='early',
    stopper_kwargs=dict(
        frequency=5,
        patience=10,
        metric='mean_reciprocal_rank',
    ),
)

# %%
# Visualize entity embeddings

# Extract entity embeddings (vectors)
entity_embeddings = result.model.entity_representations[0]().cpu().detach().numpy()
entity_labels = result.training.entity_id_to_label

# Dimensionality reduction (PCA to 2D)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(entity_embeddings)

# Visualization
plt.figure(figsize=(8, 6))
for i, label in entity_labels.items():
    x, y = emb_2d[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, label, fontsize=8)
plt.title('Entity Embeddings (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot training loss
losses = result.losses

# Create plot
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

