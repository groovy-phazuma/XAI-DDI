import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )

from model.route2.layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, kg_emb_dim, cross_attn=True):
        super().__init__()
        self.in_features = in_features # n_atom_feats
        self.hidd_dim = hidd_dim # n_atom_hid
        self.rel_total = rel_total # rel_total (number of interaction types)
        self.kge_dim = kge_dim # kge_dim (dimension of interaction matrix)
        self.n_blocks = len(blocks_params) # [2, 2, 2, 2]
        self.n_layers = len(heads_out_feat_params)
        
        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        # binary classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(kg_emb_dim, kg_emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(kg_emb_dim // 2, kg_emb_dim // 4),
            nn.ReLU(),
            nn.Linear(kg_emb_dim // 4, 1)
        )

        self.cross_attn = cross_attn
        proj_input_dim = 4 * self.hidd_dim * self.n_layers
        self.pair_proj = nn.Linear(proj_input_dim, self.kge_dim)
        self.query_proj = nn.Linear(proj_input_dim, kg_emb_dim)
        self.scale = math.sqrt(kg_emb_dim)

        self.query_norm = nn.LayerNorm(kg_emb_dim)
        self.prot_norm = nn.LayerNorm(kg_emb_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, triples, kg_features):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2] # emb
            r_t = out[3] # emb
            repr_h.append(r_h)
            repr_t.append(r_t)
        
            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        flat_h = repr_h.reshape(repr_h.size(0), -1)  # (B, n_layers*hidd_dim)
        flat_t = repr_t.reshape(repr_t.size(0), -1)  # (B, n_layers*hidd_dim)

        # pair features (concat + hadamard + diff)
        pair_feat = torch.cat([
            flat_h, flat_t,
            flat_h * flat_t,
            torch.abs(flat_h - flat_t)
        ], dim=-1)  # (B, 4 * n_layers * hidd_dim)


        # cross attention
        if self.cross_attn:
            q = self.query_proj(pair_feat)            # (B, kg_emb_dim)
            q = self.query_norm(q).unsqueeze(1)       # normalize
            k = v = self.prot_norm(kg_features).unsqueeze(0).repeat(q.size(0), 1, 1)

            attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            cross_attn_out = torch.bmm(attn_weights, v).squeeze(1)
            cross_attn_out = self.dropout(cross_attn_out)  # dropout to break symmetry
        else:
            cross_attn_out = kg_features.mean(dim=0, keepdim=True).expand(pair_feat.size(0), -1)
            attn_weights = None

        scores = self.classifier(cross_attn_out)

        return scores, attn_weights


#intra+inter
class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, h_data,t_data,b_graph):
     
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)
   
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_interRep,t_interRep = self.interAtt(h_data,t_data,b_graph)
        
        h_rep = torch.cat([h_intraRep,h_interRep],1)
        t_rep = torch.cat([t_intraRep,t_interRep],1)
        h_data.x = h_rep
        t_data.x = t_rep

        
        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)
      
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        

        return h_data,t_data, h_global_graph_emb,t_global_graph_emb



