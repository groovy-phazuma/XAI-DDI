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

from layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, kg_emb_dim):
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

        proj_input_dim = self.hidd_dim * 2 * self.n_layers
        self.pair_proj = nn.Linear(proj_input_dim, self.kge_dim)
        self.query_proj = nn.Linear(proj_input_dim, kg_emb_dim)
        self.scale = math.sqrt(kg_emb_dim)

        self.head_fusion_layer = nn.Linear(self.hidd_dim * self.n_layers + kg_emb_dim, self.kge_dim)
        self.tail_fusion_layer = nn.Linear(self.hidd_dim * self.n_layers + kg_emb_dim, self.kge_dim)

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
        kge_heads = repr_h
        kge_tails = repr_t

        # 1. generate pair features
        pair_features = torch.cat([kge_heads, kge_tails], dim=-1) # (batch_size, embed_dim * 2)
        aggregated_features = pair_features.flatten(start_dim=1)

        # 2. generate query
        query = self.query_proj(aggregated_features) # (batch_size, kg_feature_dim)

        attention_scores = torch.matmul(query, kg_features.transpose(0, 1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, kg_features)
    
        # update head/tail representations with context
        f_heads = kge_heads.flatten(start_dim=1)
        f_tails = kge_tails.flatten(start_dim=1)

        fused_h = torch.cat([f_heads, context], dim=-1) # (B, 512 + 200)
        enriched_heads = self.head_fusion_layer(fused_h) # (B, n_features)

        fused_t = torch.cat([f_tails, context], dim=-1) # (B, 512 + 200)
        enriched_tails = self.tail_fusion_layer(fused_t) # (B, n_features)
        
        #print(('kge_heads:', kge_heads.size()))
        #print('fused_h:', fused_h.size())
        #print('enriched_heads:', enriched_heads.size())
        
        # scores
        attentions = self.co_attention(enriched_heads, enriched_tails)
        #print('attentions:', attentions.size())
        scores = self.KGE(enriched_heads, enriched_tails, rels, attentions)

        return scores, attention_weights


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



