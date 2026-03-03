import torch
from torch import nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature
    
    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights= F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        
        return (p_loss + n_loss) / 2, p_loss, n_loss 

class MultiClassSigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature
    
    def forward(self, p_logits, p_targets, n_logits):
        """
        p_logits:  [batch_size, 6]
        p_targets: [batch_size]
        n_logits:  [batch_size, 6] (ネガティブサンプルが1つずつの場合)
        """
        batch_size = p_logits.size(0)
        
        # --- ① Link Loss ---
        # ポジティブ側の正解クラスのスコアを抽出
        p_scores = p_logits.gather(1, p_targets.view(-1, 1)).squeeze(-1)
        
        # ネガティブ側からも、同じクラス（Relation）のスコアを抽出
        # n_logitsが [batch_size, 6] なので、dim=1 で gather します
        n_scores = n_logits.gather(1, p_targets.view(-1, 1)).squeeze(-1)
        
        # Adversarial weighting (n_scoresが1次元[batch_size]の場合は、
        # 通常のSigmoid Lossとして計算されます)
        if self.adv_temperature and n_scores.dim() > 1:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = (weights * n_scores).sum(dim=-1)
            
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        link_loss = (p_loss + n_loss) / 2
        
        # --- ② Class Loss ---
        # ポジティブサンプルがどのDDIタイプ（6クラス）かを学習
        class_loss = F.cross_entropy(p_logits, p_targets)
        
        total_loss = link_loss + class_loss
        
        return total_loss, p_loss, n_loss