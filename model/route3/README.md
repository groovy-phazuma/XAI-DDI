# Route2
化合物ペア特徴量とKG由来のノード特徴量とのcross-attention modelを構築する

### workflow
1. DSN-DDI由来の両化合物特徴量を取得 (heads特徴量とtails特徴量)
2. MyKGをTransEなどで埋め込み、生体内分子のembeddingsを取得
3. ペア特徴量を定義
4. シンプルにペア特徴量に2をcross-attentionして、得られた特徴量から2値分類を行う