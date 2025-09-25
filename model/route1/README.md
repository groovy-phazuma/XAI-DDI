# Route1
化合物ペア特徴量とKG由来のノード特徴量とのcross-attention modelを構築する

### workflow
1. DSN-DDI由来のペア特徴量の取得
2. MyKGをTransEなどで埋め込み、生体内分子のembeddingsを取得
3. DDI予測の過程における量特徴量のcross-attentionをモデリング
