# Route1
化合物ペア特徴量とKG由来のノード特徴量とのcross-attention modelを構築する

### workflow
1. DSN-DDI由来の両化合物特徴量を取得 (heads特徴量とtails特徴量)
2. MyKGをTransEなどで埋め込み、生体内分子のembeddingsを取得
3. headsとtailsをconcatしたpair特徴量に対してcross-attention
4. 元のheadsとtailsにconcatして、co-attention mapを用いたスコアリング