# tf-gan
tensorflowでcycle-gan

## 初期設定
依存ライブラリを追加する。

```
$ pip install -r requirements.txt
```

## 学習
CylceGanによる学習は以下のコマンドで実行する。

```
$ python train.py --folderA=${学習するSource画像のパス} \
                  --folderB=${学習するTarget画像のパス} \
                  --folderA_test=${テスト用Source画像のパス} \
                  --folderB_test=${テスト用Target画像のパス} \
                  --max_epoch=10 \ #学習回数
                  --learn_rate=0.0002 \ #学習率
                  --cycle_loss_penalty=30 \ #cycle lossの重み
                  --identity_loss_penalty=30 #identity lossの重み
```

### オプション
`train.py`で選択可能なオプションの一覧

| オプション名 | パラメータ | デフォルト値 | 有効なモデル |
| - | - | - | - |
| folderA | 学習するSource画像のパス | "" |  - |
| folderB | 学習するTarget画像のパス | "" | - |
| folderA_test | テスト用Source画像のパス | "" | - |
| folderB_test | テスト用Target画像のパス | "" | - |
| resize | モデルに入力する画像の1辺のサイズ(画像は`(resize, resize)`のサイズに調整される) | 64 | - |
| gray | 濃淡画像に変換するかどうか(True or False)| False | - |
| batch_size | ミニバッチサイズ | 64 | - | 
| learn_rate | 学習率 | 0.002 | - | 
| max_epoch | 学習Epoch数 | 100 | - | 
| save_folder | 学習結果を保存するパス | "" | - |
| drop_prob | ドロップアウトの確率 | 0.5 | - |
| cycle_loss_penalty | cycle lossの重み | 10 | - |
| identity_loss_penalty | identity lossの重み | 10 | - |

