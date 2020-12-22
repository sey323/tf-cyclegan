# tf-gan
tensorflowでcycle-gan

## 初期設定
依存ライブラリを追加する。

```
$ pip install -r requirements.txt
```

## 学習
CylceGanによる学習は以下のコマンドで実行する。学習結果、各Epochで生成された画像は`results/${実行時間のタイムスタンプ}/images`以下に保存される。

```
$ python train.py --folderA=${学習するSource画像のパス} \
                  --folderB=${学習するTarget画像のパス} \
                  --folderA_test=${検証用Source画像のパス} \
                  --folderB_test=${検証用Target画像のパス} \
                  --max_epoch=10 \ #学習回数
                  --learn_rate=0.0002 \ #学習率
                  --cycle_loss_penalty=30 \ #cycle lossの重み
                  --identity_loss_penalty=30 #identity lossの重み
```

### オプション
`train.py`で選択可能なオプションの一覧

| オプション名 | パラメータ | デフォルト値 | 
| - | - | - |
| folderA | 学習するSource画像のパス | "" | 
| folderB | 学習するTarget画像のパス | "" | 
| folderA_test | 検証用Source画像のパス | "" |
| folderB_test | 検証用Target画像のパス | "" |
| resize | モデルに入力する画像の1辺のサイズ(画像は`(resize, resize)`のサイズに調整される) | 64 |
| gray | 濃淡画像に変換するかどうか(True or False)| False |
| batch_size | ミニバッチサイズ | 64 | 
| learn_rate | 学習率 | 0.002 | 
| max_epoch | 学習Epoch数 | 100 | 
| save_folder | 学習結果を保存するパス | "" |
| drop_prob | ドロップアウトの確率 | 0.5 |
| cycle_loss_penalty | cycle lossの重み | 10 |
| identity_loss_penalty | identity lossの重み | 10 |


## 評価
### 1. 学習済みcheckpointの利用
上記の学習を実行すると`results/${実行時間のタイムスタンプ}`ディレクトリに学習済み重みを記録した`checkpoint`ファイルが生成される。その`checkpoint`ファイルから学習した重みを復元し、モデルの評価を行う時に用いる。生成された画像は`results/${実行時間のタイムスタンプ}`ディレクトリに保存される。

```
$ python eval.py --model_path=${checkpointファイルのパス} \
                 --folderA=${テスト用Source画像のパス} \
                 --folderB=${テスト用Target画像のパス}
```

またcheckpointファイルを永続保存形式であるSavedModel形式に変換するときは、以下のコマンドを用いる。実行結果は`results/${実行時間のタイムスタンプ}`ディレクトリに保存される。

```
$ python eval.py --model_path=${checkpointファイルのパス} \
                 --mode=freeze
```

### 2. SavedModel形式のファイルの実行
SavedModel形式の学習済み重みを用いて画像を生成するときは、以下のコマンドを用いる。生成された画像は`results/fake/${実行時間のタイムスタンプ}`ディレクトリに保存される。

```
$ python eval_saved_model.py --folder=${生成したい画像の入力となる画像のフォルダ} \
                             --model_path=${SavedModel形式の学習済み重み}
```

### 3. 生成された画像の評価
生成された画像をsignateの提出形式に変換する場合は、以下のコマンドを用いる。実行結果はプログラムを実行したディレクトリに保存される。

```
$ python make_submit.py -p ${評価したい画像が保存されているフォルダ}
```

##### 参考
https://signate.jp/competitions/285/data
