## ビルド方法
一番上の階層でsetup.pyを用いてビルドする。

```
$ pip install -e .
```

ビルド後にpipのリストを確認し、shackdooというライブラリがあるか確認する。

```
$ pip list
```


requirements.txtで記載されているライブラリは以下のようになっている。
train.pyで実行されてるmnistの学習モデルは、tensorflow-gpuの2.0以上で動作しているが参考程度にしてほしい。


```
hydra-core
mlflow
numpy
matplotlib
```

## 動作と結果の確認
すぐ動かせるようにMnistのデータをpkl形式で保存してある。
ビルド後に下記コマンドを入力し、学習を開始する。2分ほどで学習が完了すると思う。
引数はクラス名を表している。

```
$ python train.py --task Mnist_B
```

学習後の結果の確認では、mlflowでローカルサーバーを以下のコマンドで立ち上げる。

```
$ mlflow ui
```

Dockerの環境の場合は以下のコマンドで立ち上げる。

```
$ mlflow ui --port 5000 --host 0.0.0.0
```

Dockerの場合、コンテナ作成の時点でポートフォワードしなければならないので注意。

```
$ docker run --rm -it -v "$PWD":/tf -p 8080:5000 [コンテナ名] /bin/bash
```

ブラウザで`http://localhost:5000`を入力して、結果を確認できる。(Dockerの上の場合、ポートフォワードされるので`http://localhost:8080`を入力する。)


## 考え方
### 概略
OSSのhydraとmlflowを用いて、DL・MLのタスクを再現性のあるものにすることを目的にしている。
各実験は「実験名」と「タグ」で管理し、mlflowで一覧でパラメータや結果を確認できるようにすることを目的にしている。

```
├── 実験名
│   ├── タグ1
│   └── タグl
│       ├── パラメータ
│       └── 結果
```

### ハイパーパラメータ
学習時のパラメータやデータセットはyamlファイルで管理されている。
学習データを管理するcatalog.yamlと学習時のパラメータを管理するparameters.yamlが存在し、学習時に呼び出し、mlflowに結果と共に保存する。

```
├── config
│   ├── catalog.yaml
│   └── parameters.yaml
```

### 学習時のフローの継承
学習は分類すると、データ取得(fetch_data)・データ前処理(preprocessing_train)・学習(run)の3つのフローに分けられると考え、学習の状態をTaskというクラスとして、フローを継承する。

train.pyではTaskというデータ取得(fetch_data)・データ前処理(preprocessing_train)・学習(run)の3つの関数を持った抽象クラスを作成し、Mnist_Aというクラスに継承している。

Mnist_AからMnist_Bにも継承しており、実務でハイパーパラメータが異なり実験を行った想定でparameters.yamlファイルを読み込ませており、コンストラクタでのハイパーパラメータのみが異なっている。


## 構成
構成のイメージは下の通りである。

```
├── config
│   ├── catalog.yaml <-データ種類とパスが記載されたカタログ
│   |── parameters.yaml <-学習のパラメータ
│   └── optimizer.yaml <-最適化の値
├── data
│   ├── raw <-生のデータ
│   ├── feature <-クロップなどの処理後のデータ
│   ├── intermediate <-処理の中間データ
│   ├── model <-学習後の重みデータ
│   └── report <-図やレポートなど保存
├── feature
│   └── feature.py <-前処理を行う
├── model
│   └── model.py <-学習時のモデルファイル
├── notebook <-ノートブック
├── reference <-リファレンス
├── util
│   └── util.py <-図のプロットなどの関数
├── train.py <-学習時のファイル
├── inference.py <-推論時のファイル(本当はフォルダにした方がいいかも)
```

## まとめ
今回、以下の3つの提案と効果があった。
- ①パッケージをライブラリで管理することにより、相対パスの悩みから解放
- ②hydraとmlflowを使用することにより、学習の再現性がとれ(notebookにべた書きでもok)、実行時引数の多さに辟易しなくてよい
- ③学習時のフローをタスクという粒度に分け、継承することで理解しやすくなり、再現も楽になった