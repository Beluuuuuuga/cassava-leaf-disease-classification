# 組み込みライブラリ
import os
import sys
import pickle
import gzip
import dataclasses
import argparse
import tempfile

# サードパーティライブラリ 
import numpy
from mlflow import log_metric, log_param, log_artifact, set_experiment, set_tag
from omegaconf import DictConfig, OmegaConf
import hydra.experimental

# 自作パッケージ
import data
from util.fetch_cls_info import show_inheritance as cls_info 
from util.mlflow import Mlflow 
from util.plot import plot_history_acc_loss as plot
import model.mnist as models


DATAPATH = data.__path__[0] # パッケージのパスとして取得

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Mnist_A', help='Task(Class) Name')
args = parser.parse_args()
TASK = args.task


class Task:
    """各タスクのテンプレートクラス"""
    def __init__(self):
        pass
    
    def fetch_data(self):
        """学習データ取得: タスクによってパスだけやデータそのものなど分ける"""
        pass

    def preprocessing_train(self):
        """学習データ前処理: モデルにデータを流す直前まで行う"""
        pass

    def run(self):
        """学習のメイン"""
        pass


@dataclasses.dataclass # dataclassは3.7以上が推奨です。例なので適当にお使いください。
class DataSet:
    """データのテンプレートクラス"""
    train_data: numpy.ndarray = None
    train_label: numpy.ndarray = None
    test_data: numpy.ndarray = None
    test_label: numpy.ndarray = None


class Mnist_A(Task):
    def __init__(self, params, catalog):
        super().__init__()
        print(OmegaConf.to_yaml(params))
        print(OmegaConf.to_yaml(catalog))
        
        self.params = params.Mnist_A
        self.catalog = catalog
        self.data = None
        self.dataset = None

    def fetch_data(self):
        mnist_data = self.catalog.mnist.raw_pkl # raw/mnist.pkl.gz
        path = os.path.join(DATAPATH, mnist_data) # data/raw/mnist.pkl.gz
        f = gzip.open(path, 'rb')
        self.data = pickle.load(f, encoding='bytes')
        f.close()

    def preprocessing_train(self):
        (train_images, train_labels), (test_images, test_labels) = self.data
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))

        # ピクセルの値を 0~1 の間に正規化
        train_images_regularized, test_images_regularized = train_images / 255.0, test_images / 255.0
        self.dataset = DataSet(train_images_regularized, train_labels, test_images_regularized, test_labels)

    def run(self):
        self.fetch_data()
        self.preprocessing_train()
        data = self.dataset
        model = models.base_model()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        history = model.fit(data.train_data, data.train_label, epochs=self.params.epoch) # epoch 1
        test_loss, test_acc = model.evaluate(data.test_data, data.test_label, verbose=1)
        fname = TASK + ".hdf5"
        model.save(os.path.join(self.catalog.output.model, fname))

        # Mlflowで保存
        EXPERIMENT_NAME = 'mnist_exp'
        set_experiment(EXPERIMENT_NAME)
        set_tag("task", TASK)
        log_param("data", self.catalog.mnist.raw_pkl)
        log_param("epoch", self.params.epoch)
        log_param("model name", self.params.model_name)
        log_metric("test loss", test_loss)
        log_metric("test acc", test_acc)
        plot(history)


# Mnist_Aタスクから継承: 学習データ・前処理は同じで、エポック数だけ1から5に変更したいとする
class Mnist_B(Mnist_A):
    def __init__(self, params, catalog):
        super().__init__(params, catalog)
        self.params = params.Mnist_B # epoch 2


if __name__ == "__main__":
    # Hydraインスタンス生成
    hydra.experimental.initialize(config_path="config", job_name="train_mnist")
    params = hydra.experimental.compose(config_name="parameters") # ハイパーパラメータ
    catalog = hydra.experimental.compose(config_name="catalog") # データ一覧
    
    # 学習再現のための乱数設定
    models.set_randvalue(42)

    # 文字列からクラスインスタンスを生成
    cls = globals()[TASK]
    cls_info(cls) # タスクの継承関係
    instance = cls(params, catalog)
    instance.run() # 学習実行