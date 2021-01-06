# 組み込みライブラリ
import os
import sys
import pickle
import gzip
import dataclasses
import argparse
import tempfile

# サードパーティライブラリ
import pandas as pd
import numpy
from mlflow import log_metric, log_param, log_artifact, set_experiment, set_tag
from omegaconf import DictConfig, OmegaConf
import hydra.experimental

# tensorflow ライブラリ
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import tensorflow as tf
# from tensorflow.keras import models, layers
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# 自作パッケージ
import data
from util.fetch_cls_info import show_inheritance as cls_info 
from util.plot import plot_history_acc_loss as plot
from util.plot import plot_history_acc_loss_2 as plot2
import model.mnist as models
import model.cassava as cassava_models # キャッサバコンペ用のモデル


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

@dataclasses.dataclass
class CassavaDataSet:
    """データのテンプレートクラス"""
    train_generator: DataFrameIterator = None
    valid_generator: DataFrameIterator = None
    # train_generator = None
    # valid_generator = None

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


class Cassava_A(Task):
    def __init__(self, params, catalog):
        super().__init__()
        print(OmegaConf.to_yaml(params))
        print(OmegaConf.to_yaml(catalog))
        
        self.params = params.Cassava_A
        self.catalog = catalog
        self.data = None
        self.labels_len = None
        self.dataset = None

    def fetch_data(self):
        train_csv = self.catalog.cassava.train_csv
        path = os.path.join(DATAPATH, train_csv)
        train_labels = pd.read_csv(path)

        train_img_path = self.catalog.cassava.train_imgs
        train_img_full_path = os.path.join(DATAPATH, train_img_path)
        self.labels_len = len(train_labels)
        return train_labels, train_img_full_path

    def preprocessing_train(self):
        train_labels, train_img_path = self.fetch_data()
        train_labels.label = train_labels.label.astype('str') # change obj -> str
        print(train_labels)

        train_datagen = ImageDataGenerator(validation_split = 0.2,
                                     preprocessing_function = None,
                                     rotation_range = 45,
                                     zoom_range = 0.2,
                                     horizontal_flip = True,
                                     vertical_flip = True,
                                     fill_mode = 'nearest',
                                     shear_range = 0.1,
                                     height_shift_range = 0.1,
                                     width_shift_range = 0.1)
        train_generator = train_datagen.flow_from_dataframe(train_labels,
                         directory = train_img_path,
                         subset = "training",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (self.params.size, self.params.size),
                         batch_size = self.params.batch,
                         class_mode = "sparse")

        validation_datagen = ImageDataGenerator(validation_split = 0.2)

        validation_generator = validation_datagen.flow_from_dataframe(train_labels,
                         directory = train_img_path,
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (self.params.size, self.params.size),
                         batch_size = self.params.batch,
                         class_mode = "sparse")
        
        self.dataset = CassavaDataSet(train_generator, validation_generator)

    def run(self):
        self.preprocessing_train()
        data = self.dataset
        params = self.params
        model = cassava_models.efficient_b0_2(params.size)
        
        input_model_path = os.path.join(DATAPATH, self.catalog.model.efficient_b0_imagenet)
        output_model_path = os.path.join(DATAPATH, self.catalog.model.efficient_b0)
        output_best_model_path = os.path.join(DATAPATH, self.catalog.model.efficient_b0_best)
        
        model.load_weights(input_model_path)
        model.compile(optimizer = Adam(lr = params.lrate),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])

        model_save = ModelCheckpoint(output_best_model_path, 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
        early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 2, min_delta = 0.001, 
                              mode = 'min', verbose = 1)

        history = model.fit(
            data.train_generator,
            steps_per_epoch = (self.labels_len*0.8 / params.batch),
            epochs = params.epoch,
            validation_data = data.valid_generator,
            validation_steps = (self.labels_len*0.2 / params.batch),
            callbacks = [model_save, early_stop, reduce_lr]
        )

        model.save(output_model_path)

        # Mlflowで保存
        EXPERIMENT_NAME = 'CassavaMaksym'
        set_experiment(EXPERIMENT_NAME)
        set_tag("task", TASK)
        log_param("data", self.catalog.cassava.train_imgs)
        log_param("epoch", self.params.epoch)
        log_param("model name", self.params.model_name)

        # print(history.history) # for debug

        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        log_metric("loss", loss)
        log_metric("val_loss", val_loss)

        acc = history.history['acc'][0]
        val_acc = history.history['val_acc'][0]
        log_metric("acc", acc)
        log_metric("val_acc", val_acc)
        plot2(history)

if __name__ == "__main__":
    # Hydraインスタンス生成
    hydra.experimental.initialize(config_path="config", job_name="train_mnist")
    params = hydra.experimental.compose(config_name="parameters") # ハイパーパラメータ
    catalog = hydra.experimental.compose(config_name="catalog") # データ一覧
    
    # 学習再現のための乱数設定
    # models.set_randvalue(42)
    cassava_models.set_randvalue(42)

    # # 文字列からクラスインスタンスを生成
    cls = globals()[TASK]
    cls_info(cls) # タスクの継承関係
    instance = cls(params, catalog)
    instance.run() # 学習実行
