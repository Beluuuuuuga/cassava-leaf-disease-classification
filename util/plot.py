import tempfile

import matplotlib.pyplot as plt
from mlflow import log_artifact


def plot_history_acc_loss(fit):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    # axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

    # Plot the loss in the history
    axR.plot(fit.history['accuracy'],label="accuracy for training")
    # axR.plot(fit.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

    # 仮ディレクトリに保存
    temp = tempfile.TemporaryDirectory()
    write_path = temp.name + '/training.png'
    fig.savefig(write_path)
    log_artifact(write_path)
    plt.close()