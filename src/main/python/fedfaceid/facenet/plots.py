import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    figure: plt.Figure = plt.figure()
    axes: plt.Axes = figure.add_subplot(1, 1, 1)

    axes.plot(false_positive_rate, true_positive_rate,
              color='red', lw=2, label="ROC Curve (area = {:.2f})".format(roc_auc))
    axes.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    axes.set_xlim(xmin=0, xmax=1)
    axes.set_ylim(ymin=0.0, ymax=1.05)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Receiver Operating Characteristic (ROC) Curve')
    axes.legend(loc="lower right")
    figure.savefig(figure_name, dpi=figure.dpi)


def plot_accuracy_lfw(log_dir, epochs, figure_name="lfw_accuracies.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_dir (str): Directory of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]

        fig = plt.figure()
        plt.plot(epoch_list, accuracy_list, color='red', label='LFW Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([1, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('LFW Accuracy')
        plt.title('LFW Accuracies plot')
        plt.legend(loc='lower right')
        fig.savefig(figure_name, dpi=fig.dpi)


def plot_triplet_losses(log_dir, epochs, figure_name="triplet_losses.png"):
    """PLots the Triplet loss over the training epochs.

    Args:
        log_dir (str): Directory of the training log file containing the loss values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting Triplet losses plot.
    """
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        triplet_loss_list = [float(round(float(line.split('\t')[1]), 2)) for line in lines]

        fig = plt.figure()
        plt.plot(epoch_list, triplet_loss_list, color='red', label='Triplet loss')
        plt.ylim([0.0, max(triplet_loss_list)])
        plt.xlim([1, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Triplet losses plot')
        plt.legend(loc='upper left')
        fig.savefig(figure_name, dpi=fig.dpi)
