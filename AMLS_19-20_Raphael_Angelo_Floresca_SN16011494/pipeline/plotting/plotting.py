import numpy as np
import matplotlib.pyplot as plt

# Plot the training loss and accuracy
def plot_train_loss_acc_lr(history, epochs, schedule, tla_plot_path, lr_plot_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy on A1')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss A2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(tla_plot_path)
 
    if schedule is not None:
	    schedule.plot(epochs_range)
	    plt.savefig(lr_plot_path)


