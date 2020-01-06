import numpy as np
import matplotlib.pyplot as plt

# Plot the training loss and accuracy
def plot_train_loss_acc_lr(H, epochs, schedule, tla_plot_path, lr_plot_path):
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on task A1")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(tla_plot_path)
 
    if schedule is not None:
	    schedule.plot(N)
	    plt.savefig(lr_plot_path)


