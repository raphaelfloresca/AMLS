import numpy as np
import matplotlib.pyplot as plt
from pipeline.plotting.gradcamutils import grad_cam, grad_cam_plus

# Plot the training loss and accuracy
def plot_train_loss_acc_lr(H, epochs, schedule, task_name, tla_plot_path, lr_plot_path):
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on " + task_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(tla_plot_path)
 
    if schedule is not None:
	    schedule.plot(N)
	    plt.savefig(lr_plot_path)

# Get indices of wrongfully misclassified test set
# https://stackoverflow.com/questions/39300880/how-to-find-wrong-prediction-cases-in-test-set-cnns-using-keras
def get_wrong_indices(model, X_test, y_test):
    incorrects = np.asarray(np.nonzero(model.predict(X_test).argmax(axis=-1).reshape((-1,)) != y_test))
    incorrects = incorrects.T.flatten()
    return incorrects

# This returns an array which contains the predictions for the misclassified images
def get_incorrect_preds(model, X_test, y_test, incorrects):
    incorrect_preds = []

    for incorrect in np.nditer(incorrects):
        probs_pred = model.predict(X_test[incorrect:incorrect+1])
        incorrect_pred = probs_pred.argmax(axis=-1)
        incorrect_preds.append(incorrect_pred)

    incorrect_preds = np.asarray(incorrect_preds)
    incorrect_preds = incorrect_preds.T.flatten()
    incorrect_preds = incorrect_preds.astype(float)
    return incorrect_preds

# This returns an array which contains the actual labels for the misclassified images
def get_actual_labels(X_test, y_test, incorrects):
    actual_labels = []

    for incorrect in np.nditer(incorrects):
        actual_label = y_test[incorrect]
        actual_labels.append(actual_label)

    actual_labels = np.asarray(actual_labels)
    actual_labels = actual_labels.T.flatten()
    actual_labels = actual_labels.astype(float)
    return actual_labels

# This returns an array which contains the individual losses for the misclassified images
def get_incorrect_losses(model, X_test, y_test, incorrects):
    incorrect_losses = [] 

    for incorrect in np.nditer(incorrects):
        loss = model.evaluate(X_test[incorrect:incorrect+1], y_test[incorrect:incorrect+1], verbose=0)
        incorrect_losses.append(loss[0])

    incorrect_losses = np.asarray(incorrect_losses)
    incorrect_losses = incorrect_losses.astype(float)
    return incorrect_losses

def get_probs_correct_label(model, X_test, y_test, incorrects):
# This returns an array which contains the probabilities of the actual label
# for the misclassified images
    probs_correct_label = []

    for incorrect in np.nditer(incorrects):
        prob_correct_label = model.predict(X_test[incorrect:incorrect+1])
        probs_correct_label.append(prob_correct_label[0,int(y_test[incorrect])])

    probs_correct_label = np.asarray(probs_correct_label)
    probs_correct_label = probs_correct_label.astype(float)
    return probs_correct_label

def create_loss_pred_data(model, X_test, y_test):
    incorrects = get_wrong_indices(model, X_test, y_test)
    incorrect_preds = get_incorrect_preds(model, X_test, y_test, incorrects)
    actual_labels = get_actual_labels(X_test, y_test, incorrects)
    incorrect_losses = get_incorrect_losses(model, X_test, y_test, incorrects)
    probs_correct_label = get_probs_correct_label(model, X_test, y_test, incorrects)

    # This joins together the indices of incorrectly misclassified images, their losses and
    # actual label probabilities into a numpy array. It is then sorted in descending order
    loss_pred_data = np.column_stack((incorrects.astype(float), 
                                      incorrect_preds, 
                                      actual_labels, 
                                      incorrect_losses, 
                                      probs_correct_label))
    loss_pred_data = loss_pred_data[np.argsort(loss_pred_data[:,3])[::-1]]
    return loss_pred_data



# This plots a single misclassified image alongside its predicted label,
# its actual label, the error rate for the image and the probability given
# to the actual label
def plot_wrong_image(i, loss_pred_data, img_data):
    plt.imshow(img_data[int(loss_pred_data[i,0])])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("{}/{} {:0.2f} {:0.2f}".format(int(loss_pred_data[i,1]),
                                              int(loss_pred_data[i,2]),
                                              loss_pred_data[i,3],
                                              loss_pred_data[i,4]))

# This plots the images which have been the most misclassified when running the
# model on the test set. Inspired by the plot_top_losses function in the fastai
# library
def plot_top_losses(model, X_test, y_test, ptl_plot_path):
    loss_pred_data = create_loss_pred_data(model, X_test, y_test)
    num_images = 9
    plt.figure(figsize=(6, 6))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plot_wrong_image(i, loss_pred_data, X_test)
    plt.tight_layout()
    plt.savefig(ptl_plot_path)

def get_correct_indices(model, X_test, y_test):
    corrects = np.asarray(np.nonzero(model.predict(X_test).argmax(axis=-1).reshape((-1,)) == y_test))
    corrects = corrects.T.flatten()
    return corrects

def get_correct_preds(model, X_test, y_test, corrects):
    correct_preds = []

    for correct in np.nditer(corrects):
        probs_pred = model.predict(X_test[correct:correct+1])
        correct_pred = probs_pred.argmax(axis=-1)
        correct_preds.append(correct_pred)

    correct_preds = np.asarray(correct_preds)
    correct_preds = correct_preds.T.flatten()
    correct_preds = correct_preds.astype(float)
    return correct_preds

def get_correct_accuracies(model, X_test, y_test, corrects):
    correct_accuracies = []

    for correct in np.nditer(corrects):
        accuracy = model.evaluate(X_test[correct:correct+1], y_test[correct:correct+1], verbose=0)
        correct_accuracies.append(accuracy[1])
    
    correct_accuracies = np.asarray(correct_accuracies)
    correct_accuracies = correct_accuracies.astype(float)
    return correct_accuracies

def create_top_n_data(model, X_test, y_test, top_n):
    corrects = get_correct_indices(model, X_test, y_test)
    correct_preds = get_correct_preds(model, X_test, y_test, corrects)
    correct_accuracies = get_correct_accuracies(model, X_test, y_test, corrects)

    top_n_data = np.column_stack((corrects.astype(float),
                                  correct_preds,
                                  correct_accuracies))
    top_n_data = top_n_data[np.argsort(top_n_data[:,2])[::-1]]
    top_n_data = top_n_data[:top_n,:]
    return top_n_data

def plot_grad_cam(model, X_test, y_test, top_n, layer_name, grad_cam_plot_path):
    top_n_data = create_top_n_data(model, X_test, y_test, top_n)

    plt.subplots(nrows=top_n, ncols=3)

    for i in range(top_n):
        img = X_test[int(top_n_data[i,0])] 
        img = np.expand_dims(img, axis=0)

        gradcam = grad_cam(model, img, layer_name=layer_name)
        gradcamplus = grad_cam_plus(model, img, layer_name=layer_name)

        index = i*3
        
        plt.subplot(top_n,3,index+1)
        plt.imshow(X_test[int(top_n_data[i,0])])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("{} {:0.2f}".format(int(top_n_data[i,1]),
                                           top_n_data[i,2]))
        plt.title("input image")

        plt.subplot(top_n,3,index+2)
        plt.imshow(X_test[int(top_n_data[i,0])])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gradcam,alpha=0.8,cmap="jet")
        plt.title("Grad-CAM")

        plt.subplot(top_n,3,index+3)
        plt.imshow(X_test[int(top_n_data[i,0])])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gradcamplus,alpha=0.8,cmap="jet")
        plt.title("Grad-CAM++")

    plt.tight_layout()
    plt.savefig(grad_cam_plot_path)
