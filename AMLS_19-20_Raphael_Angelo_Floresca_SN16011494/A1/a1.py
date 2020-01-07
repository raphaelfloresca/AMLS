from pipeline.datasets.celeba_gender import create_celeba_df
from pipeline.datasets.utilities import get_X_y_test_sets, go_up_three_dirs, create_datagens
from pipeline.models.mlp import train_mlp, train_cnn
from pipeline.plotting.plotting import plot_train_loss_acc_lr, plot_top_losses
import os

class A1:
    height = 218
    width = 178
    num_classes = 2
    batch_size = 32
    random_state = 42

    train_gen, val_gen, test_gen = create_datagens(
        height,
        width,
        create_celeba_df(),
        "img_name",
        "gender",
        batch_size,
        random_state,
        None)

class A1_MLP(A1):
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule,
            random_state,
            first_af="relu",
            second_af="relu",
            layer1_hn=300,
            layer2_hn=100):

        # Change random state according to constructor
        self.random_state = random_state
        A1.random_state = self.random_state

        self.epochs = epochs

        self.train_gen, self.val_gen, self.test_gen = A1.train_gen, A1.val_gen, A1.test_gen

        self.model, self.history, self.schedule = train_mlp(
            A1.height, 
            A1.width,
            A1.num_classes,
            A1.batch_size,
            self.epochs,
            learning_rate,
            schedule,
            self.train_gen,
            self.val_gen,
            first_af,
            second_af,
            layer1_hn,
            layer2_hn)

    def train(self):
        # Navigate to output folder in parent directory
        go_up_three_dirs()        

        # Plot training loss accuracy and learning rate change
        plot_train_loss_acc_lr(
            self.history,
            self.epochs,
            self.schedule,
            "A1",
            "output/train_loss_acc_A1_mlp.png",
            "output/lr_A1_mlp.png")

        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Go back to image folder
        os.chdir("data/dataset_AMLS_19-20/celeba")

        # Split ImageDataGenerator object for the test set into separate X and y test sets
        gender_X_test, gender_y_test = get_X_y_test_sets(self.test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, gender_X_test, gender_y_test, "output/plot_top_losses_A1_mlp.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(gender_X_test, gender_y_test)[-1]
        return test_accuracy

class A1_CNN(A1):
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule,
            random_state,
            num_start_filters=16,
            kernel_size=3,
            fcl_size=512):

        # Change random state according to constructor
        self.random_state = random_state
        A1.random_state = self.random_state

        self.epochs = epochs

        self.train_gen, self.val_gen, self.test_gen = A1.train_gen, A1.val_gen, A1.test_gen

        self.model, self.history, self.schedule = train_cnn(
            A1.height, 
            A1.width,
            A1.num_classes,
            A1.batch_size,
            self.epochs,
            learning_rate,
            schedule,
            self.train_gen,
            self.val_gen,
            num_start_filters,
            kernel_size,
            fcl_size)

    def train(self):
        # Navigate to output folder in parent directory
        go_up_three_dirs()        

        # Plot training loss accuracy and learning rate change
        plot_train_loss_acc_lr(
            self.history,
            self.epochs,
            self.schedule,
            "A1",
            "output/train_loss_acc_A1_cnn.png",
            "output/lr_A1_cnn.png")

        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Go back to image folder
        os.chdir("data/dataset_AMLS_19-20/celeba")

        # Split ImageDataGenerator object for the test set into separate X and y test sets
        gender_X_test, gender_y_test = get_X_y_test_sets(self.test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, gender_X_test, gender_y_test, "output/plot_top_losses_A1_cnn.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(gender_X_test, gender_y_test)[-1]
        return test_accuracy