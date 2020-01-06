from pipeline.datasets.cartoon_set_eye_color import create_eye_color_datagens
from pipeline.datasets.utilities import get_X_y_test_sets, go_up_three_dirs
from pipeline.models.mlp import train_mlp
from pipeline.plotting.plotting import plot_train_loss_acc_lr, plot_top_losses
import os

class B2_MLP:
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule,
            batch_size=16,
            test_size=0.2, 
            validation_split=0.2, 
            random_state=42,
            first_af="relu",
            second_af="relu",
            layer1_hn=300,
            layer2_hn=100):
        self.height = 299 
        self.width = 299
        self.num_classes = 5
        self.epochs = epochs
        self.eye_color_train_gen, self.eye_color_val_gen, self.eye_color_test_gen = create_eye_color_datagens(
            height=self.height,
            width=self.width,
            batch_size=batch_size,
            test_size=test_size, 
            validation_split=validation_split, 
            random_state=random_state,
            preprocessing_function=None)
        self.model, self.history, self.schedule = train_mlp(
            self.height, 
            self.width,
            self.num_classes,
            batch_size,
            self.epochs,
            learning_rate,
            schedule,
            self.eye_color_train_gen,
            self.eye_color_val_gen,
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
            "B2",
            "output/train_loss_acc_B2_mlp.png",
            "output/lr_B2_mlp.png")

        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Go back to image folder
        os.chdir("data/dataset_AMLS_19-20/cartoon_set")

        # Split ImageDataGenerator object for the test set into separate X and y test sets
        eye_color_X_test, eye_color_y_test = get_X_y_test_sets(self.eye_color_test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, eye_color_X_test, eye_color_y_test, "output/plot_top_losses_B2_mlp")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(eye_color_X_test, eye_color_y_test)[-1]
        return test_accuracy