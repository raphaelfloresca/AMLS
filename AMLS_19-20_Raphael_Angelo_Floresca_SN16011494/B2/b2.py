from pipeline.datasets.cartoon_set_eye_color import create_eye_color_df
from pipeline.datasets.utilities import get_X_y_test_sets, go_up_three_dirs, create_datagens, data_dir, cartoon_set_dir
from pipeline.models.mlp import train_mlp
from pipeline.models.cnn import train_cnn
from pipeline.models.xception import train_xception
from pipeline.plotting.plotting import plot_train_loss_acc_lr, plot_top_losses, plot_grad_cam
import os
from tensorflow.keras.applications.xception import preprocess_input

class B2:
    height = 500
    width = 500
    num_classes = 5
    batch_size = 32
    random_state = 42
    df = create_eye_color_df()

    train_gen, val_gen, test_gen = create_datagens(
        height,
        width,
        df,
        "cartoon_set",
        "file_name",
        "eye_color",
        batch_size,
        random_state,
        None)


class B2MLP(B2):
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule_type,
            find_lr,
            random_state,
            first_af="relu",
            second_af="relu",
            layer1_hn=300,
            layer2_hn=100):

        # Change to relevant image set directory
        os.chdir(os.path.join(data_dir, cartoon_set_dir))

        # Change random state according to constructor
        self.random_state = random_state
        B2.random_state = self.random_state

        self.epochs = epochs
        self.find_lr = find_lr
        self.schedule_type = schedule_type

        self.train_gen, self.val_gen, self.test_gen = B2.train_gen, B2.val_gen, B2.test_gen
        
        if find_lr == True:
            self.lr_finder = train_mlp(
            B2.height, 
            B2.width,
            B2.num_classes,
            B2.batch_size,
            self.epochs,
            learning_rate,
            schedule_type,
            find_lr,
            self.train_gen,
            self.val_gen,
            first_af,
            second_af,
            layer1_hn,
            layer2_hn)
        else:
            print("Training MLP...")
            self.model, self.history, self.schedule = train_mlp(
                B2.height, 
                B2.width,
                B2.num_classes,
                B2.batch_size,
                self.epochs,
                learning_rate,
                schedule_type,
                find_lr,
                self.train_gen,
                self.val_gen,
                first_af,
                second_af,
                layer1_hn,
                layer2_hn)

    def train(self):
        if self.find_lr == True:
            # Navigate to output folder in parent directory
            go_up_three_dirs()        

            # Plot learning rate finder plot
            self.lr_finder.plot_loss(
                "output/lr_finder_plot_B2.png"
            )
        else:
            # Plot training loss accuracy and learning rate change
            # Navigate to output folder in parent directory
            go_up_three_dirs()

            plot_train_loss_acc_lr(
                self.history,
                self.epochs,
                self.schedule,
                self.schedule_type,
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
        X_test, y_test = get_X_y_test_sets(self.test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, X_test, y_test, "output/plot_top_losses_B2_mlp.png")

        # Plot GradCam
        plot_grad_cam(self.model, X_test, y_test, 3, "conv2d_2", "output/plot_top_5_gradcam_B2_mlp.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(X_test, y_test)[-1]
        return test_accuracy


class B2CNN(B2):
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule_type,
            find_lr,
            random_state,
            num_start_filters=16,
            kernel_size=3,
            fcl_size=512):

        # Change to relevant image set directory
        os.chdir(os.path.join(data_dir, cartoon_set_dir))

        # Change random state according to constructor
        self.random_state = random_state
        B2.random_state = self.random_state

        self.epochs = epochs
        self.find_lr = find_lr
        self.schedule_type = schedule_type

        self.train_gen, self.val_gen, self.test_gen = B2.train_gen, B2.val_gen, B2.test_gen
        
        if find_lr == True:
            self.lr_finder = train_cnn(
            B2.height, 
            B2.width,
            B2.num_classes,
            B2.batch_size,
            self.epochs,
            learning_rate,
            schedule_type,
            find_lr,
            self.train_gen,
            self.val_gen,
            num_start_filters,
            kernel_size,
            fcl_size)
        else:
            print("Training CNN...")
            self.model, self.history, self.schedule = train_cnn(
                B2.height, 
                B2.width,
                B2.num_classes,
                B2.batch_size,
                self.epochs,
                learning_rate,
                schedule_type,
                find_lr,
                self.train_gen,
                self.val_gen,
                num_start_filters,
                kernel_size,
                fcl_size)

    def train(self):
        if self.find_lr == True:
            # Navigate to output folder in parent directory
            go_up_three_dirs()        

            # Plot learning rate finder plot
            self.lr_finder.plot_loss(
                "output/lr_finder_plot_B2.png"
            )
        else:
            # Plot training loss accuracy and learning rate change
            # Navigate to output folder in parent directory
            go_up_three_dirs()

            plot_train_loss_acc_lr(
                self.history,
                self.epochs,
                self.schedule,
                self.schedule_type,
                "B2",
                "output/train_loss_acc_B2_cnn.png",
                "output/lr_B2_cnn.png")

            # Get the training accuracy
            training_accuracy = self.history.history['acc'][-1]
            return training_accuracy

    def test(self):
        # Go back to image folder
        os.chdir("data/dataset_AMLS_19-20/cartoon_set")

        # Split ImageDataGenerator object for the test set into separate X and y test sets
        X_test, y_test = get_X_y_test_sets(self.test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, X_test, y_test, "output/plot_top_losses_B2_cnn.png")

        # Plot GradCam
        plot_grad_cam(self.model, X_test, y_test, 3, "conv2d_2", "output/plot_top_5_gradcam_B2_cnn.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(X_test, y_test)[-1]
        return test_accuracy

class B2Xception(B2):
    def __init__(
            self,
            epochs,
            learning_rate,
            schedule_type,
            find_lr,
            random_state,
            frozen_model_path="B2_frozen_model.h5",
            frozen_training_plot_path="train_loss_acc_B2_xception_frozen.png",
            frozen_training_plot_name="B2 (frozen model)"):

        # Change to relevant image set directory
        os.chdir(os.path.join(data_dir, cartoon_set_dir))

        # Change random state according to constructor
        self.random_state = random_state
        B2.random_state = self.random_state

        self.epochs = epochs
        self.find_lr = find_lr
        self.schedule_type = schedule_type

        self.train_gen, self.val_gen, self.test_gen = create_datagens(
            B2.height,
            B2.width,
            B2.df,
            "cartoon_set",
            "file_name",
            "eye_color",
            B2.batch_size,
            B2.random_state,
            preprocess_input)

        # Change to relevant image set directory
        os.chdir(os.path.join(data_dir, cartoon_set_dir))
        
        if find_lr == True:
            self.lr_finder = train_xception(
            B2.height, 
            B2.width,
            B2.num_classes,
            B2.batch_size,
            self.epochs,
            learning_rate,
            schedule_type,
            find_lr,
            self.train_gen,
            self.val_gen,
            frozen_model_path,
            frozen_training_plot_path,
            frozen_training_plot_name)
        else:
            print("Training Xception...")
            self.model, self.history, self.schedule = train_xception(
                B2.height, 
                B2.width,
                B2.num_classes,
                B2.batch_size,
                self.epochs,
                learning_rate,
                schedule_type,
                find_lr,
                self.train_gen,
                self.val_gen,
                frozen_model_path,
                frozen_training_plot_path,
                frozen_training_plot_name)

    def train(self):
        if self.find_lr == True:
            # Navigate to output folder in parent directory
            go_up_three_dirs()        

            # Plot learning rate finder plot
            self.lr_finder.plot_loss(
                "output/lr_finder_plot_B2.png"
            )
        else:
            # Plot training loss accuracy and learning rate change
            # Navigate to output folder in parent directory
            go_up_three_dirs()

            plot_train_loss_acc_lr(
                self.history,
                int(self.epochs/2),
                self.schedule,
                self.schedule_type,
                "B2",
                "output/train_loss_acc_B2_xception.png",
                "output/lr_B2_xception.png")

            # Get the training accuracy
            training_accuracy = self.history.history['acc'][-1]
            return training_accuracy

    def test(self):
        # Go back to image folder
        os.chdir("data/dataset_AMLS_19-20/cartoon_set")

        # Split ImageDataGenerator object for the test set into separate X and y test sets
        X_test, y_test = get_X_y_test_sets(self.test_gen)

        # Navigate to output folder in parent directory
        go_up_three_dirs()

        # Plot top losses
        plot_top_losses(self.model, X_test, y_test, "output/plot_top_losses_B2_xception.png")

        # Plot GradCam
        plot_grad_cam(self.model, X_test, y_test, 3, "block14_sepconv2", "output/plot_top_5_gradcam_B2_xception.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(X_test, y_test)[-1]
        return test_accuracy