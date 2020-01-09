from pipeline.datasets.cartoon_set_face_shape import create_cartoon_set_df
from pipeline.datasets.utilities import get_X_y_test_sets, create_datagens, go_up_three_dirs, data_dir, cartoon_set_dir
from pipeline.models.mlp import train_mlp
from pipeline.models.cnn import train_cnn
from pipeline.plotting.plotting import plot_train_loss_acc_lr, plot_top_losses, plot_grad_cam
import os

class B1:
    height = 299
    width = 299
    num_classes = 5
    batch_size = 32
    random_state = 42
    df = create_cartoon_set_df()

    train_gen, val_gen, test_gen = create_datagens(
        height,
        width,
        df,
        "cartoon_set",
        "file_name",
        "face_shape",
        batch_size,
        random_state,
        None)

class B1MLP(B1):
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
        B1.random_state = self.random_state

        self.epochs = epochs
        self.find_lr = find_lr
        self.schedule_type = schedule_type

        self.train_gen, self.val_gen, self.test_gen = B1.train_gen, B1.val_gen, B1.test_gen
        
        if find_lr == True:
            self.lr_finder = train_mlp(
            B1.height, 
            B1.width,
            B1.num_classes,
            B1.batch_size,
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
                B1.height, 
                B1.width,
                B1.num_classes,
                B1.batch_size,
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
                "output/lr_finder_plot_B1.png"
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
                "B1",
                "output/train_loss_acc_B1_mlp.png",
                "output/lr_B1_cnn.png")

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
        plot_top_losses(self.model, X_test, y_test, "output/plot_top_losses_B1_mlp.png")

        # Plot GradCam
        plot_grad_cam(self.model, X_test, y_test, 3, "conv2d_2", "output/plot_top_5_gradcam_B1_mlp.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(X_test, y_test)[-1]
        return test_accuracy


class B1CNN(B1):
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
        B1.random_state = self.random_state

        self.epochs = epochs
        self.find_lr = find_lr
        self.schedule_type = schedule_type

        self.train_gen, self.val_gen, self.test_gen = B1.train_gen, B1.val_gen, B1.test_gen
        
        if find_lr == True:
            self.lr_finder = train_cnn(
            B1.height, 
            B1.width,
            B1.num_classes,
            B1.batch_size,
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
                B1.height, 
                B1.width,
                B1.num_classes,
                B1.batch_size,
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
                "output/lr_finder_plot_B1.png"
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
                "B1",
                "output/train_loss_acc_B1_cnn.png",
                "output/lr_B1_cnn.png")

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
        plot_top_losses(self.model, X_test, y_test, "output/plot_top_losses_B1_cnn.png")

        # Plot GradCam
        plot_grad_cam(self.model, X_test, y_test, 3, "conv2d_2", "output/plot_top_5_gradcam_B1_cnn.png")

        # Get the test accuracy
        test_accuracy = self.model.evaluate(X_test, y_test)[-1]
        return test_accuracy