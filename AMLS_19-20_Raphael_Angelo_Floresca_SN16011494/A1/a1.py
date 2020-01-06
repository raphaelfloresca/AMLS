from pipeline.datasets.celeba_gender import create_gender_datagens
from pipeline.datasets.utilities import get_X_y_test_sets, go_up_three_dirs
from pipeline.models.mlp import train_mlp
from pipeline.plotting.plotting import plot_train_loss_acc_lr

class A1:
    def __init__(
            self,
            epochs,
            schedule,
            batch_size=32,
            test_size=0.2, 
            validation_split=0.2, 
            random_state=42,
            first_af="relu",
            second_af="relu",
            layer1_hn=300,
            layer2_hn=100):
        self.height = 218 
        self.width = 178
        self.num_classes = 2
        self.epochs=epochs
        self.gender_train_gen, self.gender_val_gen, self.gender_test_gen = create_gender_datagens(
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
            schedule,
            self.gender_train_gen,
            self.gender_val_gen,
            first_af,
            second_af,
            layer1_hn,
            layer2_hn)

    def train(self):
        go_up_three_dirs()        

        # Plot training loss accuracy and learning rate change
        plot_train_loss_acc_lr(
            self.history,
            self.epochs,
            self.schedule,
            "output/train_loss_acc_a1.png",
            "output/lr_a1.png")

        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Split ImageDataGenerator object for the test set into separate X and y test sets
        gender_X_test, gender_y_test = get_X_y_test_sets(self.gender_test_gen)

        # Get the test accuracy
        test_accuracy = self.model.evaluate(gender_X_test, gender_y_test)[-1]
        return test_accuracy