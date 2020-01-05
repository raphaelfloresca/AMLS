from pipeline.datasets.cartoon_set_eye_color import create_eye_color_datagens
from pipeline.models.cnn import train_cnn

class B2_CNN:
    def __init__(
            self, 
            batch_size=32, 
            test_size=0.2, 
            validation_split=0.2, 
            epochs=10, 
            random_state=42,
            num_start_filters=16,
            kernel_size=3,
            fcl_size=512):
        self.height = 250 
        self.width = 250
        self.num_classes = 5
        self.eye_color_train_gen, self.eye_color_val_gen, self.eye_color_test_gen = create_eye_color_datagens(
            height=self.height,
            width=self.width,
            batch_size=batch_size,
            test_size=test_size, 
            validation_split=validation_split, 
            random_state=random_state)
        self.model, self.history = train_cnn(
            self.height, 
            self.width,
            self.num_classes,
            epochs,
            batch_size,
            self.eye_color_train_gen,
            self.eye_color_val_gen,
            num_start_filters,
            kernel_size,
            fcl_size)

    def train(self):
        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Get the test accuracy
        test_accuracy = self.model.evaluate(self.eye_color_test_gen)[-1]
        return test_accuracy