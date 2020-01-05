from pipeline.datasets.cartoon_set_eye_color import create_eye_color_datagens
from pipeline.models.xception import train_xception
from tensorflow.keras import backend as K

class A1_Xception:
    def __init__(
            self, 
            batch_size=32, 
            test_size=0.2, 
            validation_split=0.2, 
            epochs=5, 
            random_state=42):
        self.height = 250 
        self.width = 250
        self.num_classes = 2
        self.eye_color_train_gen, self.eye_color_val_gen, self.eye_color_test_gen = create_eye_color_datagens(
            height=self.height,
            width=self.width,
            batch_size=batch_size,
            test_size=test_size, 
            validation_split=validation_split, 
            random_state=random_state)
        self.model, self.history = train_xception(
            self.height, 
            self.width,
            self.num_classes,
            epochs,
            batch_size,
            self.eye_color_train_gen,
            self.eye_color_val_gen)

    def train(self):
        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Clear gpu memory
        K.clear_session()

        # Get the test accuracy
        test_accuracy = self.model.evaluate(self.eye_color_test_gen)[-1]
        return test_accuracy