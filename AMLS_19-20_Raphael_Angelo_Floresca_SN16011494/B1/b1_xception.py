from pipeline.datasets.cartoon_set_face_shape import create_face_shape_datagens
from pipeline.models.xception import train_xception
from tensorflow.keras import backend as K

class B1_Xception:
    def __init__(
            self, 
            batch_size=16, 
            test_size=0.2, 
            validation_split=0.2, 
            epochs=10, 
            random_state=42):
        self.height = 250 
        self.width = 250
        self.num_classes = 2
        self.face_shape_train_gen, self.face_shape_val_gen, self.face_shape_test_gen = create_face_shape_datagens(
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
            self.face_shape_train_gen,
            self.face_shape_val_gen)

    def train(self):
        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Get the test accuracy
        test_accuracy = self.model.evaluate(self.face_shape_test_gen)[-1]
        return test_accuracy