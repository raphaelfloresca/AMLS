from pipeline.datasets.cartoon_set_face_shape import create_face_shape_datagens
from pipeline.models.cnn import train_cnn
from numba import cuda

class B1_CNN:
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
        self.height = 500 
        self.width = 500
        self.num_classes = 5
        self.face_shape_train_gen, self.face_shape_val_gen, self.face_shape_test_gen = create_face_shape_datagens(
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
            self.face_shape_train_gen,
            self.face_shape_val_gen,
            num_start_filters,
            kernel_size,
            fcl_size)

    def train(self):
        # Release GPU memory
        cuda.select_device(0)
        cuda.close()

        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Release GPU memory
        cuda.select_device(0)
        cuda.close()

        # Get the test accuracy
        test_accuracy = self.model.evaluate(self.face_shape_test_gen)[-1]
        return test_accuracy