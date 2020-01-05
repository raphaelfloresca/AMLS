from pipeline.datasets.celeba_gender import create_gender_datagens
from pipeline.models.xception import train_xception
from tensorflow.keras.applications.Xception import preprocess_input

class A1_Xception:
    def __init__(
            self, 
            batch_size=10, 
            test_size=0.2, 
            validation_split=0.2, 
            epochs=4, 
            random_state=42):
        self.height = 299
        self.width = 299
        self.num_classes = 2
        self.gender_train_gen, self.gender_val_gen, self.gender_test_gen = create_gender_datagens(
            height=self.height,
            width=self.width,
            batch_size=batch_size,
            test_size=test_size, 
            validation_split=validation_split, 
            random_state=random_state,
            preprocessing_function=preprocess_input)
        self.model, self.history = train_xception(
            self.num_classes,
            epochs,
            batch_size,
            self.gender_train_gen,
            self.gender_val_gen)

    def train(self):
        # Get the training accuracy
        training_accuracy = self.history.history['acc'][-1]
        return training_accuracy
        
    def test(self):
        # Get the test accuracy
        test_accuracy = self.model.evaluate(self.gender_test_gen)[-1]
        return test_accuracy