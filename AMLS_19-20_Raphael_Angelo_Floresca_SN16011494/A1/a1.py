from pipeline.datasets.celeba_gender import create_gender_datagens
from pipeline.models.mlp import MLP
from pipeline.models.cnn import CNN
from pipeline.models.resnet50_v2 import ResNet50V2_TL

class A1:
    def __init__(self, batch_size=32, test_size=0.2, validation_split=0.2, random_state=42):
        self.height = 218 
        self.width = 178
        self.num_classes = 2
        self.batch_size = batch_size 
        self.test_size = test_size
        self.validation_split = validation_split
        self.random_state = random_state
        self.gender_train_gen, self.gender_val_gen, self.gender_test_gen = create_gender_datagens(
            batch_size=batch_size,
            test_size=test_size, 
            validation_split=validation_split, 
            random_state=random_state)

    def mlp_train(
            self,
            epochs,
            first_af="relu",
            second_af="relu",
            layer1_hn=300,
            layer2_hn=100):

        # Build the model
        mlp_model = MLP.build(
            self.height, 
            self.width,
            self.num_classes, 
            first_af=first_af, 
            second_af=second_af, 
            layer1_hn=layer1_hn, 
            layer2_hn=layer2_hn)

        # We now compile the MLP model to specify the loss function
        # and the optimizer to use (SGD)
        mlp_model.compile(
            loss="sparse_categorical_crossentropy", # b/c of exclusive, sparse outputs
            optimizer='sgd', # We use SGD to optimise the ANN
            metrics=["accuracy"]) # Used for classifiers

        # Training and evaluating the MLP model on the gender dataset
        gender_history = mlp_model.fit(
            self.gender_train_gen,
            steps_per_epoch=self.gender_train_gen.samples // self.batch_size,
            validation_data=self.gender_val_gen,
            validation_steps=self.gender_val_gen.samples // self.batch_size,
            epochs=epochs)

        # Get the training accuracy
        training_accuracy = gender_history.history.get('accuracy')[-1]

        return training_accuracy, mlp_model
        
    def mlp_test(self, model):
        # Get the test accuracy
        test_accuracy = model.evaluate(self.gender_test_gen)[-1]
        return test_accuracy