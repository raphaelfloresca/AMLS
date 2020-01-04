# Transfer learning using the ResNet50-v2 architecture pretrained on ImageNet.
# It has been modified to allow a custom input size.
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras import Model

class ResNet50V2_TL:
    @staticmethod
    def build(height, width, num_classes):
        # ResNet-50v2 is used as the base architecture for the model.
        # The top layers are not included in order to perform transfer learning.
        # Modified to allow for a custom input size
        base_model = ResNet50V2(weights="imagenet",
                                include_top=False,
                                input_shape=(height, width, 3)
                                )
        # Implement own pooling layer
        avg = GlobalAveragePooling2D()(base_model.output)
        # Output layer
        output = Dense(num_classes, activation="softmax")(avg)
        # Build model
        model = Model(inputs=base_model.input, outputs=output)
        return model