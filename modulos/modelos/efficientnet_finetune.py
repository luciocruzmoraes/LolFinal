import tensorflow as tf
from tensorflow import keras

def criar_modelo_efficientnet(num_classes):
    IMG_SIZE = (224, 224, 3)

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE
    )

    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE)
    x = base(inputs, training=False)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def liberar_camadas(modelo, qtd_camadas=30):
    """
    Descongela as últimas N camadas da EfficientNet
    """
    base = modelo.layers[1]  

    base.trainable = True

    for layer in base.layers[:-qtd_camadas]:
        layer.trainable = False

    print(f"Fine-tuning ativado: {qtd_camadas} camadas.")
