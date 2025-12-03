import tensorflow as tf
from tensorflow.keras import layers, Model


def criar_modelo_efficientnet(num_classes, img_size=(224, 224)):
    """
    Cria um modelo baseado em EfficientNetB0 com pesos ImageNet.
    """

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3)
    )

    # Não treinar nada inicialmente
    base_model.trainable = False

    inputs = layers.Input(shape=(*img_size, 3))

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    modelo = Model(inputs, outputs)
    return modelo


def liberar_camadas(modelo, qtd_camadas=20):
   
    base_model = None
    for layer in modelo.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNet não encontrada no modelo!")

    print(f"\nEfficientNet encontrada: {base_model.name}")
    print(f"Total de camadas: {len(base_model.layers)}")

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-qtd_camadas:]:
        layer.trainable = True

    print(f"{qtd_camadas} camadas finais liberadas para fine-tuning.")

    return modelo


def compilar_modelo(modelo, lr=1e-4):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


def resumo(modelo):
    treinaveis = sum(1 for l in modelo.layers if l.trainable)
    congeladas = sum(1 for l in modelo.layers if not l.trainable)

    print("\n===== RESUMO =====")
    print(f"Camadas treináveis: {treinaveis}")
    print(f"Camadas congeladas: {congeladas}")
    print("==================\n")

    modelo.summary()
