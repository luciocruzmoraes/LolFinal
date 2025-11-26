import tensorflow as tf

def obter_imagens_treino(train_path, img_size=(224, 224), batch_size=16):
    return tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        shuffle=True,
        batch_size=batch_size
    )

def obter_imagens_validacao(val_path, img_size=(224, 224), batch_size=16):
    return tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        shuffle=False,
        batch_size=batch_size
    )
