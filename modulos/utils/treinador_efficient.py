import tensorflow as tf
from tensorflow import keras

def compilar_modelo(modelo, lr=0.0001):
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


def criar_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.3,
            patience=2,
            monitor="val_loss",
            verbose=1
        )
    ]


def treinar(modelo, treino, validacao, epocas=20, callbacks=[]):
    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=epocas,
        callbacks=callbacks
    )
    return historico
