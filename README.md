
# Avaliação de Skins de Campeões do League of Legends
Uma breve descrição sobre o que esse projeto faz e para quem ele é


## Equipe
Lúcio da Cruz de Moraes
## Abordagem
O projeto utiliza de CNN (Redes Neurais Convolucionais) com tranfer learning para classificar identificar a qual campeão cada skin pertence.

Estrutura Metodológica:

1. Pré Processamento: 

Organização em pastas por classe (um campeão sendo uma classe portanto uma pasta)

Split automático em:
70% treino
30% validação

2. Data Loading

Utilização de image_dataset_from_directory

Normalização das imagens

Redimensionamento para 224 × 224

3. Modelo Utilizado

EfficientNetB0 pré-treinada no ImageNet

include_top=False para extração visual

Camada superior substituída por:

GlobalAveragePooling2D

BatchNormalization

Dropout (0.2)

Dense final com softmax

4. Transfer Learning – Etapa 1

EfficientNet congelada

Treinamento apenas da cabeça densa

5. Fine-Tuning – Etapa 2

Liberação das 25 últimas camadas da EfficientNet

Recompilação do modelo

Treinamento adicional com learning_rate=1e-5

6. Avaliação

Relatório de classificação (precision/recall/f1-score)

Matriz de confusão

Predições do modelo TFLite (opcional)

7. Exportação

modelo_final.keras

modelo_final.tflite
## Acurácias obtidas

Acurácia de treinamento: 1.22%

Acurácia de validação:0.47%
## Links

Link para acesso ao repositório: https://github.com/luciocruzmoraes/LolFinal.git

Link para acesso ao vídeo:

Link para acesso ao Dataset: https://www.kaggle.com/datasets/alihabibullah/league-of-legends-skin-splash-art-collection
## Instruções

Montar o Google Drive
from google.colab import drive
drive.mount('/content/drive')

2. Navegar até o projeto
%cd /content/drive/MyDrive/projeto_lol_cnn

3. Dividir o Dataset
from modulos.utils.split import split_dataset
split_dataset()

4. Carregar Datasets
from modulos.utils.cnn_treino import obter_imagens_treino, obter_imagens_validacao

treino = obter_imagens_treino("datasets/processed/train", (224,224), 16)
valid  = obter_imagens_validacao("datasets/processed/val",  (224,224), 16)

5. Criar Modelo
from modulos.modelos.efficientnet_finetune import criar_modelo_efficientnet
modelo = criar_modelo_efficientnet(num_classes)

6. Treinamento
historico = modelo.fit(treino, validation_data=valid, epochs=10)

7. Fine-Tuning
from modulos.modelos.efficientnet_finetune import liberar_camadas, compilar_modelo

liberar_camadas(modelo, qtd_camadas=25)
modelo = compilar_modelo(modelo, lr=1e-5)
historico2 = modelo.fit(treino, validation_data=valid, epochs=10)

8. Exportar Modelo
modelo.save("checkpoints/efficientnet_finetuned.keras")

TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
tflite_model = converter.convert()
open("checkpoints/modelo_final.tflite", "wb").write(tflite_model)