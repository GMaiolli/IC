# Instalar dependências
# Execute no terminal: pip install earthengine-api tensorflow matplotlib seaborn scikit-learn pillow requests

# Imports
import ee
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# Inicializar Earth Engine
try:
    ee.Initialize(project='ndvi-analysis-455514')
except Exception:
    print("Autenticação necessária. Execute 'earthengine authenticate' no terminal.")
    raise SystemExit

# Parâmetros
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'
CLOUD_FILTER = 20
NUM_BEST_IMAGES = 5
NUM_SAMPLES = 1000
RANDOM_SEED = 42
PATCH_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
MAX_VALID_PATCHES = 50  # Limite de patches válidos

NDVI_CLASSES = {
    0: {'nome': 'Solo exposto', 'limiar': [-1.0, 0.177], 'cor': 'red'},
    1: {'nome': 'Baixa', 'limiar': [0.177, 0.331], 'cor': 'orange'},
    2: {'nome': 'Média baixa', 'limiar': [0.331, 0.471], 'cor': 'yellow'},
    3: {'nome': 'Média', 'limiar': [0.471, 0.584], 'cor': 'yellowgreen'},
    4: {'nome': 'Média alta', 'limiar': [0.584, 0.7], 'cor': 'green'},
    5: {'nome': 'Alta', 'limiar': [0.7, 1.0], 'cor': 'darkgreen'}
}

ROIs = [
    ee.Geometry.Rectangle([-48.0, -16.0, -47.5, -15.5]),
    ee.Geometry.Rectangle([-60.0, -3.0, -59.5, -2.5]),
    ee.Geometry.Rectangle([-39.0, -9.0, -38.5, -8.5]),
    ee.Geometry.Rectangle([-57.0, -17.0, -56.5, -16.5]),
    ee.Geometry.Rectangle([-46.0, -23.0, -45.5, -22.5]),
    ee.Geometry.Rectangle([-53.0, -31.0, -52.5, -30.5]),
    ee.Geometry.Rectangle([-43.3, -22.95, -43.2, -22.85]),
    ee.Geometry.Rectangle([-123.0, 49.0, -122.5, 49.5]),
    ee.Geometry.Rectangle([23.0, 19.0, 23.5, 19.5]),
    ee.Geometry.Rectangle([100.0, 0.5, 100.5, 1.0]),
    ee.Geometry.Rectangle([30.0, -2.0, 30.5, -1.5]),
    ee.Geometry.Rectangle([135.0, -33.0, 135.5, -32.5]),
    ee.Geometry.Rectangle([5.0, 52.0, 5.5, 52.5])
]

# Classes de NDVI
NDVI_CLASSES = {
    0: {'nome': 'Solo exposto', 'limiar': [-1.0, 0.177], 'cor': 'red'},
    1: {'nome': 'Baixa', 'limiar': [0.177, 0.331], 'cor': 'orange'},
    2: {'nome': 'Média baixa', 'limiar': [0.331, 0.471], 'cor': 'yellow'},
    3: {'nome': 'Média', 'limiar': [0.471, 0.584], 'cor': 'yellowgreen'},
    4: {'nome': 'Média alta', 'limiar': [0.584, 0.7], 'cor': 'green'},
    5: {'nome': 'Alta', 'limiar': [0.7, 1.0], 'cor': 'darkgreen'}
}

# Função para calcular o NDVI
def calculate_ndvi(img):
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Função para calcular o NDWI
def calculate_ndwi(img):
    return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

# Função para calcular o MNDWI
def calculate_mndwi(img):
    return img.normalizedDifference(['B3', 'B11']).rename('MNDWI')

# Função para classificar NDVI, NDWI e MNDWI
def classify_ndwi_mndwi_and_ndvi(ndvi, ndwi, mndwi):
    classified = ee.Image(-1).rename('ndvi_class')
    for class_id in NDVI_CLASSES:
        min_val, max_val = NDVI_CLASSES[class_id]['limiar']
        mask = ndvi.gte(min_val).And(ndvi.lt(max_val))
        classified = classified.where(mask, class_id)

    # Classificar água com base tanto no NDWI quanto no MNDWI (valores positivos)
    water_mask = ndwi.gt(0).And(mndwi.gt(0))  # Áreas identificadas como água por ambos os índices
    classified = classified.where(water_mask, 6)  # Classe 6: Água

    return classified

# Função para exibir e salvar imagens
def save_and_show_image(url, title):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(f"{title}.png")
    print(f"Imagem salva como {title}.png")
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Coleta e filtragem
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterDate(START_DATE, END_DATE) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)) \
    .filterBounds(ROIs[0])  # Usar a primeira ROI para filtrar

# Selecionar as melhores imagens (com menor percentual de nuvens)
best_images = collection.sort('CLOUDY_PIXEL_PERCENTAGE').limit(NUM_BEST_IMAGES)

# Gerar uma imagem ponderada usando a média das melhores imagens
weighted_composite = best_images.mean()

# Calcular índices para a imagem ponderada
ndvi = calculate_ndvi(weighted_composite)
ndwi = calculate_ndwi(weighted_composite)
mndwi = calculate_mndwi(weighted_composite)

# Classificar NDVI, NDWI e MNDWI
classified_with_water_and_mndwi = classify_ndwi_mndwi_and_ndvi(ndvi, ndwi, mndwi)

# Paleta atualizada (incluindo cinza para -1 e azul para água)
full_palette_with_water_and_mndwi = ['gray'] + [NDVI_CLASSES[i]['cor'] for i in NDVI_CLASSES] + ['blue']

# URLs para visualização
urls_weighted_composite = {
    "RGB da imagem ponderada": weighted_composite.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000).getThumbURL({
        'region': ROIs[0],
        'dimensions': '512x512',
        'format': 'png'
    }),
    "NDVI da imagem ponderada": ndvi.visualize(min=-1, max=1, palette=['blue', 'white', 'green']).getThumbURL({
        'region': ROIs[0],
        'dimensions': '512x512',
        'format': 'png'
    }),
    "NDWI da imagem ponderada": ndwi.visualize(min=-1, max=1, palette=['white', 'blue']).getThumbURL({
        'region': ROIs[0],
        'dimensions': '512x512',
        'format': 'png'
    }),
    "MNDWI da imagem ponderada": mndwi.visualize(min=-1, max=1, palette=['white', 'cyan']).getThumbURL({
        'region': ROIs[0],
        'dimensions': '512x512',
        'format': 'png'
    }),
    "Mapa NDVI, NDWI e MNDWI classificado": classified_with_water_and_mndwi.visualize(
        min=-1, max=6, palette=full_palette_with_water_and_mndwi
    ).getThumbURL({
        'region': ROIs[0],
        'dimensions': '512x512',
        'format': 'png'
    })
}

# Exibir resultados
print("Exibindo as imagens ponderadas geradas a partir das melhores imagens:")
for name, url in urls_weighted_composite.items():
    save_and_show_image(url, name)

# Amostragem
samples_list = []
for roi in ROIs:
    pts = ee.FeatureCollection.randomPoints(roi, NUM_SAMPLES // len(ROIs), seed=RANDOM_SEED)
    s = composite.sampleRegions(collection=pts, scale=10, geometries=False)
    samples_list.append(s)

all_samples = samples_list[0]
for s in samples_list[1:]:
    all_samples = all_samples.merge(s)

samples_dict = all_samples.getInfo()
ndvi_vals, labels = [], []
for feat in samples_dict['features']:
    prop = feat['properties']
    if 'NDVI' in prop and 'ndvi_class' in prop and prop['ndvi_class'] != -1:
        ndvi_vals.append(prop['NDVI'])
        labels.append(int(prop['ndvi_class']))

print(f"✅ Total de amostras válidas coletadas: {len(ndvi_vals)}")
print("Distribuição das classes:", Counter(labels))

# Treinamento do modelo CNN
def create_cnn(input_shape, classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X = np.array(ndvi_vals)[..., np.newaxis]
y = np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=RANDOM_SEED)

y_train_cat = to_categorical(y_train, num_classes=len(NDVI_CLASSES))
y_val_cat = to_categorical(y_val, num_classes=len(NDVI_CLASSES))

model = create_cnn((PATCH_SIZE, PATCH_SIZE, 1), len(NDVI_CLASSES))
history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=EPOCHS)

# Salvar modelo localmente
model.save('cnn_model_ndvi.h5')
print("✅ Modelo salvo como cnn_model_ndvi.h5")