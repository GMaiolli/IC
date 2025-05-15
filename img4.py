# Instalar dependências
# Execute no terminal: pip install earthengine-api pillow requests matplotlib

# Imports
import ee
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Inicializar Earth Engine
try:
    ee.Initialize(project='ndvi-analysis-455514')
except Exception:
    print("Autenticação necessária. Execute 'earthengine authenticate' no terminal.")
    raise SystemExit

# Parâmetros
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'
CLOUD_FILTER = 20  # Permitir mais nuvens para garantir mais imagens
NUM_BEST_IMAGES = 5  # Número de melhores imagens a serem utilizadas para a composição ponderada

# Região de interesse
ROI = ee.Geometry.Rectangle([-123.0, 49.0, -122.5, 49.5])  # Quadrado perfeito

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
    .filterBounds(ROI)

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
        'region': ROI,
        'dimensions': '512x512',
        'format': 'png'
    }),
    "NDVI da imagem ponderada": ndvi.visualize(min=-1, max=1, palette=['blue', 'white', 'green']).getThumbURL({
        'region': ROI,
        'dimensions': '512x512',
        'format': 'png'
    }),
    "NDWI da imagem ponderada": ndwi.visualize(min=-1, max=1, palette=['white', 'blue']).getThumbURL({
        'region': ROI,
        'dimensions': '512x512',
        'format': 'png'
    }),
    "MNDWI da imagem ponderada": mndwi.visualize(min=-1, max=1, palette=['white', 'cyan']).getThumbURL({
        'region': ROI,
        'dimensions': '512x512',
        'format': 'png'
    }),
    "Mapa NDVI, NDWI e MNDWI classificado": classified_with_water_and_mndwi.visualize(
        min=-1, max=6, palette=full_palette_with_water_and_mndwi
    ).getThumbURL({
        'region': ROI,
        'dimensions': '512x512',
        'format': 'png'
    })
}

# Exibir resultados
print("Exibindo as imagens ponderadas geradas a partir das melhores imagens:")
for name, url in urls_weighted_composite.items():
    save_and_show_image(url, name)