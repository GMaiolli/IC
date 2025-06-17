# Instalar dependências:
# pip install earthengine-api tensorflow matplotlib seaborn scikit-learn pillow requests numpy

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
import pickle
import os

# Configurar matplotlib para mostrar em janelas separadas
plt.ion()  # Modo interativo

# Criar diretório para salvar dados localmente
SAVE_DIR = './results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Inicializar Earth Engine com autenticação via link (igual ao Colab)
def initialize_earth_engine():
    try:
        # Tentar inicializar sem projeto primeiro (caso já tenha um padrão)
        ee.Initialize()
        print("✅ Earth Engine já está autenticado")
    except Exception as e:
        try:
            print("🔐 Iniciando autenticação do Earth Engine...")
            print("Siga as instruções abaixo para autenticação:")
            ee.Authenticate()
            
            # Tentar inicializar sem projeto
            ee.Initialize()
            print("✅ Earth Engine autenticado com sucesso!")
        except Exception as e2:
            try:
                # Se falhar, usar um projeto padrão do Earth Engine
                print("🔧 Configurando projeto padrão...")
                ee.Initialize(project='ndvi-analysis-455514')
                print("✅ Earth Engine autenticado com projeto padrão!")
            except Exception as e3:
                # Última tentativa: pedir para o usuário criar um projeto
                print("❌ Erro na inicialização do Earth Engine.")
                print("📋 Você precisa de um projeto no Google Cloud Platform.")
                print("🔗 Visite: https://console.cloud.google.com/")
                print("1. Crie um novo projeto (ou use um existente)")
                print("2. Ative a Earth Engine API")
                print("3. Execute o código novamente")
                raise e3

# Parâmetros
START_DATE = '2022-01-01'
END_DATE = '2022-12-31'
CLOUD_FILTER = 20
NUM_SAMPLES = 3000
RANDOM_SEED = 42
PATCH_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
NUM_BEST_IMAGES = 5  # Número de melhores imagens a serem usadas para o composto

# Classes de NDVI
NDVI_CLASSES = {
    0: {'nome': 'Solo exposto', 'limiar': [-1.0, 0.177], 'cor': 'red'},
    1: {'nome': 'Baixa', 'limiar': [0.177, 0.331], 'cor': 'orange'},
    2: {'nome': 'Média baixa', 'limiar': [0.331, 0.471], 'cor': 'yellow'},
    3: {'nome': 'Média', 'limiar': [0.471, 0.584], 'cor': 'yellowgreen'},
    4: {'nome': 'Média alta', 'limiar': [0.584, 0.7], 'cor': 'green'},
    5: {'nome': 'Alta', 'limiar': [0.7, 1.0], 'cor': 'darkgreen'}
}

# Função para criar as ROIs (será chamada após inicialização do EE)
def create_rois():
    return [
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

# Funções para calcular índices
def calculate_ndvi(img):
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')

def calculate_ndwi(img):
    return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

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
    water_mask = ndwi.gt(0).And(mndwi.gt(0))
    classified = classified.where(water_mask, 6)  # Classe 6: Água

    return classified

# Função para exibir e salvar imagens localmente
def save_and_show_image(url, title, save_to_disk=True):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        if save_to_disk:
            img.save(f"{SAVE_DIR}/{title}.png")
            print(f"✅ Imagem salva como {SAVE_DIR}/{title}.png")
        
        # Mostrar imagem em janela separada
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)  # Não bloquear execução
        
        return img
    except Exception as e:
        print(f"❌ Erro ao baixar/mostrar imagem {title}: {e}")
        return None

# Função para visualizar uma região de treinamento específica
def visualize_training_region(roi_index, ROIs):
    roi = ROIs[roi_index]

    print(f"🔍 Visualizando região de treinamento {roi_index}...")

    try:
        # Coleta e filtragem
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(START_DATE, END_DATE) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)) \
            .filterBounds(roi)

        # Verificar se há imagens disponíveis
        count = collection.size().getInfo()
        if count == 0:
            print(f"⚠️ Nenhuma imagem encontrada para região {roi_index}")
            return

        print(f"📊 Encontradas {count} imagens para região {roi_index}")

        # Selecionar as melhores imagens
        best_images = collection.sort('CLOUDY_PIXEL_PERCENTAGE').limit(NUM_BEST_IMAGES)

        # Criar composto
        weighted_composite = best_images.mean()

        # Calcular índices
        ndvi = calculate_ndvi(weighted_composite)
        ndwi = calculate_ndwi(weighted_composite)
        mndwi = calculate_mndwi(weighted_composite)

        # Classificar
        classified_with_water = classify_ndwi_mndwi_and_ndvi(ndvi, ndwi, mndwi)

        # Paleta para visualização
        full_palette_with_water = ['gray'] + [NDVI_CLASSES[i]['cor'] for i in NDVI_CLASSES] + ['blue']

        # URLs de visualização
        print("🌐 Gerando URLs de visualização...")
        urls = {
            "RGB": weighted_composite.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000).getThumbURL({
                'region': roi,
                'dimensions': '512x512',
                'format': 'png'
            }),
            "NDVI": ndvi.visualize(min=-1, max=1, palette=['blue', 'white', 'green']).getThumbURL({
                'region': roi,
                'dimensions': '512x512',
                'format': 'png'
            }),
            "NDWI": ndwi.visualize(min=-1, max=1, palette=['white', 'blue']).getThumbURL({
                'region': roi,
                'dimensions': '512x512',
                'format': 'png'
            }),
            "MNDWI": mndwi.visualize(min=-1, max=1, palette=['white', 'cyan']).getThumbURL({
                'region': roi,
                'dimensions': '512x512',
                'format': 'png'
            }),
            "Classificação": classified_with_water.visualize(
                min=-1, max=6, palette=full_palette_with_water
            ).getThumbURL({
                'region': roi,
                'dimensions': '512x512',
                'format': 'png'
            })
        }

        # Mostrar e salvar imagens
        print("🖼️ Baixando e exibindo imagens...")
        for name, url in urls.items():
            save_and_show_image(url, f"{name}_Regiao_{roi_index}", save_to_disk=False)
            
    except Exception as e:
        print(f"❌ Erro ao processar região {roi_index}: {e}")

# Função para criar modelo CNN
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

# Função para coletar amostras e treinar o modelo
def collect_samples_and_train(ROIs):
    print("🚀 Iniciando coleta de amostras para treinamento...")

    # Lista para armazenar amostras
    samples_list = []

    # Para cada região de interesse
    for i, roi in enumerate(ROIs):
        print(f"📍 Processando região {i+1}/{len(ROIs)}...")

        try:
            # Coletar imagens
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(START_DATE, END_DATE) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)) \
                .filterBounds(roi)

            # Verificar se há imagens
            count = collection.size().getInfo()
            if count == 0:
                print(f"⚠️ Região {i+1} sem imagens válidas, pulando...")
                continue

            # Selecionar as melhores imagens
            best_images = collection.sort('CLOUDY_PIXEL_PERCENTAGE').limit(NUM_BEST_IMAGES)
            composite = best_images.mean()

            # Calcular índices
            ndvi = calculate_ndvi(composite)
            ndwi = calculate_ndwi(composite)
            mndwi = calculate_mndwi(composite)

            # Classificar NDVI e água
            classified = classify_ndwi_mndwi_and_ndvi(ndvi, ndwi, mndwi)

            # Preparar imagem para amostragem
            image_for_sampling = ndvi.addBands(classified)

            # Coletar pontos aleatórios dentro da região
            pts = ee.FeatureCollection.randomPoints(roi, NUM_SAMPLES // len(ROIs), seed=RANDOM_SEED)

            # Extrair valores de NDVI e classes nos pontos
            s = image_for_sampling.sampleRegions(collection=pts, scale=10, geometries=False)
            samples_list.append(s)
            
        except Exception as e:
            print(f"❌ Erro na região {i+1}: {e}")
            continue

    if not samples_list:
        raise Exception("❌ Nenhuma amostra foi coletada!")

    # Mesclar todas as amostras
    print("🔄 Mesclando amostras...")
    all_samples = samples_list[0]
    for s in samples_list[1:]:
        all_samples = all_samples.merge(s)

    # Converter para dicionário Python
    print("📥 Baixando dados do Earth Engine...")
    samples_dict = all_samples.getInfo()

    # Extrair valores NDVI e classes
    ndvi_vals, labels = [], []
    for feat in samples_dict['features']:
        prop = feat['properties']
        if 'NDVI' in prop and 'ndvi_class' in prop and prop['ndvi_class'] != -1:
            ndvi_vals.append(prop['NDVI'])
            labels.append(int(prop['ndvi_class']))

    print(f"✅ Total de amostras válidas coletadas: {len(ndvi_vals)}")
    print("📊 Distribuição das classes:", Counter(labels))

    # Salvar amostras coletadas
    with open(f"{SAVE_DIR}/ndvi_samples.pkl", "wb") as f:
        pickle.dump({
            'ndvi_vals': ndvi_vals,
            'labels': labels
        }, f)
    print(f"💾 Amostras salvas em {SAVE_DIR}/ndvi_samples.pkl")

    # Criar patches para treinamento
    print("🧠 Criando patches para treinamento...")
    X = np.array([
        np.clip(np.random.normal(loc=v, scale=0.05, size=(PATCH_SIZE, PATCH_SIZE)), -1, 1)
        for v in ndvi_vals
    ])
    X = X[..., np.newaxis]
    y = np.array(labels)

    # Dividir em conjuntos de treino e validação
    unique_counts = Counter(y)
    if min(unique_counts.values()) < 2:
        print("⚠️ Poucas amostras em algumas classes. Realizando divisão sem estratificação.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=RANDOM_SEED)

    # One-hot encoding das classes
    num_classes = max(labels) + 1
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    # Criar e treinar modelo CNN
    print("🏗️ Criando modelo CNN...")
    model = create_cnn((PATCH_SIZE, PATCH_SIZE, 1), num_classes)
    model.summary()

    # Preparar datasets para treinamento
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat)).shuffle(1000).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(BATCH_SIZE)

    # Treinar modelo
    print("🎯 Iniciando treinamento...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ])

    # Visualizar curvas de treinamento
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Acurácia')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    print(f"💾 Curvas de treinamento salvas em {SAVE_DIR}/training_curves.png")

    # Avaliar modelo
    print("📊 Avaliando modelo...")
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis], annot=True, cmap='Blues',
            xticklabels=[
                NDVI_CLASSES[i]['nome'] if i in NDVI_CLASSES else 'Água'
                for i in sorted(np.unique(y))
            ],
            yticklabels=[
                NDVI_CLASSES[i]['nome'] if i in NDVI_CLASSES else 'Água'
                for i in sorted(np.unique(y))
            ])
    plt.title("Matriz de Confusão Normalizada")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    print(f"💾 Matriz de confusão salva em {SAVE_DIR}/confusion_matrix.png")

    # Mostrar relatório de classificação
    labels_presentes = sorted(list(unique_labels(y_val, y_pred)))
    nomes_presentes = [
        NDVI_CLASSES[i]['nome'] if i in NDVI_CLASSES else 'Água'
        for i in labels_presentes
    ]
    class_report = classification_report(y_val, y_pred, target_names=nomes_presentes)
    print("\n📈 Relatório de Classificação:")
    print(class_report)

    # Salvar relatório de classificação em arquivo de texto
    with open(f"{SAVE_DIR}/classification_report.txt", "w", encoding='utf-8') as f:
        f.write(class_report)
    print(f"💾 Relatório salvo em {SAVE_DIR}/classification_report.txt")

    # Salvar modelo
    model.save(f"{SAVE_DIR}/cnn_ndvi_classifier.h5")
    print(f"💾 Modelo salvo em {SAVE_DIR}/cnn_ndvi_classifier.h5")

    # Salvar classes NDVI (para ser usado no script de previsão)
    with open(f"{SAVE_DIR}/ndvi_classes.pkl", "wb") as f:
        pickle.dump(NDVI_CLASSES, f)
    print(f"💾 Classes NDVI salvas em {SAVE_DIR}/ndvi_classes.pkl")

    print(f"\n✅ Modelo e dados salvos com sucesso em {SAVE_DIR}")

    return model

# Função principal para executar o treinamento completo
def run_training():
    print("🚀 Iniciando NDVI Classifier - Versão VS Code")
    print("=" * 50)
    
    # Inicializar Earth Engine
    initialize_earth_engine()
    
    # Criar ROIs após inicialização
    ROIs = create_rois()
    
    print("\n🔍 Visualizando região de exemplo antes do treinamento...")
    # Visualizar uma região de exemplo antes do treinamento
    visualize_training_region(2, ROIs)  # Mostra a região 2 (índice 2, que corresponde à 3ª região na lista)
    
    input("\n⏸️ Pressione Enter para continuar com o treinamento após ver as imagens...")

    # Treinar o modelo
    print("\n🎯 Iniciando processo de treinamento...")
    model = collect_samples_and_train(ROIs)

    print("\n" + "=" * 50)
    print("✅ TREINAMENTO COMPLETO!")
    print("=" * 50)
    print(f"📁 Diretório de resultados: {os.path.abspath(SAVE_DIR)}")
    print(f"🤖 Modelo salvo em: {SAVE_DIR}/cnn_ndvi_classifier.h5")
    print(f"📊 Classes NDVI salvas em: {SAVE_DIR}/ndvi_classes.pkl")
    print(f"💾 Amostras salvas em: {SAVE_DIR}/ndvi_samples.pkl")
    print(f"📈 Gráficos salvos em: {SAVE_DIR}/")
    print("\n🔮 Utilize um script de previsão para aplicar o modelo em novas áreas.")
    
    # Manter janelas abertas
    input("\n⏸️ Pressione Enter para fechar o programa...")

# Executar o treinamento
if __name__ == "__main__":
    run_training()