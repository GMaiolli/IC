import streamlit as st
import ee
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
import atexit
import shutil
from typing import Dict, List, Tuple, Optional, Any
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

# ===================== CONFIGURAÇÕES E CONSTANTES =====================

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parâmetros compatíveis com o treinamento
PATCH_SIZE = 256  # Mesmo valor do treinamento
DEFAULT_BUFFER_SIZE = 2500
DEFAULT_CLOUD_PERCENTAGE = 20
DEFAULT_DATE_RANGE_MONTHS = 6
MIN_PIXELS_FOR_ANALYSIS = 10
MAX_PIXELS_SAMPLE = 5000
THUMBNAIL_SIZE = '512x512'
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
NDVI_THRESHOLD_WATER = -0.3
MIN_CLASS_PERCENTAGE = 1.0

# Caminhos dos arquivos do modelo treinado
MODEL_DIR = './results'
MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_ndvi_classifier.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'ndvi_classes.pkl')
SAMPLES_PATH = os.path.join(MODEL_DIR, 'ndvi_samples.pkl')

# Verificar se arquivos do modelo existem
MODEL_AVAILABLE = all([
    os.path.exists(MODEL_PATH),
    os.path.exists(CLASSES_PATH)
])

# Cores para visualização
COLORS = {
    'primary': '#8b0000',
    'primary_hover': '#a52a2a',
    'background': '#1e1e1e',
    'secondary_background': '#2d2d2d',
    'sidebar_background': '#1a1a1a',
    'success': '#00ff00',
    'warning': '#ffa500',
    'error': '#ff4444',
    'white': '#ffffff'
}

# Configuração da página
st.set_page_config(
    page_title="🌱 Análise NDVI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['background']};
        color: {COLORS['white']};
    }}

    .main .block-container {{
        background-color: {COLORS['secondary_background']};
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid {COLORS['primary']};
    }}

    .stSidebar {{
        background-color: {COLORS['sidebar_background']};
        border-right: 2px solid {COLORS['primary']};
    }}

    .stSidebar .sidebar-content {{
        background-color: {COLORS['sidebar_background']};
    }}

    h1, h2, h3 {{
        color: {COLORS['white']};
        border-bottom: 2px solid {COLORS['primary']};
        padding-bottom: 10px;
    }}

    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['primary_hover']};
    }}

    .stSelectbox, .stSlider, .stNumberInput, .stTextInput, .stTextArea {{
        background-color: {COLORS['secondary_background']};
        color: {COLORS['white']};
    }}

    .stAlert {{
        background-color: {COLORS['secondary_background']};
        border: 1px solid {COLORS['primary']};
    }}

    .metric-container {{
        background-color: {COLORS['secondary_background']};
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid {COLORS['primary']};
        margin: 0.5rem 0;
    }}

    .success-text {{
        color: {COLORS['success']};
    }}

    .warning-text {{
        color: {COLORS['warning']};
    }}

    .error-text {{
        color: {COLORS['error']};
    }}
</style>
""", unsafe_allow_html=True)

# ===================== INICIALIZAÇÃO EARTH ENGINE =====================

def initialize_earth_engine():
    """Inicializa o Earth Engine com autenticação - VERSÃO CORRIGIDA"""
    try:
        # Forçar inicialização com seu projeto específico
        ee.Initialize(project='ndvi-analysis-455514')
        print("✅ Earth Engine autenticado com projeto específico!")
        return True
    except Exception as e:
        try:
            print("🔐 Iniciando autenticação do Earth Engine...")
            print("Siga as instruções abaixo para autenticação:")
            
            # Fazer autenticação
            ee.Authenticate()
            
            # SEMPRE usar seu projeto específico
            ee.Initialize(project='ndvi-analysis-455514')
            print("✅ Earth Engine autenticado com projeto específico!")
            return True
            
        except Exception as e2:
            print("❌ Erro na inicialização do Earth Engine.")
            print("📋 Soluções possíveis:")
            print("1. Acesse: https://code.earthengine.google.com/")
            print("2. Faça login com sua conta Google")
            print("3. Aceite os termos de uso do Earth Engine")
            print("4. Tente executar novamente")
            
            # Mostrar erro na interface Streamlit
            st.error("❌ Erro ao inicializar Google Earth Engine")
            st.error(f"Detalhes: {str(e2)}")
            
            with st.expander("🔧 Como resolver este erro"):
                st.markdown("""
                **Reautentique com seu projeto:**
                1. Abra terminal/prompt
                2. Execute: `py -3.10 -c "import ee; ee.Authenticate(); ee.Initialize(project='ndvi-analysis-455514')"`
                3. Siga as instruções de autenticação
                4. Reinicie esta aplicação
                """)
            
            return False

# Tentar inicializar Earth Engine
EE_INITIALIZED = initialize_earth_engine()

if not EE_INITIALIZED:
    st.stop()  # Para execução se Earth Engine não funcionar

# ===================== CARREGAMENTO DO MODELO TREINADO =====================

@st.cache_resource
def load_trained_model():
    """Carrega o modelo CNN treinado e as classes NDVI"""
    if not MODEL_AVAILABLE:
        return None, None
    
    try:
        # Carregar modelo CNN
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Modelo carregado de: {MODEL_PATH}")
        
        # Carregar classes NDVI
        with open(CLASSES_PATH, 'rb') as f:
            ndvi_classes = pickle.load(f)
        logger.info(f"✅ Classes NDVI carregadas de: {CLASSES_PATH}")
        
        return model, ndvi_classes
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return None, None

# Carregar modelo e classes
TRAINED_MODEL, LOADED_NDVI_CLASSES = load_trained_model()

# ===================== CLASSES E ESTRUTURAS DE DADOS =====================

class NDVIClasses:
    """Classes de NDVI com seus limiares e cores - SINCRONIZADO COM TREINAMENTO"""
    
    # Usar classes carregadas do modelo se disponível, senão usar as MESMAS do treinamento
    if LOADED_NDVI_CLASSES:
        CLASSES = LOADED_NDVI_CLASSES
        st.success("✅ Usando classes NDVI do modelo treinado")
    else:
        # Classes IDÊNTICAS ao código de treinamento
        CLASSES = {
            0: {'nome': 'Solo exposto', 'limiar': [-1.0, 0.177], 'cor': 'red'},
            1: {'nome': 'Baixa', 'limiar': [0.177, 0.331], 'cor': 'orange'},
            2: {'nome': 'Média baixa', 'limiar': [0.331, 0.471], 'cor': 'yellow'},
            3: {'nome': 'Média', 'limiar': [0.471, 0.584], 'cor': 'yellowgreen'},
            4: {'nome': 'Média alta', 'limiar': [0.584, 0.7], 'cor': 'green'},
            5: {'nome': 'Alta', 'limiar': [0.7, 1.0], 'cor': 'darkgreen'}
        }
        st.warning("⚠️ Usando classes NDVI padrão (modelo não encontrado)")

    WATER_CLASS = 6
    WATER_COLOR = 'blue'
    WATER_NAME = 'Água'

# ===================== FUNÇÕES DE PROCESSAMENTO DE IMAGENS =====================

def calculate_ndvi(img: ee.Image) -> ee.Image:
    """Calcula NDVI (Normalized Difference Vegetation Index)"""
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')

def calculate_ndwi(img: ee.Image) -> ee.Image:
    """Calcula NDWI (Normalized Difference Water Index)"""
    return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

def calculate_mndwi(img: ee.Image) -> ee.Image:
    """Calcula MNDWI (Modified Normalized Difference Water Index)"""
    return img.normalizedDifference(['B3', 'B11']).rename('MNDWI')

def calculate_evi(img: ee.Image) -> ee.Image:
    """Calcula EVI (Enhanced Vegetation Index)"""
    return img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
        }
    ).rename('EVI')

def calculate_savi(img: ee.Image) -> ee.Image:
    """Calcula SAVI (Soil Adjusted Vegetation Index)"""
    return img.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * (1.5)',
        {
            'NIR': img.select('B8'),
            'RED': img.select('B4')
        }
    ).rename('SAVI')

# ===================== FUNÇÕES DE UTILIDADE =====================

def fetch_image_sync(url: str) -> Optional[Image.Image]:
    """Busca imagem de forma síncrona com retry"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                logger.warning(f"Tentativa {attempt + 1}: Status {response.status_code} para URL {url}")
        except requests.Timeout:
            logger.warning(f"Tentativa {attempt + 1}: Timeout ao buscar imagem")
        except Exception as e:
            logger.error(f"Tentativa {attempt + 1}: Erro ao buscar imagem: {e}")

        if attempt < MAX_RETRIES - 1:
            import time
            time.sleep(2 ** attempt)  # Backoff exponencial

    return None

def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, Optional[str]]:
    """Valida coordenadas geográficas"""
    try:
        lat = float(latitude)
        lon = float(longitude)

        if lat < -90 or lat > 90:
            return False, "Latitude deve estar entre -90 e 90"

        if lon < -180 or lon > 180:
            return False, "Longitude deve estar entre -180 e 180"

        return True, None
    except (ValueError, TypeError):
        return False, "Coordenadas devem ser valores numéricos"

def classify_ndvi_with_model(ndvi_values: np.ndarray) -> np.ndarray:
    """
    Classifica NDVI usando o modelo CNN treinado ou método de limiares
    SINCRONIZADO COM O CÓDIGO DE TREINAMENTO
    """
    
    if TRAINED_MODEL is not None:
        try:
            # Usar modelo CNN treinado
            st.info("🤖 Usando modelo CNN treinado para classificação")
            
            # Criar patches para predição (MESMO PATCH_SIZE DO TREINAMENTO: 256)
            patches = []
            
            for val in ndvi_values.flatten():
                # Criar patch sintético com variação ao redor do valor NDVI
                # MESMA LÓGICA DO TREINAMENTO
                patch = np.clip(
                    np.random.normal(loc=val, scale=0.05, size=(PATCH_SIZE, PATCH_SIZE)),
                    -1, 1
                )
                patches.append(patch)
            
            X = np.array(patches)[..., np.newaxis]
            
            # Fazer predição
            predictions = TRAINED_MODEL.predict(X, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            logger.info(f"✅ Classificação CNN concluída: {len(predicted_classes)} pixels")
            return predicted_classes
            
        except Exception as e:
            logger.error(f"❌ Erro na classificação CNN: {e}")
            st.warning("⚠️ Erro no modelo CNN, usando classificação por limiares")
            # Fallback para classificação por limiares
            return classify_ndvi_with_thresholds(ndvi_values)
    else:
        # Usar classificação por limiares
        st.info("📊 Usando classificação por limiares de NDVI")
        return classify_ndvi_with_thresholds(ndvi_values)

def classify_ndwi_mndwi_and_ndvi(ndvi, ndwi, mndwi):
    """
    Função IDÊNTICA ao código de treinamento para classificar NDVI, NDWI e MNDWI
    """
    classified = ee.Image(-1).rename('ndvi_class')
    for class_id in NDVIClasses.CLASSES:
        min_val, max_val = NDVIClasses.CLASSES[class_id]['limiar']
        mask = ndvi.gte(min_val).And(ndvi.lt(max_val))
        classified = classified.where(mask, class_id)

    # Classificar água com base tanto no NDWI quanto no MNDWI (valores positivos)
    water_mask = ndwi.gt(0).And(mndwi.gt(0))
    classified = classified.where(water_mask, 6)  # Classe 6: Água

    return classified

def classify_ndvi_with_thresholds(ndvi_values: np.ndarray) -> np.ndarray:
    """
    Classifica NDVI usando método de limiares tradicionais
    SINCRONIZADO COM CÓDIGO DE TREINAMENTO
    """
    
    # Classificar pixels usando OS MESMOS LIMIARES do treinamento
    classified = np.full(ndvi_values.shape[0], -1)

    for class_id, class_info in NDVIClasses.CLASSES.items():
        min_val, max_val = class_info['limiar']
        mask = (ndvi_values.flatten() >= min_val) & (ndvi_values.flatten() < max_val)
        classified[mask] = class_id

    # Identificar água usando MESMA LÓGICA do treinamento
    # No treinamento usa NDWI e MNDWI, aqui fazemos aproximação por NDVI baixo
    water_mask = ndvi_values.flatten() < NDVI_THRESHOLD_WATER
    classified[water_mask] = NDVIClasses.WATER_CLASS
    
    return classified

def calculate_health_score(class_counts: Dict[int, int]) -> Tuple[Optional[float], str]:
    """Calcula o índice de saúde da vegetação"""
    health_scores = {
        0: 0.0,    # Solo exposto
        1: 0.2,    # Baixa
        2: 0.4,    # Média baixa
        3: 0.6,    # Média
        4: 0.8,    # Média alta
        5: 1.0,    # Alta
        6: None    # Água
    }

    # Calcular pixels não-água
    non_water_pixels = sum(count for class_id, count in class_counts.items()
                          if class_id != NDVIClasses.WATER_CLASS)

    if non_water_pixels == 0:
        return None, "SEM VEGETAÇÃO"

    # Calcular média ponderada
    weighted_sum = sum(health_scores[class_id] * count
                      for class_id, count in class_counts.items()
                      if class_id != NDVIClasses.WATER_CLASS and class_id in health_scores)

    score = weighted_sum / non_water_pixels

    # Determinar categoria
    if score < 0.3:
        category = "CRÍTICA"
    elif score < 0.5:
        category = "BAIXA"
    elif score < 0.7:
        category = "MODERADA"
    else:
        category = "BOA/EXCELENTE"

    return score, category

# ===================== FUNÇÃO PRINCIPAL DE ANÁLISE =====================

def analyze_area(latitude: float, longitude: float, buffer_size: int = DEFAULT_BUFFER_SIZE,
                 date_range_months: int = DEFAULT_DATE_RANGE_MONTHS,
                 cloud_percentage: int = DEFAULT_CLOUD_PERCENTAGE,
                 min_pixels_for_analysis: int = MIN_PIXELS_FOR_ANALYSIS) -> Optional[Dict[str, Any]]:
    """Realiza análise completa de uma área a partir de coordenadas"""

    # Usar session state para armazenar resultados
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

    # Criar placeholders que persistem
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    with progress_placeholder.container():
        progress_bar = st.progress(0)

    with status_placeholder.container():
        status_text = st.empty()

    try:
        # Validar coordenadas
        valid, error_msg = validate_coordinates(latitude, longitude)
        if not valid:
            st.error(f"❌ {error_msg}")
            return None

        lat, lon = float(latitude), float(longitude)
        status_text.text(f"🔍 Analisando área centrada em {lat:.4f}, {lon:.4f}")
        progress_bar.progress(20)

        # Definir área de interesse
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(buffer_size)

        # Definir intervalo de datas
        current_date = datetime.utcnow().isoformat()
        end_date = ee.Date(current_date)
        start_date = end_date.advance(-date_range_months, 'month')

        status_text.text("📅 Buscando imagens de satélite...")
        progress_bar.progress(30)

        # Coletar imagens Sentinel-2
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)) \
            .filterBounds(area)

        collection_size = collection.size().getInfo()
        if collection_size == 0:
            st.error("❌ Nenhuma imagem disponível para esta região no período selecionado.")
            st.warning("⚠️ Tente aumentar o período ou o percentual de nuvens aceito.")
            return None

        st.success(f"✅ {collection_size} imagens encontradas")
        progress_bar.progress(40)

        # Criar composição
        best_images = collection.sort('CLOUDY_PIXEL_PERCENTAGE').limit(5)
        composite = best_images.median()

        # Calcular índices
        ndvi = calculate_ndvi(composite)
        ndwi = calculate_ndwi(composite)
        mndwi = calculate_mndwi(composite)
        evi = calculate_evi(composite)
        savi = calculate_savi(composite)

        progress_bar.progress(50)

        # Configurações de visualização
        rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
        ndvi_vis = {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}

        # Obter URLs das imagens
        status_text.text("🖼️ Gerando visualizações...")

        try:
            rgb_url = composite.visualize(**rgb_vis).getThumbURL({
                'region': area,
                'dimensions': THUMBNAIL_SIZE,
                'format': 'png'
            })

            ndvi_url = ndvi.visualize(**ndvi_vis).getThumbURL({
                'region': area,
                'dimensions': THUMBNAIL_SIZE,
                'format': 'png'
            })

            # Buscar imagens
            rgb_img = fetch_image_sync(rgb_url)
            ndvi_img = fetch_image_sync(ndvi_url)

            if rgb_img and ndvi_img:
                # Armazenar imagens no session state
                st.session_state.analysis_results['rgb_image'] = rgb_img
                st.session_state.analysis_results['ndvi_image'] = ndvi_img
            else:
                logger.warning("Falha ao buscar imagens de visualização")

        except Exception as e:
            logger.error(f"Erro ao gerar visualizações: {e}")
            st.warning("⚠️ Não foi possível gerar as visualizações")

        progress_bar.progress(60)

        # Baixar dados NDVI
        status_text.text("⏳ Processando dados NDVI...")

        try:
            # Adicionar valor padrão para pixels mascarados
            ndvi_masked = ndvi.unmask(-9999)

            # Usar sampleRegion
            ndvi_points = ndvi_masked.sample(
                region=area,
                scale=10,
                numPixels=MAX_PIXELS_SAMPLE,
                seed=42,
                dropNulls=True
            )

            count = ndvi_points.size().getInfo()

            if count < min_pixels_for_analysis:
                # Tentar método alternativo
                st.warning(f"⚠️ Poucos pontos válidos ({count}). Tentando método alternativo...")

                ndvi_stats = ndvi_masked.reduceRegion(
                    reducer=ee.Reducer.histogram().combine(ee.Reducer.mean(), None, True),
                    geometry=area,
                    scale=10,
                    maxPixels=1e9
                ).getInfo()

                ndvi_values = []
                if 'NDVI_histogram' in ndvi_stats and ndvi_stats['NDVI_histogram']:
                    hist = ndvi_stats['NDVI_histogram']
                    if 'bucketMeans' in hist and 'counts' in hist:
                        for value, count in zip(hist['bucketMeans'], hist['counts']):
                            if value != -9999:
                                ndvi_values.extend([value] * int(count))

                if len(ndvi_values) < min_pixels_for_analysis:
                    st.error("❌ Dados insuficientes para análise")
                    return None

                ndvi_array = np.array(ndvi_values).reshape(-1, 1)
            else:
                # Processar dados do sample
                ndvi_values = [feature['properties']['NDVI']
                             for feature in ndvi_points.getInfo()['features']
                             if feature['properties']['NDVI'] != -9999]

                ndvi_array = np.array(ndvi_values).reshape(-1, 1)

            st.success(f"✅ {len(ndvi_values)} pixels processados")

        except Exception as e:
            logger.error(f"Erro ao obter dados NDVI: {e}")
            st.error("❌ Erro ao processar dados NDVI")
            return None

        progress_bar.progress(70)

        # Armazenar dados NDVI no session state
        st.session_state.analysis_results['ndvi_data'] = ndvi_array

        # Classificação
        status_text.text("📊 Classificando vegetação...")

        # Usar modelo treinado ou classificação por limiares
        classified = classify_ndvi_with_model(ndvi_array)

        progress_bar.progress(80)

        # Calcular estatísticas
        valid_pixels = classified[classified >= 0]
        total_pixels = valid_pixels.size

        if total_pixels == 0:
            st.error("❌ Nenhum pixel válido encontrado")
            return None

        class_counts = {i: np.sum(valid_pixels == i)
                       for i in range(len(NDVIClasses.CLASSES) + 1)
                       if i in np.unique(valid_pixels)}

        # Calcular saúde
        health_score, health_category = calculate_health_score(class_counts)

        progress_bar.progress(90)

        # Preparar resultados
        class_results = []
        non_water_pixels = sum(count for class_id, count in class_counts.items()
                              if class_id != NDVIClasses.WATER_CLASS)

        for class_id in sorted(class_counts.keys()):
            if class_id == NDVIClasses.WATER_CLASS:
                percentage = (class_counts[class_id] / total_pixels) * 100
                class_results.append((NDVIClasses.WATER_NAME, percentage))
            elif class_id in NDVIClasses.CLASSES:
                if non_water_pixels > 0:
                    percentage = (class_counts[class_id] / non_water_pixels) * 100
                else:
                    percentage = 0
                class_results.append((NDVIClasses.CLASSES[class_id]['nome'], percentage))

        # Armazenar todos os resultados no session state
        st.session_state.analysis_results.update({
            'coordinates': (lat, lon),
            'buffer_size': buffer_size,
            'health_score': health_score,
            'health_category': health_category,
            'class_distribution': class_results,
            'class_counts': class_counts,
            'total_pixels': total_pixels,
            'date_range': {
                'start': start_date.format('YYYY-MM-dd').getInfo(),
                'end': end_date.format('YYYY-MM-dd').getInfo()
            },
            'images_used': collection_size,
            'analysis_complete': True
        })

        progress_bar.progress(100)
        status_text.text("✅ Análise completa!")

        # Limpar placeholders após pequeno delay
        import time
        time.sleep(1)
        progress_placeholder.empty()
        status_placeholder.empty()

        return st.session_state.analysis_results

    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        st.error(f"❌ Erro durante a análise: {str(e)}")
        return None

# ===================== FUNÇÃO DE EXIBIÇÃO DE RESULTADOS =====================

def display_results():
    """Exibe os resultados da análise armazenados no session state"""

    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results.get('analysis_complete'):
        return

    results = st.session_state.analysis_results

    # Exibir imagens RGB e NDVI
    if 'rgb_image' in results and 'ndvi_image' in results:
        st.subheader("🖼️ Imagem RGB e NDVI da área selecionada:")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Imagem RGB**")
            st.image(results['rgb_image'], use_container_width=True)

        with col2:
            st.markdown("**NDVI**")
            st.image(results['ndvi_image'], use_container_width=True)

    # Exibir histograma NDVI
    if 'ndvi_data' in results:
        st.subheader("📊 Distribuição de valores NDVI na área")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS['secondary_background'])
        ax.set_facecolor(COLORS['secondary_background'])

        ax.hist(results['ndvi_data'].flatten(), bins=50, range=(-1, 1),
                color=COLORS['primary'], alpha=0.7, edgecolor='white')
        ax.set_title('Distribuição de valores NDVI na área', color='white', fontsize=16)
        ax.set_xlabel('Valor NDVI', color='white')
        ax.set_ylabel('Frequência', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='white')

        # Adicionar linhas verticais para limiares das classes
        for class_info in NDVIClasses.CLASSES.values():
            ax.axvline(x=class_info['limiar'][0], color=class_info['cor'],
                      linestyle='--', alpha=0.5, linewidth=2)

        st.pyplot(fig)
        plt.close()

    # Exibir estatísticas
    st.subheader("📊 Estatísticas da área:")
    st.write(f"**Total de pixels válidos:** {results.get('total_pixels', 0):,}")
    st.write(f"**Imagens utilizadas:** {results.get('images_used', 0)}")
    st.write(f"**Período:** {results['date_range']['start']} a {results['date_range']['end']}")

    # Exibir classificação
    if results.get('class_distribution'):
        st.subheader("📋 Resumo da classificação:")
        for class_name, percentage in results['class_distribution']:
            if percentage > 0:
                st.write(f"- **{class_name}:** {percentage:.1f}%")

    # Exibir índice de saúde
    if results.get('health_score') is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🌿 Índice de Saúde da Vegetação",
                value=f"{results['health_score']:.2f}",
                help="Escala de 0 a 1, onde 1 representa vegetação mais saudável"
            )

        with col2:
            health_category = results.get('health_category', 'INDEFINIDA')
            if health_category == "CRÍTICA":
                st.markdown(f'<p class="error-text">❌ Saúde: {health_category}</p>',
                           unsafe_allow_html=True)
            elif health_category == "BAIXA":
                st.markdown(f'<p class="warning-text">⚠️ Saúde: {health_category}</p>',
                           unsafe_allow_html=True)
            elif health_category == "MODERADA":
                st.markdown(f'<p class="warning-text">✓ Saúde: {health_category}</p>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="success-text">✅ Saúde: {health_category}</p>',
                           unsafe_allow_html=True)

        # Sugestões baseadas na saúde
        if results['health_score'] < 0.3:
            st.warning("**Sugestão:** Esta área possui predominantemente solo exposto e vegetação de baixa densidade. Recomenda-se investigar causas de degradação e considerar estratégias de recuperação.")
        elif results['health_score'] < 0.5:
            st.info("**Sugestão:** A área apresenta sinais de estresse. Pode ser necessário monitoramento adicional.")
        elif results['health_score'] < 0.7:
            st.info("**Sugestão:** Vegetação em condição média. Monitoramento regular é recomendado.")
        else:
            st.success("**Sugestão:** A vegetação parece estar em boas condições.")

    # Gráfico de pizza
    if results.get('class_counts') and results.get('total_pixels', 0) > 0:
        st.subheader("📊 Distribuição das Classes na Área")

        labels = []
        sizes = []
        colors_pie = []

        for class_id in sorted(results['class_counts'].keys()):
            if class_id == NDVIClasses.WATER_CLASS:
                label = NDVIClasses.WATER_NAME
                color = NDVIClasses.WATER_COLOR
            elif class_id in NDVIClasses.CLASSES:
                label = NDVIClasses.CLASSES[class_id]['nome']
                color = NDVIClasses.CLASSES[class_id]['cor']
            else:
                continue

            percentage = (results['class_counts'][class_id] / results['total_pixels']) * 100
            if percentage >= MIN_CLASS_PERCENTAGE:
                labels.append(f"{label}\n{percentage:.1f}%")
                sizes.append(results['class_counts'][class_id])
                colors_pie.append(color)

        if len(sizes) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(COLORS['secondary_background'])

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                               autopct='%1.1f%%', shadow=True,
                                               startangle=90, textprops={'color': 'white'})

            # Melhorar legibilidade dos textos
            for text in texts:
                text.set_color('white')
                text.set_fontsize(10)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.set_title('Distribuição das Classes de Vegetação', color='white', fontsize=16, pad=20)
            st.pyplot(fig)
            plt.close()

    # Mapa interativo
    if results.get('coordinates'):
        st.subheader("🗺️ Localização da Área Analisada")

        lat, lon = results['coordinates']
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles='OpenStreetMap')

        # Adicionar círculo da área analisada
        popup_html = f"""
        <div style='font-family: Arial; font-size: 14px;'>
            <b>Área de Análise</b><br>
            <b>Coordenadas:</b> {lat:.4f}, {lon:.4f}<br>
            <b>Buffer:</b> {results.get('buffer_size', DEFAULT_BUFFER_SIZE)}m<br>
            <b>Saúde:</b> {results.get('health_score', 'N/A'):.2f}<br>
            <b>Categoria:</b> {results.get('health_category', 'INDEFINIDA')}
        </div>
        """

        folium.Circle(
            location=[lat, lon],
            radius=results.get('buffer_size', DEFAULT_BUFFER_SIZE),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip="Clique para mais informações",
            color=COLORS['primary'],
            fill=True,
            fillColor=COLORS['primary'],
            fillOpacity=0.2,
            weight=3
        ).add_to(m)

        # Adicionar marcador central
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip="Centro da análise",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)

        # Exibir mapa
        st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])

# ===================== INTERFACE PRINCIPAL =====================

def main():
    """Função principal do aplicativo"""

    st.title("🌱 Sistema de Análise de Vegetação por Imagens de Satélite")
    st.markdown("**Análise avançada de NDVI usando imagens Sentinel-2 para avaliação da saúde da vegetação**")

    # Inicializar variáveis de sessão
    if 'latitude' not in st.session_state:
        st.session_state.latitude = -15.8
    if 'longitude' not in st.session_state:
        st.session_state.longitude = -47.9

    # Sidebar com parâmetros
    with st.sidebar:
        st.header("⚙️ Parâmetros de Análise")

        # Coordenadas
        st.subheader("📍 Coordenadas")

        # Input de coordenadas
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.latitude,
                step=0.0001,
                format="%.6f",
                help="Latitude em graus decimais"
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.longitude,
                step=0.0001,
                format="%.6f",
                help="Longitude em graus decimais"
            )

        # Atualizar session state
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude

        # Parâmetros de análise
        st.subheader("🔧 Configurações")

        buffer_size = st.slider(
            "Buffer (metros)",
            min_value=500,
            max_value=5000,
            value=DEFAULT_BUFFER_SIZE,
            step=100,
            help="Raio da área circular de análise em metros"
        )

        cloud_percentage = st.slider(
            "Máximo de nuvens (%)",
            min_value=0,
            max_value=100,
            value=DEFAULT_CLOUD_PERCENTAGE,
            step=5,
            help="Percentual máximo de cobertura de nuvens nas imagens"
        )

        date_range_months = st.slider(
            "Período de busca (meses)",
            min_value=1,
            max_value=24,
            value=DEFAULT_DATE_RANGE_MONTHS,
            step=1,
            help="Período retroativo para busca de imagens"
        )

        # Sidebar com informações do modelo
        st.subheader("🤖 Status do Modelo")
        
        if MODEL_AVAILABLE:
            st.success("✅ Modelo CNN carregado")
            st.info(f"📁 Modelo: {MODEL_PATH}")
            
            # Mostrar informações do modelo se disponível
            if TRAINED_MODEL:
                total_params = TRAINED_MODEL.count_params()
                st.metric("Parâmetros do modelo", f"{total_params:,}")
                
                # Verificar se existe arquivo de amostras
                if os.path.exists(SAMPLES_PATH):
                    try:
                        with open(SAMPLES_PATH, 'rb') as f:
                            samples_data = pickle.load(f)
                        st.metric("Amostras de treinamento", len(samples_data.get('ndvi_vals', [])))
                    except:
                        pass
        else:
            st.warning("⚠️ Modelo CNN não encontrado")
            st.info("📊 Usando classificação por limiares")
            
            with st.expander("Como usar o modelo treinado"):
                st.markdown("""
                **Para usar o modelo CNN treinado:**
                
                1. Execute primeiro o script `treinamento.py`
                2. Aguarde o treinamento completar
                3. Verifique se os arquivos foram salvos em `./results/`:
                   - `cnn_ndvi_classifier.h5`
                   - `ndvi_classes.pkl`
                   - `ndvi_samples.pkl`
                4. Reinicie esta aplicação Streamlit
                
                **Vantagens do modelo CNN:**
                - Maior precisão na classificação
                - Melhor em paisagens complexas
                - Aprende padrões espaciais
                """)

        st.markdown("---")

        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas"):
            min_pixels_for_analysis = st.number_input(
                "Pixels mínimos para análise",
                min_value=1,
                max_value=1000,
                value=MIN_PIXELS_FOR_ANALYSIS,
                step=10,
                help="Número mínimo de pixels válidos para realizar a análise"
            )

            st.info("""
            **Dicas para melhores resultados:**
            - Use buffer maior para áreas rurais
            - Aumente o período em regiões com muitas nuvens
            - Reduza pixels mínimos para áreas pequenas
            """)

        st.markdown("---")

        # Botão de análise
        analyze_button = st.button(
            "🔍 Analisar Área",
            type="primary",
            use_container_width=True,
            help="Iniciar análise da área selecionada"
        )

        # Executar análise
        if analyze_button:
            if latitude and longitude:
                # Limpar resultados anteriores
                if 'analysis_results' in st.session_state:
                    st.session_state.analysis_results = {}

                # Realizar análise
                result = analyze_area(
                    latitude=latitude,
                    longitude=longitude,
                    buffer_size=buffer_size,
                    cloud_percentage=cloud_percentage,
                    date_range_months=date_range_months,
                    min_pixels_for_analysis=min_pixels_for_analysis
                )

                if result:
                    st.success("🎉 Análise concluída com sucesso!")
            else:
                st.error("❌ Por favor, insira coordenadas válidas.")

        # Informações e exemplos
        st.markdown("---")

        # Exemplos de coordenadas
        st.subheader("📍 Locais de Exemplo")

        examples = {
            "Brasília, DF": (-15.7801, -47.9292),
            "São Paulo, SP": (-23.5505, -46.6333),
            "Rio de Janeiro, RJ": (-22.9068, -43.1729),
            "Manaus, AM": (-3.1190, -60.0217),
            "Pantanal, MT": (-16.3500, -56.8000),
            "Chapada Diamantina, BA": (-12.9000, -41.4000),
            "Fernando de Noronha, PE": (-3.8400, -32.4100)
        }

        for city, (lat, lon) in examples.items():
            if st.button(f"📍 {city}", key=f"btn_{city}", use_container_width=True):
                st.session_state.latitude = lat
                st.session_state.longitude = lon
                st.rerun()

        st.markdown("---")

        # Informações sobre o sistema
        st.subheader("ℹ️ Sobre o Sistema")

        with st.expander("📊 Índices de Vegetação"):
            st.markdown("""
            **NDVI (Normalized Difference Vegetation Index)**
            - Índice mais usado para análise de vegetação
            - Varia de -1 a 1
            - Valores altos indicam vegetação saudável

            **NDWI (Normalized Difference Water Index)**
            - Identifica corpos d'água
            - Útil para delimitar áreas úmidas

            **EVI (Enhanced Vegetation Index)**
            - Versão melhorada do NDVI
            - Melhor em áreas com vegetação densa

            **SAVI (Soil Adjusted Vegetation Index)**
            - Ajustado para influência do solo
            - Melhor em áreas com vegetação esparsa
            """)

        with st.expander("🛰️ Sobre o Sentinel-2"):
            st.markdown("""
            **Sentinel-2** é uma missão de observação da Terra da ESA:
            - Resolução: 10m (bandas visíveis)
            - Revisita: 5 dias
            - 13 bandas espectrais
            - Cobertura global
            - Dados gratuitos e abertos
            """)

        with st.expander("📈 Interpretação dos Resultados"):
            st.markdown("""
            **Índice de Saúde da Vegetação:**
            - 0.0 - 0.3: Crítica (solo exposto/vegetação morta)
            - 0.3 - 0.5: Baixa (vegetação estressada)
            - 0.5 - 0.7: Moderada (vegetação média)
            - 0.7 - 1.0: Boa/Excelente (vegetação saudável)

            **Classes de Vegetação:**
            - Solo exposto: NDVI < 0.2
            - Vegetação baixa: 0.2 ≤ NDVI < 0.4
            - Vegetação média: 0.4 ≤ NDVI < 0.6
            - Vegetação alta: NDVI ≥ 0.6
            - Água: NDVI < -0.3
            """)

        # Rodapé
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888;'>
                <small>
                    Desenvolvido usando Google Earth Engine<br>
                    Dados: Copernicus Sentinel-2
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Área principal - Exibir resultados
    display_results()

    # Se não houver análise, mostrar instruções
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results.get('analysis_complete'):
        st.info("""
        👋 **Bem-vindo ao Sistema de Análise de Vegetação!**

        Este sistema utiliza imagens de satélite Sentinel-2 para analisar a saúde da vegetação
        em qualquer local do mundo.

        **Como usar:**
        1. Insira as coordenadas do local desejado na barra lateral
        2. Ajuste os parâmetros de análise conforme necessário
        3. Clique em "🔍 Analisar Área" para iniciar

        **O que você receberá:**
        - Imagens RGB e NDVI da área
        - Classificação detalhada da vegetação
        - Índice de saúde da vegetação
        - Estatísticas e gráficos
        - Mapa interativo da região

        💡 **Dica:** Use os exemplos de coordenadas na barra lateral para começar!
        """)

# ===================== EXECUÇÃO =====================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erro fatal na aplicação: {e}")
        st.error(f"❌ Erro fatal: {str(e)}")
        st.info("🔄 Por favor, recarregue a página para tentar novamente.")