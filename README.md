# Sistema Automatizado para Monitoramento da Saúde da Vegetação 🌱

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Sistema integrado para classificação de cobertura vegetal utilizando índices espectrais derivados de imagens Sentinel-2, processamento automatizado via Google Earth Engine (GEE) e classificação baseada em redes neurais convolucionais (CNN).

## 📖 Visão Geral

Este projeto implementa uma metodologia completa para análise da saúde da vegetação, combinando sensoriamento remoto, aprendizado profundo e computação em nuvem. O sistema alcança **98% de acurácia** na classificação de vegetação, superando métodos tradicionais como Random Forest (87%) e SVM (82%).

### 🎯 Características Principais

- **🤖 IA Avançada**: Rede neural convolucional com 98% de acurácia
- **🛰️ Dados Globais**: Integração com Google Earth Engine e imagens Sentinel-2
- **🌍 Escala Global**: Treinamento com 13 regiões representativas mundiais
- **📊 Interface Interativa**: Dashboard web com Streamlit
- **⚡ GPU Universal**: Suporte automático para NVIDIA (CUDA) e AMD (DirectML)
- **📈 Análise Completa**: Múltiplos índices espectrais (NDVI, NDWI, EVI, SAVI)

## 🚀 Demonstração

### Interface Web
![Sistema Web](https://via.placeholder.com/800x400/1e1e1e/ffffff?text=Sistema+Web+Interativo)

### Resultados de Classificação
![Resultados](https://via.placeholder.com/800x300/2d2d2d/ffffff?text=Classificação+de+Vegetação)

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| **Acurácia Geral** | 98.0% |
| **Precisão Média** | 97.8% |
| **Recall Médio** | 97.9% |
| **Classes Detectadas** | 6 + Água |
| **Regiões de Treinamento** | 13 globais |

### 🏆 Comparação com Literatura

| Método | Acurácia | Referência |
|--------|----------|------------|
| **Nossa CNN** | **98.0%** | Este trabalho |
| Random Forest | 87.0% | Maxwell et al. (2018) |
| SVM | 82.0% | Maxwell et al. (2018) |

## 🛠️ Instalação

### Pré-requisitos

- Python 3.10+ (recomendado)
- Conta Google Earth Engine ([registrar aqui](https://signup.earthengine.google.com/))
- 8GB+ RAM (16GB recomendado para treinamento)

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ndvi-vegetation-health.git
cd ndvi-vegetation-health

# Instale dependências básicas
pip install -r requirements.txt

# Para GPU NVIDIA (opcional)
pip install tensorflow[and-cuda]

# Para GPU AMD (opcional)
pip install tensorflow-directml
```

### Dependências Completas

```bash
# Instalar todas as dependências
pip install streamlit earthengine-api tensorflow matplotlib seaborn scikit-learn pillow requests numpy pandas folium streamlit-folium
```

## 🔧 Configuração

### 1. Autenticação Google Earth Engine

```bash
# Autenticar Earth Engine
python -c "import ee; ee.Authenticate(); ee.Initialize(project='seu-projeto-id')"
```

### 2. Verificar Instalação

```bash
# Testar configuração
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 📖 Uso

### Treinamento do Modelo

```bash
# Executar treinamento completo
python treinamento.py
```

**Saída esperada:**
- Modelo treinado: `./results/cnn_ndvi_classifier.h5`
- Classes NDVI: `./results/ndvi_classes.pkl`
- Relatórios: `./results/classification_report.txt`
- Gráficos: `./results/training_curves.png`

### Interface Web

```bash
# Iniciar aplicação Streamlit
streamlit run app.py
```

Acesse: `http://localhost:8501`

### Modo Offline (Teste sem Internet)

```bash
# Testar modelo sem Earth Engine
streamlit run app_offline.py
```

## 🎮 Exemplo de Uso

```python
import ee
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo treinado
model = load_model('./results/cnn_ndvi_classifier.h5')

# Analisar área específica
latitude, longitude = -15.7801, -47.9292  # Brasília
buffer_size = 2500  # metros

# O sistema fará automaticamente:
# 1. Busca imagens Sentinel-2
# 2. Calcula índices espectrais
# 3. Classifica com CNN
# 4. Gera relatório de saúde
```

## 📋 Estrutura do Projeto

```
├── 📄 app.py                      # Interface Streamlit principal
├── 📄 app_offline.py              # Versão offline para testes
├── 📄 treinamento.py             # Script de treinamento
├── 📄 requirements.txt           # Dependências Python
├── 📁 results/                   # Resultados do treinamento
│   ├── 🤖 cnn_ndvi_classifier.h5 # Modelo treinado
│   ├── 📊 ndvi_classes.pkl       # Classes de vegetação
│   ├── 📈 training_curves.png    # Curvas de treinamento
│   └── 📋 classification_report.txt # Relatório detalhado
├── 📁 docs/                      # Documentação
│   ├── 📝 artigo-ptbr.docx       # Artigo científico (PT-BR)
│   └── 📝 artigo-en.docx         # Artigo científico (EN)
└── 📄 README.md                  # Este arquivo
```

## 🌍 Regiões de Estudo

O modelo foi treinado com dados de **13 regiões globalmente distribuídas**:

| Região | Coordenadas | Bioma |
|--------|-------------|-------|
| 🇧🇷 Cerrado | -48.0, -16.0 | Savana tropical |
| 🇧🇷 Amazônia | -60.0, -3.0 | Floresta tropical |
| 🇧🇷 Caatinga | -39.0, -9.0 | Floresta seca |
| 🇧🇷 Pantanal | -57.0, -17.0 | Zona úmida |
| 🇧🇷 Mata Atlântica | -46.0, -23.0 | Floresta costeira |
| 🇧🇷 Pampa | -53.0, -31.0 | Pradaria |
| 🇧🇷 Área Urbana (RJ) | -43.3, -22.95 | Urbano |
| 🇺🇸 Floresta Temperada | -123.0, 49.0 | Floresta temperada |
| 🇱🇾 Deserto Saara | 23.0, 19.0 | Deserto |
| 🇮🇩 Floresta Tropical | 100.0, 0.5 | Floresta tropical |
| 🇰🇪 Savana Africana | 30.0, -2.0 | Savana |
| 🇦🇺 Semiárido | 135.0, -33.0 | Semiárido |
| 🇳🇱 Zona Agrícola | 5.0, 52.0 | Agricultura |

## 📊 Classes de Vegetação

| Classe | NDVI Range | Descrição | Cor |
|--------|------------|-----------|-----|
| **Solo Exposto** | [-1.0, 0.177] | Áreas sem vegetação | 🔴 Vermelho |
| **Vegetação Baixa** | [0.177, 0.331] | Vegetação esparsa | 🟠 Laranja |
| **Vegetação Média Baixa** | [0.331, 0.471] | Vegetação moderada | 🟡 Amarelo |
| **Vegetação Média** | [0.471, 0.584] | Vegetação estabelecida | 🟢 Verde claro |
| **Vegetação Média Alta** | [0.584, 0.7] | Vegetação densa | 🟢 Verde |
| **Vegetação Alta** | [0.7, 1.0] | Vegetação muito densa | 🟢 Verde escuro |
| **Água** | NDWI & MNDWI > 0 | Corpos d'água | 🔵 Azul |

## 🎯 Aplicações

### 🌿 Monitoramento Ambiental
- Detecção de desmatamento em tempo real
- Monitoramento de recuperação de áreas degradadas
- Análise de impacto de mudanças climáticas

### 🚜 Agricultura de Precisão
- Avaliação da saúde de cultivos
- Otimização de irrigação e fertilização
- Previsão de produtividade

### 🏙️ Planejamento Urbano
- Análise de cobertura vegetal urbana
- Planejamento de áreas verdes
- Estudos de ilha de calor urbana

### 🔬 Pesquisa Científica
- Estudos de biodiversidade
- Análise de mudanças sazonais
- Validação de modelos climáticos

## 🔬 Metodologia Científica

### Arquitetura da CNN

```
Input (256x256x1) → Conv2D(32) → MaxPool → 
Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → 
Flatten → Dense(128) → Dropout → Dense(Classes)
```

### Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| **Patch Size** | 256x256 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Epochs** | 50 |
| **Dropout** | 0.3 |

### Índices Espectrais

- **NDVI**: `(B8 - B4) / (B8 + B4)` - Vegetação
- **NDWI**: `(B3 - B8) / (B3 + B8)` - Água
- **MNDWI**: `(B3 - B11) / (B3 + B11)` - Água modificado
- **EVI**: `2.5 * ((B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1))` - Vegetação melhorada
- **SAVI**: `((B8 - B4) / (B8 + B4 + 0.5)) * 1.5` - Solo ajustado

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, siga estas diretrizes:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### 📋 Tipos de Contribuições

- 🐛 Correção de bugs
- ✨ Novas funcionalidades
- 📚 Melhoria da documentação
- 🎨 Melhorias de interface
- 🚀 Otimizações de performance
- 🧪 Adição de testes

## 📝 Publicações

Este trabalho foi desenvolvido como parte de pesquisa científica. Publicações relacionadas:

- **Artigo completo**: "Sistema Automatizado para Monitoramento da Saúde da Vegetação com NDVI e Redes Neurais Convolucionais"
- **Conferência**: Em submissão para revista internacional
- **Dataset**: Disponível mediante solicitação

### 📊 Citação

Se você usar este sistema em sua pesquisa, por favor cite:

```bibtex
@software{ndvi_vegetation_health_2025,
  title={Sistema Automatizado para Monitoramento da Saúde da Vegetação},
  author={[Seu Nome]},
  year={2025},
  url={https://github.com/seu-usuario/ndvi-vegetation-health},
  note={Sistema integrado de classificação de vegetação com CNN}
}
```

## 🛡️ Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Gabriel Maiolli** - *Desenvolvimento principal* - [@GMaiolli](https://github.com/GMaiolli)

## 🙏 Agradecimentos

- **Google Earth Engine** - Pela plataforma de processamento de dados geoespaciais
- **ESA Copernicus** - Pelos dados Sentinel-2 gratuitos e abertos
- **TensorFlow Team** - Pelo framework de deep learning
- **Streamlit** - Pela plataforma de desenvolvimento de aplicações web

## 📞 Suporte

- 📧 **Email**: maiolligabriel@gmail.com

## 🔗 Links Úteis

- [Google Earth Engine](https://earthengine.google.com/) - Plataforma de dados geoespaciais
- [Sentinel-2 Data](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) - Informações sobre o satélite
- [TensorFlow](https://tensorflow.org/) - Framework de machine learning
- [Streamlit](https://streamlit.io/) - Framework de aplicações web

---

<div align="center">

**🌱 Contribuindo para um monitoramento ambiental mais inteligente e acessível**

[![Star this repo](https://img.shields.io/github/stars/seu-usuario/ndvi-vegetation-health?style=social)](https://github.com/seu-usuario/ndvi-vegetation-health)
[![Fork this repo](https://img.shields.io/github/forks/seu-usuario/ndvi-vegetation-health?style=social)](https://github.com/seu-usuario/ndvi-vegetation-health/fork)

</div>
