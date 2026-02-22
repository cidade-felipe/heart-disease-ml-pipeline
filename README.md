# Heart Disease ML Pipeline

<div align="center">

![Status](https://img.shields.io/badge/status-ativo-success)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Task](https://img.shields.io/badge/ml-binary%20classification-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Pipeline completo de ciência de dados para predição de risco de doença cardíaca.

</div>

## Visão Geral

Este repositório implementa um fluxo end-to-end de machine learning para classificação binária da variável `HeartDisease`, cobrindo:

- exploração e tratamento dos dados
- pré-processamento e engenharia de atributos
- treinamento de múltiplos modelos
- comparação por acurácia, acertos e validação cruzada

## Objetivo

Comparar algoritmos clássicos e de boosting para identificar a melhor combinação entre desempenho e estabilidade em um problema real de saúde.

## Dataset

- Base original: `data/csv/heart.csv`
- Base tratada: `data/csv/processed/heart_tratado.csv`
- Registros: **917**
- Colunas: **12** (11 features + 1 alvo)
- Alvo: `HeartDisease` (`0` = ausência, `1` = presença)

### Features

`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.

## Arquitetura do Projeto

### Mapa rápido

| Bloco                     | Função                                         |
| ------------------------- | ------------------------------------------------ |
| `data/csv/`             | dados brutos e versão tratada                   |
| `notebooks/`            | EDA, pré-processamento e classificação        |
| `figures/`              | gráficos de comparação dos modelos            |
| `pkl/`                  | artefatos serializados                           |
| `comparacao_red_dim.py` | script auxiliar de redução de dimensionalidade |

### Estrutura de pastas

```text
heart-disease-ml-pipeline/
|-- data/
|   |-- csv/
|       |-- heart.csv
|       |-- processed/
|           |-- heart_tratado.csv
|-- figures/
|   |-- train_test_accuracy.png
|   |-- cv_accuracy.png
|   |-- model_hits_test.png
|-- notebooks/
|   |-- exploracao_analise_e_tratamento.ipynb
|   |-- pre_processamento_reducao_dimensionalidade.ipynb
|   |-- classificacao.ipynb
|   |-- catboost_info/
|-- pkl/
|-- comparacao_red_dim.py
|-- requirements.txt
|-- README.md
```

## Fluxo dos Notebooks

### `notebooks/exploracao_analise_e_tratamento.ipynb`

- leitura e inspeção inicial do dataset
- estatística descritiva e análise de distribuição
- tratamento de inconsistências (ex.: zeros em variáveis clínicas)
- consolidação da base para modelagem

### `notebooks/pre_processamento_reducao_dimensionalidade.ipynb`

- codificação de variáveis categóricas
- separação entre previsores e alvo
- escalonamento e normalização
- expansão com `OneHotEncoder`
- experimentos de redução de dimensionalidade

### `notebooks/classificacao.ipynb`

- split treino/teste (`70/30`, `random_state=12`)
- treinamento de modelos supervisionados
- avaliação com métricas e matrizes de confusão
- validação cruzada com `KFold (30)`
- geração dos gráficos finais em `figures/`

## Modelos Avaliados

- Naive Bayes
- SVC
- Regressão Logística
- KNN
- Árvore de Decisão
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Resultados

Fonte dos valores: `figures/train_test_accuracy.png`, `figures/model_hits_test.png` e `figures/cv_accuracy.png`.

### Acurácia e Acertos no Teste

| Modelo                | Acurácia Treino/Teste (%) | Acertos no Teste |
| :-------------------- | -------------------------: | ---------------: |
| Naive Bayes           |                      84.78 |              234 |
| SVC                   |                      85.61 |              238 |
| Regressão Logística |                      85.83 |              238 |
| KNN                   |                      85.83 |              234 |
| Árvore de Decisão   |                      81.13 |              223 |
| Random Forest         |                      86.06 |              234 |
| XGBoost               |                      87.04 |              237 |
| LightGBM              |                      86.84 |              234 |
| **CatBoost**    |            **86.96** |    **240** |

### Destaques

- Melhor acurácia: **CatBoost (87.70%)**
- Maior número de acertos no teste: **CatBoost (240)**
- Modelos de boosting (XGBoost/LightGBM/CatBoost) dominaram o topo do ranking

## Gráficos

- `figures/train_test_accuracy.png`: comparação de acurácia dos modelos
- `figures/cv_accuracy.png`: acurácia média em validação cruzada
- `figures/model_hits_test.png`: número total de acertos no conjunto de teste

## Como Executar

### 1. Clonar o projeto

```bash
git clone https://github.com/cidade-felipe/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Criar e ativar ambiente virtual

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar os notebooks

```bash
jupyter notebook
```

Ordem sugerida:

1. `notebooks/exploracao_analise_e_tratamento.ipynb`
2. `notebooks/pre_processamento_reducao_dimensionalidade.ipynb`
3. `notebooks/classificacao.ipynb`

## Próximos Passos

- consolidar o pipeline em scripts modulares
- adicionar rastreamento de experimentos (MLflow)
- ampliar métricas: ROC-AUC, PR-AUC e calibração
- empacotar o melhor modelo em API de inferência

## Licença

Distribuído sob licença MIT. Consulte `LICENSE`.

## Autor

- Felipe Cidade Soares
- Linkedin: [https://www.linkedin.com/in/cidadefelipe/](https://www.linkedin.com/in/cidadefelipe/)
