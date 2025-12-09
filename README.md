# Heart Disease ML Pipeline

Pipeline de machine learning para análise e predição de risco de doença cardíaca, passando por todo o fluxo de trabalho de ciência de dados, da preparação dos dados até a avaliação e comparação de modelos.

<div align="center">

![Status](https://img.shields.io/badge/status-estável-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.x-blue)
![ML](https://img.shields.io/badge/machine%20learning-classificação-red)

</div>

---

## 🎯 Objetivo

Este projeto foi criado para estudar um problema real de saúde usando machine learning, praticando o pipeline completo de ciência de dados.  
A ideia é sair do clássico “rodar um modelo” e ir além, incluindo análise exploratória, preparação dos dados, comparação de algoritmos e visualização dos resultados.

O foco não é só acertar o maior número de previsões, mas entender o comportamento dos dados e dos modelos.

---

## 🧠 O que o projeto faz

De forma geral, o pipeline segue estes passos:

1. Carrega e organiza o dataset de doença cardíaca  
2. Faz limpeza e preparação dos dados  
3. Explora as variáveis com estatísticas e gráficos  
4. Cria, treina e avalia modelos de classificação  
5. Compara o desempenho dos modelos  
6. Gera gráficos para interpretar os resultados

Dependendo da versão do projeto, os modelos podem incluir, por exemplo:

- Regressão logística  
- KNN  
- Random Forest  
- Outros classificadores do scikit-learn

---

## 📊 Dataset

O projeto utiliza um dataset clássico de doença cardíaca, com variáveis clínicas e alvo binário indicando presença ou ausência de doença.  
Exemplos de tipos de atributos presentes:

- Idade  
- Pressão arterial em repouso  
- Colesterol  
- Frequência cardíaca máxima  
- Tipo de dor no peito  
- Outras variáveis clínicas relevantes

Se você quiser trocar o dataset por outro, basta manter a mesma ideia geral:  
um conjunto de atributos numéricos/categóricos e uma coluna alvo binária.

---

## 🛠️ Tecnologias utilizadas

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib e, se usado, Seaborn  
- Jupyter Notebook

---

## 🗂️ Estrutura do projeto

A estrutura pode variar, mas um formato típico para este tipo de pipeline é algo assim:

```text
heart-disease-ml-pipeline/
├── data/
│   └── csv/
│       └── heart.csv
├── figures/
├── notebooks/
│   ├── catboost_info/
│   ├── analise_tratamento.ipynb
│   ├── classificacao.ipynb
│   ├── exploracao_analise.ipynb
│   ├── pre_processamento_reducao_dimensionalidade.ipynb
│   └── tratamento_classificacao.ipynb
├── pkl/
│   └── heart.pkl
├── venv/
├── .gitignore
├── comparacao_red_dim.py
├── LICENSE
├── README.md
└── requirements.txt
````


---

## 📦 Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/cidade-felipe/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Criar ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# ou
.\.venv\Scripts\activate    # Windows
```

### 3. Instalar dependências

Se existir um arquivo `requirements.txt`, execute:

```bash
pip install -r requirements.txt
```

Caso contrário, instale manualmente as principais bibliotecas:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Rodar o notebook

```bash
jupyter notebook
```

Abra o notebook principal (por exemplo, `01_heart_disease_pipeline.ipynb`) e execute as células na ordem.

---

## 📈 Resultados e métricas

Os resultados típicos incluem:

* Acurácia em treino e teste
* Matriz de confusão
* Outras métricas relevantes, como precisão, revocação e F1-score
* Gráficos comparando o desempenho dos modelos

Você pode adicionar aqui:

* Imagens de gráficos de acurácia
* Prints da matriz de confusão
* Qual modelo teve melhor equilíbrio entre acerto e generalização

Exemplo de trecho descritivo que você pode completar depois:

> Nos testes realizados, os modelos X e Y apresentaram desempenho semelhante,
> com destaque para o modelo Z, que manteve boa acurácia em teste e bom equilíbrio entre classes.

---

## 🚀 Próximos passos e melhorias

Algumas ideias de evolução do projeto:

* Adicionar validação cruzada mais robusta
* Testar técnicas de balanceamento de classes, se necessário
* Incluir métodos de explicabilidade de modelos (por exemplo, SHAP ou LIME)
* Organizar o código em módulos reutilizáveis fora do notebook
* Criar uma API simples para exposição do modelo

---

## 👨‍💻 Autor

**Felipe Cidade**

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT.
Veja o arquivo `LICENSE` para mais detalhes.
