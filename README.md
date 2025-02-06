# Projeto de Predição de Preços de Aluguéis Temporários na Cidade de Nova York com XGBoost

## Visão Geral
Este projeto tem como objetivo a predição de preços de aluguel utilizando aprendizado de máquina. O modelo treinado é baseado no algoritmo **XGBoost** e foi desenvolvido no Google Colab.

O repositório contém:
- `Modelo.ipynb`: Notebook com o código completo para treinamento, avaliação e predição do modelo.
- `requirements.txt`: Arquivo com todos os pacotes utilizados e suas versões.
- `modelo_xgb.pkl`: Arquivo contendo o modelo treinado.
- `README.md`: Documentação sobre instalação e execução do projeto.

---

## **Requisitos**
Certifique-se de ter instalado os seguintes pacotes antes de rodar o projeto:

- Python 3.8+
- Jupyter Notebook ou Google Colab
- `scikit-learn==1.5.2`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `xgboost`

Para instalar os pacotes necessários, execute:

```bash
pip install -r requirements.txt
```

Caso esteja executando no Google Colab, inclua no início do seu notebook:

```python
!pip uninstall -y scikit-learn
!pip install scikit-learn==1.5.2
```

---

## **Execução do Projeto**
### **1️⃣ Clonar o Repositório**

Se ainda não clonou o repositório, faça isso usando:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### **2️⃣ Abrir o Notebook**

Abra o arquivo `Modelo.ipynb` no Google Colab ou Jupyter Notebook.

### **3️⃣ Executar o Notebook**

Siga as instruções no notebook para carregar os dados, treinar o modelo e fazer previsões. Caso queira apenas carregar o modelo salvo e prever novos valores, utilize:

```python
import joblib
import pandas as pd

# Carregar o modelo treinado
modelo = joblib.load("modelo_xgb.pkl")

# Criar um novo exemplo para predição
novo_apartamento = pd.DataFrame([{
    'id': 2595,
    'host_id': 2845,
    'bairro_group': 'Manhattan',
    'bairro': 'Midtown',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type': 'Entire home/apt',
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355
}])

# Aplicar transformações de pré-processamento
if 'ultima_review' not in novo_apartamento.columns:
    novo_apartamento['dias_desde_ultima_review'] = -1
    novo_apartamento['tem_review'] = 0
    novo_apartamento['ultima_review'] = pd.Timestamp('1900-01-01')
else:
    novo_apartamento['dias_desde_ultima_review'] = (pd.Timestamp.now() - novo_apartamento['ultima_review']).dt.days
    novo_apartamento['tem_review'] = 1

# Convertendo variáveis categóricas em dummies (one-hot encoding)
novo_apartamento['bairro_group_Brooklyn'] = 0
novo_apartamento['bairro_group_Manhattan'] = 0
novo_apartamento['bairro_group_Queens'] = 0
novo_apartamento['bairro_group_Staten Island'] = 0
if (novo_apartamento['bairro_group'].values[0] == 'Brooklyn'):
    novo_apartamento['bairro_group_Brooklyn'] = 1    
elif (novo_apartamento['bairro_group'].values[0] == 'Manhattan'):
    novo_apartamento['bairro_group_Manhattan'] = 1
elif (novo_apartamento['bairro_group'].values[0] == 'Queens'):
    novo_apartamento['bairro_group_Queens'] = 1
elif (novo_apartamento['bairro_group'].values[0] == 'Staten Island'):
    novo_apartamento['bairro_group_Staten Island'] = 1     

novo_apartamento['room_type_Private room'] = 0
novo_apartamento['room_type_Shared room'] = 0
if (novo_apartamento['room_type'].values[0] == 'Private room'):
    novo_apartamento['room_type_Private room'] = 1    
elif (novo_apartamento['room_type'].values[0] == 'Shared room'):
    novo_apartamento['room_type_Shared room'] = 1

# Smoothing para evitar overfitting
smoothing_factor = 10

# Target Encoding com Smoothing
novo_apartamento['bairro_encoded'] = (bairro_means[novo_apartamento['bairro'].values[0]] * bairro_counts[novo_apartamento['bairro'].values[0]] + global_mean * smoothing_factor) / (bairro_counts[novo_apartamento['bairro'].values[0]] + smoothing_factor)
novo_apartamento['host_encoded'] = (host_means[novo_apartamento['host_id'].values[0]] * host_counts[novo_apartamento['host_id'].values[0]] + global_mean * smoothing_factor) / (host_counts[novo_apartamento['host_id'].values[0]] + smoothing_factor)

# Frequency Encoding
novo_apartamento['bairro_freq'] = bairro_freq[novo_apartamento['bairro'].values[0]]
novo_apartamento['host_freq'] = host_freq[novo_apartamento['host_id'].values[0]]

novo_apartamento['preco_por_disponibilidade'] = preco_medio_por_disponibilidade[(preco_medio_por_disponibilidade['host_id'] == novo_apartamento['host_id'].values[0]) & (preco_medio_por_disponibilidade['bairro'] == novo_apartamento['bairro'].values[0])]['preco_por_disponibilidade'].values[0]
novo_apartamento['preco_por_reviews'] = preco_medio_por_reviews[(preco_medio_por_reviews['host_id'] == novo_apartamento['host_id'].values[0]) & (preco_medio_por_reviews['bairro'] == novo_apartamento['bairro'].values[0])]['preco_por_reviews'].values[0]
novo_apartamento['preco_por_minimo_noites'] = preco_medio_por_minimo_noites[(preco_medio_por_minimo_noites['host_id'] == novo_apartamento['host_id'].values[0]) & (preco_medio_por_minimo_noites['bairro'] == novo_apartamento['bairro'].values[0])]['preco_por_minimo_noites'].values[0]

# Coordenadas do centro da cidade de Nova York
centro_latitude = 40.7128
centro_longitude = -74.0060

# Calcular distância euclidiana
novo_apartamento['distancia_ao_centro'] = np.sqrt(
    (novo_apartamento['latitude'] - centro_latitude)**2 + (novo_apartamento['longitude'] - centro_longitude)**2
)

# Clusters geográficos usando o mesmo modelo KMeans treinado anteriormente
kmeans_trained = joblib.load("kmeans.pkl")
novo_apartamento['regiao_cluster'] = kmeans_trained.predict(novo_apartamento[['latitude', 'longitude']])

novo_apartamento['variação_preco_host'] = variação_media_preco_host[variação_media_preco_host['host_id'] == novo_apartamento['host_id'].values[0]]['variação_preco_host'].values[0]

novo_apartamento.drop(columns=['id', 'bairro_group', 'room_type', 'bairro', 'host_id'], inplace=True)

# Aplicar o mesmo pipeline de pré-processamento usado no treinamento
pipeline_preprocessamento = joblib.load("preprocessor.pkl")

# Pegar os nomes das colunas usadas no treinamento
colunas_treinadas = pipeline_preprocessamento.feature_names_in_

# Manter apenas as colunas que estavam no treinamento
novo_apartamento_alinhado = novo_apartamento[colunas_treinadas]

# Transformar os dados corretamente
novo_apartamento_encoded = pipeline_preprocessamento.transform(novo_apartamento_alinhado)

novo_apartamento_encoded = pd.DataFrame(novo_apartamento_encoded)
colunas_faltando = set(novo_apartamento.columns) - set(novo_apartamento_encoded.columns)
for coluna in colunas_faltando:
    novo_apartamento_encoded[coluna] = novo_apartamento[coluna]
novo_apartamento_encoded = novo_apartamento_encoded[X.columns]
# Carregar o modelo salvo
final_modelo = joblib.load('modelo_xgb.pkl')

# Fazer a previsão
start_time = time.time()
preco_previsto = final_modelo.predict(novo_apartamento_encoded)
inference_time = time.time() - start_time
print(f"Tempo de predição: {inference_time:.4f} segundos")

# Exibir o resultado
print(f"Preço previsto para o apartamento: ${preco_previsto[0]:.2f}")
```

---

## **Estrutura do Projeto**

```
/
├── Modelo.ipynb            # Notebook principal do projeto
├── modelo_xgb.pkl          # Modelo treinado salvo
├── kmeans.pkl              # Modelo KMeans treinado
├── preprocessor.pkl        # Pipeline de pré-processamento salvo
├── README.md               # Documentação do projeto
└── requirements.txt        # Lista de dependências
```

---

## **Ajuste de Hiperparâmetros**
Caso deseje melhorar o modelo, ajuste os hiperparâmetros do **XGBoost** usando `RandomizedSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Definir o modelo base
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Espaço de busca para os hiperparâmetros
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# RandomizedSearchCV para encontrar os melhores hiperparâmetros
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Treinar a busca
random_search.fit(X_train, y_train)
print("Melhores Hiperparâmetros:", random_search.best_params_)
```

---

## **Contato**
Caso tenha dúvidas ou sugestões, sinta-se à vontade para entrar em contato!

**Autor:** Igor Araujo de Mattos  
**Email:** yigor88mattos@gmail.com  
**LinkedIn:** [linkedin.com/in/igor-araujo-de-mattos-765a931a6](https://www.linkedin.com/in/igor-araujo-de-mattos-765a931a6/)

