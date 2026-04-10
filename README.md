# Análise de Churn em Telecomunicações

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Projeto de análise preditiva para identificação de clientes com risco de cancelamento em empresas de telecomunicações.

## Sobre o Projeto

Este trabalho foi desenvolvido para construir um modelo de machine learning capaz de identificar padrões nos dados dos clientes que indicam maior probabilidade de cancelamento dos serviços (churn). O projeto inclui análise exploratória, feature engineering, modelagem e um dashboard interativo.

## Problema de Negócio

Empresas de telecomunicações enfrentam desafios significativos com a rotatividade de clientes. Identificar quais clientes estão propensos a cancelar permite que a empresa tome medidas preventivas, como ofertas personalizadas ou melhorias no atendimento.

## Resultados Principais

O modelo Random Forest apresentou o melhor desempenho:
- **ROC AUC:** 0.744
- **Precision:** 56.5%
- **Recall:** 53.8%

As features mais importantes para predição de churn:
1. Número de reclamações nos últimos 30 dias
2. Tempo de relacionamento com a empresa (tenure)
3. Tipo de contrato

## Dashboard

### Overview

![Overview](./reports/img/Overview.png)

### Análise de Clientes

![Analise de Clientes](./reports/img/Analise%20de%20Clientes.png)

### Churn por Segmentos

![Churn por Segmentos](./reports/img/churn%20por%20segmentos.png)

### Modelos de Predição

![Modelos de Predição](./reports/img/Modelos%20de%20predição.png)

### Score de Risco

![Score de Risco](./reports/img/Score%20de%20Risco.png)

## Estrutura do Projeto

```
projeto_churn_telecom/
├── data/
│   ├── raw/          # Dados brutos
│   └── processed/    # Dados processados
├── notebooks/        # Jupyter notebooks
├── src/
│   ├── data/        # Carregamento de dados
│   ├── features/    # Feature engineering
│   ├── models/      # Modelos de ML
│   └── visualization/  # Gráficos
├── reports/
│   ├── img/         # Imagens do dashboard
│   └── tables/      # Tabelas exportadas
├── config/          # Configurações
└── dashboard/      # Dashboard Streamlit
```

## Como Executar

```bash
# Clonar repositório
git clone https://github.com/DiegoWMenezes/projeto-churn-telecom.git
cd projeto-churn-telecom

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar análise (notebooks)
jupyter lab notebooks/

# Executar dashboard
streamlit run dashboard/app.py
```

## Tecnologias

- **Python 3.10+**
- **Pandas, NumPy** - Manipulação de dados
- **Scikit-learn** - Machine learning
- **XGBoost, LightGBM** - Gradient Boosting
- **Matplotlib, Seaborn, Plotly** - Visualização
- **Streamlit** - Dashboard interativo

## Insights Principais

1. **Reclamações são o fator mais importante**: Clientes com 3+ reclamações em 30 dias têm chance significativamente maior de churn.

2. **Contratos MTM têm alto risco**: Contratos month-to-month apresentam taxa de churn muito superior a contratos anuais ou bienais.

3. **Tenure baixo = risco alto**: Clientes nos primeiros 6 meses são os mais propensos a cancelar.

4. **Serviços de internet fiber optic** apresentam o maior churn entre os tipos de conexão.

## Próximos Passos

- Implementar modelo em produção
- Testar com dados reais da empresa
- Desenvolver sistema de alertas em tempo real
- Criar API para predições

## Licença

MIT License
