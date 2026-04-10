#!/bin/bash

# Script de instalação rápida para o projeto Churn Telecom

echo "🚀 Instalando projeto_churn_telecom..."

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Instalação concluída!"
echo ""
echo "Para ativar o ambiente virtual: source venv/bin/activate"
echo "Para executar o dashboard: streamlit run dashboard/app.py"
echo "Para executar os notebooks: jupyter lab"
