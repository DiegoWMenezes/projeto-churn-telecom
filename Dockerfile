FROM python:3.10-slim

WORKDIR /app

# Copiar arquivos de requirements primeiro para cache
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Expor porta do Streamlit
EXPOSE 8501

# Comando para iniciar o dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
