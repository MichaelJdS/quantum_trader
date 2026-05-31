# Usa a versão slim para menor footprint no servidor
FROM python:3.11-slim

# Evita que o Python grave arquivos .pyc e força o stdout direto (fundamental para logs no Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalação de dependências de compilação do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Otimização do cache de pacotes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia a infraestrutura completa do bot
COPY src/ src/
COPY scripts/ scripts/
COPY main.py .

# Exposição da porta de métricas (Prometheus)
EXPOSE 8000

# Executa o motor central
CMD ["python", "main.py"]