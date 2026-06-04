# ─────────────────────────────────────────────────────────────────
# Quantum Trader — Dockerfile (Backend Cloud)
# Roda no Google Compute Engine e2-micro (free tier)
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Evita prompts interativos do apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala dependências do sistema (necessárias para compilar algumas libs Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python do backend
# (versão sem PyQt6, textual e torch pesado — apenas o necessário para o cloud)
COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

# Copia o código do projeto
COPY cloud_api/ ./cloud_api/
COPY core/ ./core/
COPY infra/ ./infra/
COPY ml/ ./ml/
COPY .env* ./

# Porta exposta (Cloud Run / VM)
EXPOSE 8080

# Healthcheck para o GCP saber que o serviço está saudável
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando de inicialização
CMD ["python", "-m", "uvicorn", "cloud_api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
