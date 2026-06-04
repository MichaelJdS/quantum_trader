#!/usr/bin/env bash
# scripts/deploy_vm.sh
# Deploy do Quantum Trader Backend no Google Compute Engine e2-micro (free tier)
#
# Uso:
#   bash scripts/deploy_vm.sh
#
# Pré-requisitos:
#   - gcloud CLI instalado e autenticado (gcloud auth login)
#   - Projeto configurado (gcloud config set project quantum-trader-app)
#   - Billing DESATIVADO é OK para e2-micro no free tier (us-central1)

set -euo pipefail

# ═══════════════════════════════════════════════════
# CONFIGURAÇÕES — Edite conforme necessário
# ═══════════════════════════════════════════════════
PROJECT_ID="quantum-trader-app"
INSTANCE_NAME="quantum-trader-backend"
ZONE="us-central1-a"            # us-central1 é elegível ao free tier
MACHINE_TYPE="e2-micro"         # 1 vCPU, 1GB RAM — FREE TIER
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
DISK_SIZE="20GB"                # 30GB é o limite do free tier para boot disk
PORT="8080"
API_TOKEN="$(openssl rand -hex 32)"  # Token gerado automaticamente

# ═══════════════════════════════════════════════════
# CORES
# ═══════════════════════════════════════════════════
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo -e "${BLUE}  Quantum Trader — Deploy GCP e2-micro    ${NC}"
echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo ""

# Define o projeto
gcloud config set project "$PROJECT_ID" --quiet

# ── 1. Criar a VM (se não existir) ──────────────────────────────────
echo -e "${YELLOW}[1/5] Verificando instância VM...${NC}"
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${GREEN}✓ Instância já existe. Pulando criação.${NC}"
else
    echo -e "${YELLOW}Criando VM e2-micro em $ZONE...${NC}"
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type="pd-standard" \
        --tags="quantum-trader,http-server" \
        --metadata="enable-osconfig=true" \
        --quiet
    echo -e "${GREEN}✓ VM criada!${NC}"
    echo -e "${YELLOW}Aguardando VM inicializar (30s)...${NC}"
    sleep 30
fi

# ── 2. Abrir porta no firewall ────────────────────────────────────────
echo -e "${YELLOW}[2/5] Configurando firewall...${NC}"
if ! gcloud compute firewall-rules describe "allow-quantum-trader" &>/dev/null 2>&1; then
    gcloud compute firewall-rules create "allow-quantum-trader" \
        --allow="tcp:${PORT}" \
        --source-ranges="0.0.0.0/0" \
        --target-tags="quantum-trader" \
        --description="Quantum Trader API" \
        --quiet
fi
echo -e "${GREEN}✓ Firewall configurado.${NC}"

# ── 3. Instalar dependências na VM ───────────────────────────────────
echo -e "${YELLOW}[3/5] Instalando Python e Docker na VM...${NC}"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --quiet -- bash << 'REMOTE_EOF'
set -e
# Atualiza sistema
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl

# Instala Docker (caso queira rodar via container no futuro)
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

echo "✓ Dependências instaladas"
REMOTE_EOF
echo -e "${GREEN}✓ VM preparada.${NC}"

# ── 4. Enviar código para a VM ────────────────────────────────────────
echo -e "${YELLOW}[4/5] Enviando código para a VM...${NC}"
# Cria um tar do projeto (excluindo .venv, __pycache__ e db local)
TMPTAR="/tmp/quantum_trader_deploy.tar.gz"
tar -czf "$TMPTAR" \
    --exclude=".venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".git" \
    --exclude="quantum_trader.db" \
    --exclude="models_store" \
    --exclude="logs" \
    -C "$(dirname "$(pwd)")" \
    "$(basename "$(pwd)")"

gcloud compute scp "$TMPTAR" "${INSTANCE_NAME}:/tmp/" --zone="$ZONE" --quiet

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --quiet -- bash << REMOTE_EOF2
set -e
mkdir -p ~/quantum_trader
tar -xzf /tmp/quantum_trader_deploy.tar.gz -C ~/quantum_trader --strip-components=1
cd ~/quantum_trader

# Cria virtualenv e instala dependências
python3.11 -m venv .venv
source .venv/bin/activate
pip install --quiet --no-cache-dir -r requirements_cloud.txt

# Cria/atualiza o .env com as variáveis de ambiente
cat > .env << ENV_EOF
API_TOKEN=${API_TOKEN}
PORT=${PORT}
GEMINI_API_KEY=\${GEMINI_API_KEY:-}
GEMINI_INTERVAL=300
ENV_EOF

echo "✓ Código instalado"
REMOTE_EOF2
echo -e "${GREEN}✓ Código enviado.${NC}"

# ── 5. Criar e iniciar serviço systemd ──────────────────────────────
echo -e "${YELLOW}[5/5] Criando serviço systemd (24/7)...${NC}"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --quiet -- bash << 'REMOTE_EOF3'
set -e
# Cria serviço systemd para rodar o bot como daemon
sudo tee /etc/systemd/system/quantum-trader.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Quantum Trader Cloud Backend
After=network.target

[Service]
Type=simple
User=USER_PLACEHOLDER
WorkingDirectory=/home/USER_PLACEHOLDER/quantum_trader
ExecStart=/home/USER_PLACEHOLDER/quantum_trader/.venv/bin/python -m uvicorn cloud_api.main:app --host 0.0.0.0 --port 8080 --workers 1
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
EnvironmentFile=/home/USER_PLACEHOLDER/quantum_trader/.env

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Substitui o placeholder pelo usuário atual
CURRENT_USER=$(whoami)
sudo sed -i "s/USER_PLACEHOLDER/${CURRENT_USER}/g" /etc/systemd/system/quantum-trader.service

sudo systemctl daemon-reload
sudo systemctl enable quantum-trader
sudo systemctl restart quantum-trader

echo "✓ Serviço systemd ativo"
systemctl status quantum-trader --no-pager | head -20
REMOTE_EOF3

# ── Resultado final ───────────────────────────────────────────────────
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Deploy Concluído com Sucesso!           ${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}URL do Backend:${NC}  http://${EXTERNAL_IP}:${PORT}"
echo -e "  ${BLUE}API Token:${NC}       ${API_TOKEN}"
echo ""

# ── Grava automaticamente no .env local (para o App Windows) ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
if [[ -f "$ENV_FILE" ]]; then
    # Atualiza CLOUD_SERVER_URL
    if grep -q "^CLOUD_SERVER_URL=" "$ENV_FILE"; then
        sed -i "s|^CLOUD_SERVER_URL=.*|CLOUD_SERVER_URL=http://${EXTERNAL_IP}:${PORT}|" "$ENV_FILE"
    else
        echo "CLOUD_SERVER_URL=http://${EXTERNAL_IP}:${PORT}" >> "$ENV_FILE"
    fi
    # Atualiza API_TOKEN
    if grep -q "^API_TOKEN=" "$ENV_FILE"; then
        sed -i "s|^API_TOKEN=.*|API_TOKEN=${API_TOKEN}|" "$ENV_FILE"
    else
        echo "API_TOKEN=${API_TOKEN}" >> "$ENV_FILE"
    fi
    echo -e "  ${GREEN}✓ URL e Token gravados no .env local automaticamente!${NC}"
    echo -e "  ${GREEN}  → Abra o App Windows e as configurações já estarão preenchidas.${NC}"
else
    echo -e "  ${YELLOW}⚠️  IMPORTANTE: Salve o API Token acima no App Windows → Configurações!${NC}"
fi
echo ""
echo -e "  Para ver os logs da VM:"
echo -e "  ${BLUE}gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- journalctl -u quantum-trader -f${NC}"
echo ""
