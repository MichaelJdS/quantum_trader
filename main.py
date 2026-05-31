import os
import asyncio
import logging
from dotenv import load_dotenv
from prometheus_client import start_http_server
from src.core.orchestrator import DerivOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | SYSTEM:%(name)s - %(message)s"
)
logger = logging.getLogger("MainControl")

def setup_environment():
    """Carrega variáveis de ambiente e valida integridade."""
    load_dotenv()
    
    app_id = os.getenv("DERIV_APP_ID")
    # CORREÇÃO: Agora aceita tanto DERIV_API_TOKEN quanto o seu DERIV_TOKEN
    api_token = os.getenv("DERIV_API_TOKEN") or os.getenv("DERIV_TOKEN")
    
    if not app_id or not api_token:
        logger.error("🚨 Credenciais da Deriv ausentes! Verifique o seu arquivo .env.")
        exit(1)
        
    return app_id, api_token

async def main():
    app_id, api_token = setup_environment()
    
    # 1. Ativação do Motor de Telemetria (Grafana/Prometheus)
    logger.info("📊 Inicializando servidor de telemetria Prometheus na porta 8000...")
    start_http_server(8000)
    
    # 2. Instanciação do Motor HFT
    bot = DerivOrchestrator(app_id=app_id, api_token=api_token, symbol="R_100")
    
    # 3. Execução
    try:
        logger.info("🚀 Iniciando Sequência de Ignição Quantum Trader...")
        await bot.start()
    except asyncio.CancelledError:
        logger.info("⚠️ Tarefa assíncrona cancelada.")
    except Exception as e:
        logger.critical(f"❌ Erro fatal no motor de execução: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 [SYSTEM] Parada de Emergência acionada pelo operador.")