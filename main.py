import os
import asyncio
import logging
from dotenv import load_dotenv
from src.core.orchestrator import DerivOrchestrator

# Configuração de observabilidade de nível raiz
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | SYSTEM:%(name)s - %(message)s"
)
logger = logging.getLogger("MainControl")

def setup_environment():
    """Carrega variáveis de ambiente e valida integridade de chaves."""
    load_dotenv()
    
    app_id = os.getenv("DERIV_APP_ID")
    api_token = os.getenv("DERIV_API_TOKEN")
    
    if not app_id or not api_token:
        logger.error("🚨 Credenciais da Deriv ausentes!")
        logger.error("Crie um arquivo .env na raiz do projeto com DERIV_APP_ID e DERIV_API_TOKEN.")
        exit(1)
        
    return app_id, api_token

async def main():
    # 1. Validação de Segurança
    app_id, api_token = setup_environment()
    
    # 2. Instanciação do Motor HFT
    # O ativo está configurado para R_100 (Volatility 100 Index). Altere se treinou para outro.
    bot = DerivOrchestrator(app_id=app_id, api_token=api_token, symbol="R_100")
    
    # 3. Execução Isolada
    try:
        logger.info("🚀 Iniciando Sequência de Ignição Quantum Trader...")
        await bot.start()
    except asyncio.CancelledError:
        logger.info("⚠️ Tarefa assíncrona cancelada. Desligando...")
    except Exception as e:
        logger.critical(f"❌ Erro fatal no motor de execução: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        # Loop Assíncrono de Alta Performance
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 [SYSTEM] Parada de Emergência acionada pelo operador (Ctrl+C). Fechando conexões de forma segura.")