import sys
import asyncio
from PyQt6.QtWidgets import QApplication
from src.ui.dashboard import TradingDashboard
from src.core.orchestrator import OrchestratorThread
from src.database import init_db
from loguru import logger

logger.add("quantum.log", rotation="10MB", retention="7d", level="INFO")

def main():
    # 1️⃣ Inicializa DB de forma síncrona com segurança
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db())
    loop.close()

    app = QApplication(sys.argv)
    ui = TradingDashboard()
    ui.show()

    # 2️⃣ Thread segura para WebSockets + Asyncio
    worker = OrchestratorThread()
    
    # 3️⃣ Conexões thread-safe (Signals)
    worker.emitter.tick_update.connect(ui.update_data)
    worker.emitter.log_message.connect(ui.log_msg)
    worker.emitter.connection_status.connect(
        lambda status: ui.log_msg("🟢 Conectado à Deriv" if status else "🔴 Desconectado")
    )

    # 4️⃣ Inicia/Para o worker ao clicar no botão
    def toggle_system():
        if not worker.isRunning():
            worker.start()
            ui.log_msg("🚀 Sistema iniciado em background thread")
            ui.btn_start.setText("⏹ PARAR")
        else:
            ui.log_msg("⏳ Parando sistema de forma segura...")
            worker.stop()
            worker.wait()
            ui.log_msg("🛑 Sistema finalizado com segurança")
            ui.btn_start.setText("▶ INICIAR")

    ui.btn_start.clicked.connect(toggle_system)

    # 5️⃣ Gancho de encerramento seguro (Evita processos zumbis na VPS)
    app.aboutToQuit.connect(worker.stop)
    app.aboutToQuit.connect(worker.wait)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()