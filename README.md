# ⚡ Quantum Trader

> Sistema automatizado de trading para a plataforma **Deriv** com Machine Learning,
> backtesting vetorizado, TUI em tempo real e monitoramento via Prometheus + Grafana.

![Python](https://img.shields.io/badge/Python-3.11+-4f98a3?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-dd6974?style=flat-square&logo=pytorch)
![Textual](https://img.shields.io/badge/TUI-Textual-6daa45?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-fdab43?style=flat-square)

---

## 🆕 Atualizações Recentes

- **Correções de Infraestrutura**: Melhorias de concorrência e thread-safety no `SymbolManager` e `Cache`, incluindo timestamps precisos e handlers de ticks globais otimizados.
- **Cliente Deriv Resiliente**: Correção na inicialização assíncrona do WebSocket e fallback automático caso a conexão caia.
- **Suporte Expandido a Ativos**: Suporte corrigido para Índices Sintéticos (Volatility Index), garantindo a leitura correta de preços pelo campo `quote`.

## ⚠️ Aviso de Risco

> **Trading automatizado envolve risco real de perda de capital.**
> Sempre use `--dry-run` para validar estratégias antes de operar com dinheiro real.
> Resultados de backtesting não garantem performance futura.

---

## 📐 Arquitetura

┌─────────────────────────────────────────────────────────────────────┐
│ QUANTUM TRADER │
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Deriv API │────▶│ Symbol Mgr │────▶│ Feature Engineer │ │
│ │ WebSocket │ │ OHLCV Cache │ │ EMA/RSI/MACD/ATR │ │
│ └──────────────┘ └──────────────┘ └────────┬─────────┘ │
│ │ │
│ ┌───────────────────────────────────────────────────▼──────────┐ │
│ │ ML PIPELINE │ │
│ │ ┌─────────────┐ ┌──────────────┐ ┌──────────────────────┐│ │
│ │ │ LSTM Model │ │ Online River │ │ Ensemble (Dynamic ││ │
│ │ │ (PyTorch) │ │ Hoeffding + │ │ Weighted Voting) ││ │
│ │ │ NeuronMonit │ │ ADWIN Drift │ │ Kelly Sizing ││ │
│ │ └──────┬──────┘ └──────┬───────┘ └──────────┬───────────┘│ │
│ └─────────┼────────────────┼─────────────────────┼────────────┘ │
│ └────────────────┴─────────────────────┘ │
│ │ Sinal (BUY/SELL + confiança) │
│ ┌───────────────────────────▼──────────────────────────────────┐ │
│ │ EXECUTION ENGINE │ │
│ │ Risk Manager (Kelly, ATR Stop, Max DD) ──▶ Trade Executor │ │
│ │ Session State ──▶ DB (SQLAlchemy) ──▶ Metrics Update │ │
│ └───────────────────────────────────────────────────────────────┘ │
│ │
│ ┌─────────────────┐ ┌──────────────┐ ┌────────────────────────┐ │
│ │ TUI (Textual) │ │ Backtester │ │ Prometheus + Grafana │ │
│ │ 6 telas live │ │ Vetorizado │ │ MLflow Experiments │ │
│ │ Dashboard/Risk │ │ Event-Drvn │ │ Alertas automáticos │ │
│ │ Neurônios/ML │ │ Walk-Fwd │ │ Dashboard interativo │ │
│ └─────────────────┘ └──────────────┘ └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘


---

## 🚀 Início Rápido

### 1. Clone e Configure

```bash
git clone https://github.com/MichaelJdS/quantum_trader.git
cd quantum_trader

# Cria ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Instala dependências
make install

# Configura variáveis de ambiente
cp .env.example .env
# Edite .env com seu DERIV_APP_ID e DERIV_API_TOKEN
```

### 2. Modo Dry-Run (Recomendado para Início)

```bash
# Executa sem TUI, apenas logs
python main.py --dry-run --symbols R_50 R_75

# Com TUI interativa
python main.py --dry-run --symbols R_50 R_75 --tui

# Via Makefile
make run-dry
```

### 3. Backtesting

```bash
# Todas as estratégias, 5000 candles, com Walk-Forward
make backtest

# Estratégia específica
python scripts/run_backtest.py \
    --symbol R_50 \
    --strategy ema_rsi \
    --candles 3000 \
    --walk-forward

# Modo event-driven (mais realista, mais lento)
python scripts/run_backtest.py \
    --symbol R_75 \
    --all-strategies \
    --event-driven \
    --candles 1000

# Relatório HTML gerado em ./reports/backtest_report_*.html
```

### 4. Stack Docker Completa

```bash
# Sobe trader + Prometheus + Grafana + MLflow + Redis
make up

# Interfaces:
# Grafana:    http://localhost:3000  (admin / quantum2024)
# Prometheus: http://localhost:9090
# MLflow:     http://localhost:5000
# Métricas:   http://localhost:8000/metrics

# Logs em tempo real
make logs

# Parar tudo
make down
```

---

## 📁 Estrutura do Projeto
quantum_trader/
├── main.py ← Entrypoint CLI principal
├── .env.example ← Template de configuração
├── requirements.txt ← Dependências Python
├── Makefile ← Comandos utilitários
├── Dockerfile ← Multi-stage build
├── docker-compose.yml ← Stack completa
│
├── core/
│ ├── settings.py ← Configurações (Pydantic Settings)
│ ├── entities.py ← Dataclasses de domínio
│ ├── enums.py ← Enums (StakeMode, Direction…)
│ ├── exceptions.py ← Exceções customizadas
│ ├── strategy_base.py ← Contrato base das estratégias
│ ├── risk_manager.py ← Kelly, ATR, drawdown, circuit breaker
│ ├── execution_engine.py ← Orquestrador principal
│ └── strategies/
│ ├── ema_rsi.py ← EMA Crossover + RSI + Volume
│ ├── bollinger_reversion.py ← Mean Reversion + Keltner
│ └── breakout.py ← Breakout + ADX + Volume
│
├── infra/
│ ├── deriv_client.py ← WebSocket client com reconexão
│ ├── symbol_manager.py ← Subscriptions e cache de candles
│ ├── cache.py ← LRU/LFU cache in-memory
│ └── db/
│ ├── database.py ← SQLAlchemy async engine
│ ├── models.py ← ORM models
│ └── repository.py ← Repositórios CRUD
│
├── ml/
│ ├── feature_engineer.py ← 40+ features: EMA, RSI, MACD, ATR…
│ ├── online_learner.py ← River (Hoeffding + ADWIN drift)
│ ├── neuron_monitor.py ← Saúde das camadas LSTM
│ ├── mlops.py ← MLflow + Optuna
│ └── models/
│ ├── lstm_model.py ← LSTM PyTorch + atenção
│ └── ensemble.py ← Ensemble com pesos dinâmicos
│
├── backtesting/
│ ├── vectorized_backtester.py ← NumPy/Pandas, 100k candles/2s
│ ├── event_driven_backtester.py ← Priority queue, latência real
│ ├── walk_forward.py ← Walk-Forward com Purge Gap
│ ├── metrics.py ← 25+ métricas de performance
│ └── report_generator.py ← Relatório HTML interativo (Plotly)
│
├── tui/
│ ├── app.py ← App Textual principal
│ ├── state.py ← Estado compartilhado TUI↔Engine
│ ├── screens/ ← Dashboard, Símbolos, Trades…
│ └── widgets/ ← KPICard, RiskGauge, Sparkline…
│
├── monitoring/
│ ├── prometheus.yml ← Configuração scrape
│ ├── alerts.yml ← Regras de alerta
│ ├── metrics_server.py ← Servidor Prometheus Python
│ └── grafana/
│ ├── provisioning/ ← Auto-provisioning datasource/dash
│ └── dashboards/ ← quantum_trader.json
│
├── scripts/
│ ├── run_backtest.py ← CLI backtesting
│ └── train_offline.py ← Treino offline do LSTM
│
└── tests/
├── unit/ ← Testes unitários isolados
└── integration/ ← Testes com WebSocket mock

---

## ⚙️ Configuração de Risco

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `STAKE_MODE` | `fractional_kelly` | `fixed` \| `fractional` \| `fractional_kelly` |
| `BASE_STAKE` | `1.0` | Stake fixo (apenas modo `fixed`) |
| `STOP_LOSS_PCT` | `0.03` | Para se perder 3% da banca |
| `STOP_WIN_PCT` | `0.05` | Para se ganhar 5% da banca |
| `MAX_DRAWDOWN_PCT` | `0.05` | Drawdown máximo diário |
| `MAX_CONSECUTIVE_LOSSES` | `5` | Pausa após N perdas seguidas |
| `KELLY_FRACTION` | `0.25` | Fração do Kelly (conservador) |
| `CONFIDENCE_THRESHOLD` | `0.60` | Confiança mínima para abrir trade |

---

## 🧪 Testes

```bash
# Todos os testes com cobertura
make test

# Apenas unitários (rápido)
pytest tests/unit/ -v

# Com relatório HTML de cobertura
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## 🔬 Qualidade de Código

```bash
# Lint + auto-fix
make lint

# Apenas ruff
ruff check . --fix

# Type checking
mypy . --ignore-missing-imports

# Pre-commit (executa automaticamente antes de cada commit)
pre-commit run --all-files
```

---

## 📊 Métricas Monitoradas

| Métrica | Descrição |
|---|---|
| `qt_current_balance` | Saldo atual da conta |
| `qt_win_rate` | Win rate da sessão |
| `qt_session_drawdown_pct` | Drawdown percentual |
| `qt_consecutive_losses` | Perdas consecutivas |
| `qt_sharpe_ratio` | Sharpe ratio em tempo real |
| `qt_model_confidence` | Distribuição de confiança dos sinais |
| `qt_online_learner_accuracy` | Accuracy do learner online |
| `qt_neuron_dead_pct` | % neurônios mortos por camada |
| `qt_ws_latency_ms` | Latência WebSocket |

---

## 🛡️ Segurança

- **Nunca** commite o arquivo `.env` (está no `.gitignore`)
- Use `DRY_RUN=true` até validar sua estratégia exaustivamente
- O usuário Docker é não-root (`trader:1000`)
- Tokens Deriv nunca aparecem em logs
- Circuit breaker automático por drawdown e perdas consecutivas

---

## 📜 Licença

MIT License — use por sua conta e risco.