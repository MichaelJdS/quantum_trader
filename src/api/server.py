from fastapi import FastAPI

app = FastAPI(title="QuantumTrader API")

@app.get("/")
def read_root():
    return {"status": "QuantumTrader Core is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}