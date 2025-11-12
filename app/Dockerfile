FROM python:3.12-slim

WORKDIR /app

# 1️⃣ Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 git \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Copia apenas o arquivo de dependências primeiro (mantém cache entre builds)
COPY requirements.txt .

# 3️⃣ Instala as libs Python (mantém cache se o requirements não mudar)
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copia o resto do código (esta etapa muda com frequência)
COPY . .

# 5️⃣ Comando padrão
CMD ["python", "main.py"]
