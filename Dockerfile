FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Esponiamo esplicitamente la porta 8000 (quella standard di Chainlit)
EXPOSE 8000

# Il comando blindato che forza l'host e la porta
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]