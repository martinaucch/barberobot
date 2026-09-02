FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face richiede che l'app giri sulla porta 7860
CMD ["chainlit", "run", "app.py", "--port", "7860", "--host", "0.0.0.0"]