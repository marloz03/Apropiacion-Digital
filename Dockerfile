# Imagen base de Python optimizada
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY dependencies.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r dependencies.txt

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data models

# Exponer puerto
EXPOSE 5000

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=5000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "production_app.py", "--server.port=5000", "--server.address=0.0.0.0"]