#!/bin/bash

# Script de despliegue para AWS EC2
# Monitor de Inclusión Digital Colombia

echo "🚀 Iniciando despliegue del Monitor de Inclusión Digital Colombia"

# Actualizar sistema
echo "📦 Actualizando sistema..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Instalar Docker
echo "🐳 Instalando Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Iniciar Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Construir imagen Docker
echo "🔨 Construyendo imagen Docker..."
sudo docker build -t monitor-inclusion-digital .

# Ejecutar contenedor
echo "▶️ Ejecutando aplicación..."
sudo docker run -d \
    --name monitor-inclusion \
    -p 5000:5000 \
    --restart unless-stopped \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    monitor-inclusion-digital

echo "✅ Despliegue completado!"
echo "🌐 La aplicación está disponible en: http://$(curl -s ifconfig.me):5000"
echo "📊 Para ver logs: sudo docker logs monitor-inclusion"
echo "🔄 Para reiniciar: sudo docker restart monitor-inclusion"