@echo off
REM Script de despliegue para Windows
REM Monitor de Inclusión Digital Colombia

echo 🚀 Iniciando despliegue del Monitor de Inclusión Digital Colombia

echo 📦 Verificando Docker Desktop...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Desktop no está instalado
    echo 📥 Descarga Docker Desktop desde: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo 🔨 Construyendo imagen Docker...
docker build -t monitor-inclusion-digital .
if %errorlevel% neq 0 (
    echo ❌ Error construyendo la imagen
    pause
    exit /b 1
)

echo ⏹️ Deteniendo contenedor anterior si existe...
docker stop monitor-inclusion >nul 2>&1
docker rm monitor-inclusion >nul 2>&1

echo ▶️ Ejecutando aplicación...
docker run -d --name monitor-inclusion -p 5000:5000 --restart unless-stopped monitor-inclusion-digital
if %errorlevel% neq 0 (
    echo ❌ Error ejecutando el contenedor
    pause
    exit /b 1
)

echo ✅ Despliegue completado!
echo 🌐 La aplicación está disponible en: http://localhost:5000
echo 📊 Para ver logs: docker logs monitor-inclusion
echo 🔄 Para reiniciar: docker restart monitor-inclusion
echo 🛑 Para detener: docker stop monitor-inclusion

pause