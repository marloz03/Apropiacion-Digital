# Monitor de Inclusión Digital Colombia

Dashboard interactivo para el análisis de patrones de adopción digital en Colombia basado en datos de la Comisión Nacional de Telecomunicaciones (CNC) 2023.

## 📊 ¿Qué es este programa?

El Monitor de Inclusión Digital Colombia es una herramienta avanzada de análisis de datos que:

- **Identifica segmentos de población** según su nivel de adopción digital usando metodología FAMD (Factor Analysis of Mixed Data) + K-Means
- **Genera insights automáticos** con inteligencia artificial para entender patrones de comportamiento digital
- **Visualiza geográficamente** la distribución de la brecha digital en el territorio colombiano
- **Proporciona recomendaciones** basadas en datos para políticas públicas de inclusión digital

## 🎯 ¿Qué hace?

### Funcionalidades Principales

1. **Análisis de Segmentación**
   - Identifica 7 segmentos únicos de población basados en comportamiento digital
   - Utiliza 15+ variables relacionadas con acceso y uso de tecnología
   - Aplica metodología estadística avanzada (FAMD + clustering)

2. **Insights con IA**
   - Genera análisis automático de cada segmento usando GPT-4o
   - Proporciona recomendaciones específicas para políticas públicas
   - Identifica oportunidades y desafíos por segmento

3. **Visualización Interactiva**
   - Dashboard web profesional con 3 secciones principales
   - Mapas geográficos interactivos de Colombia
   - Gráficos estadísticos en tiempo real

4. **Carga de Datos Personalizada**
   - Permite subir nuevos datasets para análisis
   - Regenera automáticamente clusters y análisis
   - Mantiene histórico de resultados

**Para más información sobre cómo utilizar este producto, por favor descargar el manual de usuario ubicado en la carpeta "Manuales"**

## 🚀 Instalación con Docker

### Requisitos Previos

- **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux)
- **8GB RAM mínimo** recomendado
- **Puerto 5000** disponible

### Instalación en AWS EC2

1. **Conectarse al servidor EC2**
```bash
ssh -i tu-clave.pem ubuntu@tu-ip-ec2
```

2. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd Apropiacion-Digital
```

3. **Configurar clave de OpenAI (opcional)**
```bash
export OPENAI_API_KEY="tu-clave-openai"
```

4. **Ejecutar script de instalación**
```bash
chmod +x deploy_aws.sh
./deploy_aws.sh
```

5. **Acceder a la aplicación**
```
http://tu-ip-ec2:5000
```

### Personalización de Datos

El sistema incluye datos CNC 2023 preconfigurados, pero puedes:

1. **Subir nuevos datasets** a través de la interfaz web
2. **Reemplazar archivo base** en `data/original/`
3. **Ajustar variables de análisis** en `famd_clustering.py`

## 📈 Uso del Dashboard

### 1. Resumen Ejecutivo
- Visión general de los 9 segmentos identificados
- Métricas clave de adopción digital
- Tendencias geográficas principales

### 2. Análisis de Segmentos
- Exploración detallada de cada segmento
- Características demográficas y tecnológicas
- Mapas de distribución territorial

### 3. Insights de IA
- Análisis automático con inteligencia artificial
- Recomendaciones específicas por segmento
- Identificación de oportunidades de política pública

## 📊 Datos y Metodología

### Fuente de Datos
- **Comisión Nacional de Telecomunicaciones (CNC) 2023**
- Encuesta nacional de adopción digital

### Metodología de Análisis
1. **Preprocesamiento**: Limpieza y codificación de variables categóricas
2. **FAMD**: Reducción dimensional para variables mixtas (categóricas + numéricas)
3. **K-Means**: Clustering final para identificar segmentos
4. **Validación**: Análisis de silueta y coherencia interna

### Variables Clave Analizadas
- Acceso a internet fijo y móvil
- Dispositivos tecnológicos en el hogar
- Habilidades digitales de los miembros
- Estrato socioeconómico
- Ubicación geográfica
- Nivel educativo

---

**Desarrollado con:** Python, Streamlit, Plotly, Scikit-learn, OpenAI GPT-4o  
**Optimizado para:** AWS EC2, Docker, Windows Desktop
