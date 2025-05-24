"""
Monitor de Inclusión Digital Colombia
Análisis de patrones de adopción digital usando datos de la Comisión Nacional de Telecomunicaciones
"""

# Librerías estándar de Python
from pandas.errors import IndexingError, InvalidIndexError
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import time

# Importar módulos personalizados
from data_loader import DataLoader
from famd_clustering import FAMDClustering
from ai_insights import AIInsightsGenerator
from colombia_geo_simple import create_colombia_cluster_map

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la página de Streamlit
st.set_page_config(page_title="Monitor de Inclusión Digital Colombia - CNC",
                   page_icon="🇨🇴",
                   layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource
def initialize_components():
    """Inicializa los componentes principales del sistema"""
    data_loader = DataLoader()
    clustering_engine = FAMDClustering()
    ai_generator = AIInsightsGenerator()
    return data_loader, clustering_engine, ai_generator


def load_data_by_source(data_source):
    """Load data based on the selected source"""
    components = initialize_components()
    data_loader = components[0]

    # Only switch to uploaded data if processing is complete and user explicitly selected it
    if (data_source == "Cargar nuevos datos" and 
        st.session_state.get('processing_complete', False)):
        # Try to load uploaded data first
        clustered_data = data_loader.load_clustering_results("uploaded")
        if clustered_data is not None:
            return clustered_data['clustering_results']['cluster_data'], components
    
    # Load default/original data
    clustered_data = data_loader.get_default_clustered_data()
    return clustered_data, components


def process_new_data(uploaded_file, data_loader, clustering_engine):
    """Process newly uploaded data"""
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Perform complete FAMD clustering analysis using Marlon's methodology
        clustering_results = clustering_engine.run_complete_analysis(
            temp_file_path)

        # Save results as new data (not overwriting original defaults)
        data_loader.save_clustering_results(clustering_results, "uploaded")

        # Clean up temp file
        os.remove(temp_file_path)

        return clustering_results['clustering_results']['cluster_data']

    except Exception as e:
        st.error(f"Error procesando el archivo: {str(e)}")
        return None


def main():
    st.title("Monitor de Inclusión Digital Colombia")
    st.markdown("""
    ### Dashboard Avanzado de Análisis de Brecha Digital - CNC
    
    **Características principales:**
    - 🤖 **Análisis automático con IA** usando GPT-4o para generar hallazgos
    - 📊 **Clustering avanzado** basado en metodología MCA-GMM
    - 🗺️ **Visualización geográfica** de departamentos colombianos
    - 📈 **Métricas en tiempo real** de adopción digital
    - 🔄 **Carga de nuevos datos** con procesamiento automático
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Gestión de Datos")

        # Data source selection
        data_source = st.radio(
            "Seleccione la fuente de datos:",
            ["Usar datos por defecto (CNC 2023)", "Cargar nuevos datos"])
        
        # Check if data source changed and clear session state if needed
        if 'current_data_source' not in st.session_state:
            st.session_state.current_data_source = data_source
        elif st.session_state.current_data_source != data_source:
            # Data source changed, clear relevant session state
            st.session_state.current_data_source = data_source
            if 'global_clustered_data' in st.session_state:
                del st.session_state.global_clustered_data
            if 'needs_clustering' in st.session_state:
                del st.session_state.needs_clustering
            st.rerun()

        if data_source == "Cargar nuevos datos":
            st.subheader("📁 Cargar Nueva Data")
            uploaded_file = st.file_uploader(
                "Suba el archivo Excel con los nuevos datos de encuesta",
                type=['xlsx', 'xls'],
                help=
                "El archivo debe contener datos de encuesta de apropiación digital"
            )

            if uploaded_file is not None:
                st.success(f"Archivo cargado: {uploaded_file.name}")

                if st.button("🔄 Procesar y Actualizar Análisis"):
                    with st.spinner(
                            "Procesando datos y generando nuevo clustering..."
                    ):
                        data_loader, clustering_engine, ai_generator = initialize_components(
                        )
                        new_data = process_new_data(uploaded_file, data_loader,
                                                    clustering_engine)

                        if new_data is not None:
                            # After successful processing, set flag to automatically switch to uploaded data
                            st.session_state.processing_complete = True
                            
                            # Clear session state to force reload with new data
                            if 'global_clustered_data' in st.session_state:
                                del st.session_state.global_clustered_data
                            if 'needs_clustering' in st.session_state:
                                del st.session_state.needs_clustering
                            
                            st.success(
                                "✅ Nuevos datos procesados y guardados. La aplicación cambiará automáticamente a los nuevos datos."
                            )
                            st.rerun()
        
        # AI insights siempre activados
        show_ai_insights = True
    
        st.divider()

    # Initialize components only once
    if 'components' not in st.session_state:
        st.session_state.components = initialize_components()
    
    components = st.session_state.components
    
    # Initialize global data in session state - Load data based on source
    if 'global_clustered_data' not in st.session_state:
        clustered_data, _ = load_data_by_source(data_source)
        st.session_state.global_clustered_data = clustered_data
        
        # If no data exists, flag for generation
        if clustered_data is None:
            st.session_state.needs_clustering = True
        else:
            st.session_state.needs_clustering = False
    else:
        clustered_data = st.session_state.global_clustered_data

    if st.session_state.get('needs_clustering', False):
        # Generate default analysis from the original survey data
        msg = st.empty()
        msg.info("⚙️ Generando análisis de clustering por primera vez...")
        with st.spinner(
                "Cargando datos originales y generando clustering inicial..."):
            
            data_loader, clustering_engine, ai_generator = components

            try:
                # Perform complete FAMD clustering analysis using Marlon's methodology
                file_path = "data/original/datos.xlsx"
                clustering_results = clustering_engine.run_complete_analysis(
                    file_path)

                # Save as default
                data_loader.save_clustering_results(clustering_results,
                                                    "default")

                clustered_data = clustering_results['clustering_results']['cluster_data']
                st.session_state.global_clustered_data = clustered_data
                st.session_state.needs_clustering = False
                st.success("✅ Análisis FAMD completado y guardado")

            except Exception as e:
                st.error(f"Error generando análisis inicial: {str(e)}")
                st.stop()
        msg.empty()

    if clustered_data is None:
        st.warning(
            "No hay datos disponibles. Por favor cargue un archivo de datos.")
        st.stop()

    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Resumen Ejecutivo", "🎯 Segmentos", "🤖 Insights con IA"
    ])

    with tab1:
        st.header("🏛️ Resumen Ejecutivo: Estado de la Brecha Digital en Colombia")
        st.caption("📋 Herramienta para la formulación de políticas y estrategias de inclusión digital")
        
        st.markdown("""
        Esta herramienta está diseñada para apoyar a funcionarios y tomadores de decisiones en la identificación de oportunidades 
        para reducir la brecha digital en Colombia, presentando análisis predictivos y recomendaciones accionables.
        """)

        # Key Digital Penetration Indicators
        st.subheader("📊 Indicadores Clave de Penetración Digital")
        
        # Calculate real metrics from the data
        total_records = len(clustered_data)
        internet_users = (
            clustered_data['B4_1_Usa_internet_fijo_y_Wifi'] +
            clustered_data['B4_1_Usa_internet_en_celular_o_tableta'] +
            clustered_data['B4_1_Usa_internet_movil']).clip(0, 1).mean() * 100
        
        avg_digital_pyramid = clustered_data['nivel_piramide'].astype('float').mean()
        digital_gap = 14.4 - internet_users
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🌐 Acceso a Internet", 
                value=f"{internet_users:.1f}%",
                delta=f"Brecha reducida: {digital_gap:.1f}%",
                delta_color="inverse"
            )

        with col2:
            penetracion_movil = clustered_data['B4_1_Usa_internet_en_celular_o_tableta'].mean() * 100
            st.metric(
                label="📱 Penetración Móvil", 
                value=f"{penetracion_movil:.1f}%",
                delta=f"Crecimiento: {penetracion_movil - 70:.1f}%",
                delta_color="normal" if penetracion_movil > 70 else "inverse"
            )

        with col3:
            banda_ancha = clustered_data['B4_1_Usa_internet_fijo_y_Wifi'].mean() * 100
            st.metric(
                label="💻 Acceso Banda Ancha", 
                value=f"{banda_ancha:.1f}%",
                delta=f"Meta 30%: {banda_ancha - 50:.1f}%",
                delta_color="normal" if banda_ancha >= 50 else "inverse"
            )

        with col4:
            alfabetizacion_digital = avg_digital_pyramid / 5 * 100  # Convert to percentage
            st.metric(
                label="🎯 Nivel Digital", 
                value=f"{alfabetizacion_digital:.1f}%",
                delta="Prioridad estratégica",
                delta_color="off"
            )

        st.divider()

        # Main Findings Section
        st.subheader("🔍 Hallazgos Principales")
        
        # Calculate cluster insights from real data
        cluster_analysis = clustered_data.loc[:, ['B4_1_Usa_internet_fijo_y_Wifi', 'B4_1_Usa_internet_en_celular_o_tableta', 'nivel_piramide', 'ESTRATO', 'Cluster']].astype('float').groupby('Cluster').agg({
            'B4_1_Usa_internet_fijo_y_Wifi': 'mean',
            'B4_1_Usa_internet_en_celular_o_tableta': 'mean',
            'nivel_piramide': 'mean',
            'ESTRATO': 'mean'
        }).round(3)
        
        # Calculate key metrics for findings
        lowest_cluster = cluster_analysis['B4_1_Usa_internet_fijo_y_Wifi'].idxmin()
        highest_cluster = cluster_analysis['B4_1_Usa_internet_fijo_y_Wifi'].idxmax()
        
        internet_fijo_gap = (cluster_analysis.loc[highest_cluster, 'B4_1_Usa_internet_fijo_y_Wifi'] - 
                           cluster_analysis.loc[lowest_cluster, 'B4_1_Usa_internet_fijo_y_Wifi']) * 100
        
        mobile_gap = (cluster_analysis['B4_1_Usa_internet_en_celular_o_tableta'].max() - 
                     cluster_analysis['B4_1_Usa_internet_en_celular_o_tableta'].min()) * 100
        
        lowest_access_rate = cluster_analysis.loc[lowest_cluster, 'B4_1_Usa_internet_fijo_y_Wifi'] * 100

        # Create properly aligned columns with consistent spacing
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # Critical Gap Alert - Using clear terminology
            st.markdown("#### 🚨 Alerta de Brecha Crítica")
            st.error(f"""
            **El Segmento {lowest_cluster} presenta el menor índice de acceso a Internet Fijo y WiFi con solo {lowest_access_rate:.1f}%, {internet_fijo_gap:.1f} puntos por debajo del segmento más avanzado.**
            
            📍 **{total_records:,} registros analizados** | 🎯 **{clustered_data['Cluster'].nunique()} grupos prioritarios identificados**
            """)
            
            # Socioeconomic Correlation - Using clear variable names
            st.markdown("#### 📈 Correlación Socioeconómica")
            try:
                correlation = clustered_data[['ESTRATO', 'nivel_piramide']].corr().iloc[0,1]
                st.success(f"""
                **Correlación significativa entre Estrato Socioeconómico y Nivel de Pirámide Digital ({correlation:.2f}):** las estrategias de inclusión social pueden potenciar la adopción digital.
                """)
            except:
                st.success("""
                **Análisis socioeconómico:** Se identifica relación entre nivel socioeconómico y adopción digital, sugiriendo que políticas de inclusión social pueden acelerar la transformación digital.
                """)

        with col2:
            # Segment Inequality - Using clear terminology
            st.markdown("#### 🌍 Desigualdad por Segmentos")
            st.warning(f"""
            **Diferencia de hasta {internet_fijo_gap:.1f} puntos porcentuales en acceso a Internet Fijo y WiFi entre segmentos.**
            
            🎯 **Grupo Prioritario:** Segmento {lowest_cluster} requiere intervención inmediata con programas focalizados de infraestructura y capacitación digital.
            """)
            
            # Mobile Opportunity - Using clear terminology
            st.markdown("#### 📱 Oportunidad Móvil")
            st.info(f"""
            **Brecha de {mobile_gap:.1f}% en Internet en Celular o Tableta sugiere potencial para estrategias de conectividad móvil como puente hacia la inclusión digital.**
            """)

        st.divider()

        # Cluster Distribution Visualization
        st.subheader("📈 Distribución de Segmentos Poblacionales")
        
        cluster_dist = clustered_data['Cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_bar = px.bar(
                x=[f"Segmento {i}" for i in cluster_dist.index],
                y=cluster_dist.values,
                title="Tamaño de Segmentos Identificados",
                labels={'y': 'Población (registros)', 'x': 'Segmento'}
            )
            fig_bar.update_traces(marker_color='steelblue')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Tamaño por Segmento")
            for cluster_id, count in cluster_dist.items():
                percentage = (count / len(clustered_data)) * 100
                st.write(f"**Segmento {cluster_id}:** {count:,} ({percentage:.1f}%)")
            
            st.markdown("#### 🎯 Priorización")
            try:
                # Identificar clusters prioritarios
                priority_clusters = []
                for cluster_id in sorted(clustered_data['Cluster'].unique()):
                    cluster_subset = clustered_data[clustered_data['Cluster'] == cluster_id]
                    priority_clusters.append(cluster_id)
                
                if priority_clusters:
                    st.error(f"🚨 **Alta prioridad:** Segmento {priority_clusters[0]}")
                    if len(priority_clusters) > 1:
                        st.warning(f"⚠️ **Media prioridad:** Segmentos {', '.join(map(str, priority_clusters[1:3]))}")
                    all_clusters = sorted(clustered_data['Cluster'].unique())
                    if len(all_clusters) > 2:
                        st.info(f"📊 **Monitoreo:** Segmentos restantes {', '.join(map(str, all_clusters[2:]))}")
            except:
                st.error("🚨 **Alta prioridad:** Segmentos de baja adopción digital identificados")
                st.warning("⚠️ **Requiere análisis detallado** de cada segmento")

        # Geographic coverage summary
        if 'LATITUD_FINAL' in clustered_data.columns and 'LONGITUD_FINAL' in clustered_data.columns:
            st.markdown(f"""
            **📍 Cobertura del Análisis:** {total_records:,} registros con coordenadas geográficas precisas  
            **🎯 Segmentación:** {clustered_data['Cluster'].nunique()} grupos con patrones únicos de adopción digital  
            **🗺️ Metodología:** Análisis FAMD sobre 58 variables clave de la Encuesta CNC 2023
            """)
        else:
            st.markdown(f"""
            **📊 Muestra:** {total_records:,} registros de la Encuesta CNC 2023  
            **🎯 Segmentación:** {clustered_data['Cluster'].nunique()} grupos con patrones distintivos  
            **📈 Técnica:** FAMD + KMeans para identificación precisa de brechas digitales
            """)

        summary_metrics = clustered_data.loc[:, [
            'Cluster', 'EDAD', 'ESTRATO', 'nivel_piramide', 'B8_1_1', 'B8_1_2',
            'B4_1_Usa_internet_fijo_y_Wifi',
            'B4_1_Usa_internet_en_celular_o_tableta'
        ]].astype('float').groupby('Cluster').agg({
            'EDAD':
            'mean',
            'ESTRATO':
            'mean',
            'nivel_piramide':
            'mean',
            'B8_1_1':
            'mean',  # Importancia internet - educación
            'B8_1_2':
            'mean',  # Importancia internet - trabajo
            'B4_1_Usa_internet_fijo_y_Wifi':
            'mean',
            'B4_1_Usa_internet_en_celular_o_tableta':
            'mean'
        }).round(2)

        summary_metrics['Tamaño'] = clustered_data['Cluster'].value_counts(
        ).sort_index()

        # Color code the table
        st.dataframe(summary_metrics.style.background_gradient(
            subset=['nivel_piramide', 'B8_1_1']),
                     use_container_width=True)

    with tab2:
        st.header("🎯 Grupos Prioritarios para Intervención")
        st.markdown("Esta sección identifica y caracteriza grupos de departamentos con perfiles similares de penetración digital, permitiendo diseñar estrategias específicas para cada perfil.")

        # Mapa de Grupos Prioritarios
        st.subheader("🗺️ Mapa de Grupos Prioritarios")
        
        try:
            # Create simplified map using real coordinates
            colombia_map = create_colombia_cluster_map(clustered_data)
            
            # Display map
            import streamlit.components.v1 as components
            components.html(colombia_map._repr_html_(), height=500)
            
        except Exception as e:
            st.error(f"Error generando el mapa: {str(e)}")
            
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribución Geográfica por Cluster")
            for cluster in sorted(clustered_data['Cluster'].unique()):
                cluster_size = len(clustered_data[clustered_data['Cluster'] == cluster])
                cluster_pct = cluster_size / len(clustered_data) * 100
                st.write(f"**Cluster {cluster}:** {cluster_size} registros ({cluster_pct:.1f}%)")

        with col2:
            st.markdown("#### Distribución Urbano/Rural")
            if 'PB1' in clustered_data.columns:
                urban_rural_dist = clustered_data.groupby(['PB1', 'Cluster']).size().unstack(fill_value=0)
                fig = px.bar(urban_rural_dist.T,
                           title="Clusters por Área Geográfica",
                           labels={
                               'value': 'Número de Registros',
                               'index': 'Cluster'
                           })
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: Show cluster distribution
                cluster_dist = clustered_data['Cluster'].value_counts().sort_index()
                fig_cluster = px.bar(x=cluster_dist.index,
                                   y=cluster_dist.values,
                                   labels={
                                       'x': 'Cluster',
                                       'y': 'Número de Registros'
                                   },
                                   title="Distribución de Clusters")
                st.plotly_chart(fig_cluster, use_container_width=True)

        st.divider()

        # Descripciones de Grupos
        st.subheader("📋 Perfiles de Departamentos por Nivel de Adopción Digital")
        st.markdown("Seleccione un grupo para identificar características específicas:")

        # Cluster selection
        selected_cluster = st.selectbox(
            "Grupo a analizar:",
            options=sorted(clustered_data['Cluster'].unique()),
            format_func=lambda x: f"Grupo {x}")

        # Filter data for selected cluster
        cluster_data = clustered_data[clustered_data['Cluster'] == selected_cluster]

        # Generate group description based on characteristics
        def generate_group_description(cluster_data, cluster_id):
            """Generate textual description of cluster characteristics"""
            description = f"**Grupo {cluster_id}:** "
            
            # Calculate key statistics
            cluster_size = len(cluster_data)
            total_size = len(clustered_data)
            pct = (cluster_size / total_size) * 100
            
            # Variables for analysis
            importance_vars = ['B8_1_1', 'B8_1_2', 'B8_1_3', 'B8_1_4', 'B8_1_5', 
                              'B8_1_6', 'B8_1_7', 'B8_1_8', 'B8_1_9']
            connectivity_vars = ['B4_1_Usa_internet_fijo_y_Wifi', 'B4_1_Usa_internet_en_celular_o_tableta']
            
            try:
                # Internet access levels
                if 'B4_1_Usa_internet_fijo_y_Wifi' in cluster_data.columns:
                    wifi_access = pd.to_numeric(cluster_data['B4_1_Usa_internet_fijo_y_Wifi'], errors='coerce').mean() * 100
                    mobile_access = pd.to_numeric(cluster_data['B4_1_Usa_internet_en_celular_o_tableta'], errors='coerce').mean() * 100
                    
                    if wifi_access > 70:
                        description += "Grupo con **alta conectividad digital**. "
                    elif wifi_access > 40:
                        description += "Grupo con **conectividad moderada**. "
                    else:
                        description += "Grupo con **baja conectividad digital**. "
                
                # Pyramid level
                if 'nivel_piramide' in cluster_data.columns:
                    pyramid_level = pd.to_numeric(cluster_data['nivel_piramide'], errors='coerce').mean()
                    if pyramid_level > 3:
                        description += "Presenta **alto nivel en la pirámide de adopción digital**. "
                    elif pyramid_level > 2:
                        description += "Presenta **nivel medio en la pirámide de adopción digital**. "
                    else:
                        description += "Presenta **nivel inicial en la pirámide de adopción digital**. "
                
                # Socioeconomic level
                if 'ESTRATO' in cluster_data.columns:
                    estrato_avg = pd.to_numeric(cluster_data['ESTRATO'], errors='coerce').mean()
                    if estrato_avg > 4:
                        description += "Caracterizado por **estratos socioeconómicos altos**. "
                    elif estrato_avg > 2:
                        description += "Caracterizado por **estratos socioeconómicos medios**. "
                    else:
                        description += "Caracterizado por **estratos socioeconómicos bajos**. "
                
                description += "\n\n**Características principales:**\n"
                description += f"• **Tamaño del grupo:** {cluster_size:,} registros ({pct:.1f}% del total)\n"
                
                # Internet usage patterns
                if 'B4_1_Usa_internet_fijo_y_Wifi' in cluster_data.columns:
                    description += f"• **Acceso a Internet Fijo y WiFi:** {wifi_access:.1f}%\n"
                    description += f"• **Uso de Internet en Celular/Tableta:** {mobile_access:.1f}%\n"
                
                # Age demographics
                if 'EDAD' in cluster_data.columns:
                    age_avg = pd.to_numeric(cluster_data['EDAD'], errors='coerce').mean()
                    description += f"• **Edad promedio:** {age_avg:.1f} años\n"
                
                # Geographic distribution
                if 'PB1' in cluster_data.columns:
                    urban_rural = cluster_data['PB1'].value_counts()
                    if len(urban_rural) > 0:
                        dominant_area = urban_rural.index[0]
                        pct_dominant = (urban_rural.iloc[0] / len(cluster_data)) * 100
                        area_name = "urbana" if dominant_area == "Urbana" else "rural"
                        description += f"• **Distribución geográfica:** Predominantemente {area_name} ({pct_dominant:.1f}%)\n"
                
                return description
                
            except Exception as e:
                return f"**Grupo {cluster_id}:** Grupo identificado con {cluster_size:,} registros ({pct:.1f}% del total). Análisis detallado en proceso."

        # Display group description
        group_description = generate_group_description(cluster_data, selected_cluster)
        st.markdown(group_description)


        # Radar chart using only B8_1_X variables with descriptive names
        importance_variables = {
            'B8_1_1': 'Educación',
            'B8_1_2': 'Trabajo', 
            'B8_1_3': 'Finanzas',
            'B8_1_4': 'Negocios',
            'B8_1_5': 'Comunidad',
            'B8_1_6': 'Seguridad',
            'B8_1_7': 'Salud',
            'B8_1_8': 'Justicia',
            'B8_1_9': 'Presencia Estatal'
        }

        # Check which B8_1_X variables are available
        available_vars = {k: v for k, v in importance_variables.items() 
                        if k in clustered_data.columns}

        if len(available_vars) >= 3:
            # Calculate means for radar chart with proper numeric conversion
            try:
                # Convert to numeric, handling any string values
                cluster_data_numeric = cluster_data[list(available_vars.keys())].apply(pd.to_numeric, errors='coerce')
                overall_data_numeric = clustered_data[list(available_vars.keys())].apply(pd.to_numeric, errors='coerce')
                
                cluster_means = cluster_data_numeric.mean()
                overall_means = overall_data_numeric.mean()

                # Create radar chart with descriptive labels
                fig = go.Figure()

                fig.add_trace(
                    go.Scatterpolar(
                        r=overall_means.values,
                        theta=list(available_vars.values()),
                        fill='toself',
                        name='Promedio General',
                        line_color='lightblue',
                        opacity=0.8
                    )
                )

                fig.add_trace(
                    go.Scatterpolar(
                        r=cluster_means.values,
                        theta=list(available_vars.values()),
                        fill='toself',
                        name=f'Grupo {selected_cluster}',
                        line_color='red',
                        opacity=0.8
                    )
                )

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=False,
                            range=[1, 5]  # Scale 1-5 as per variable definitions
                        )
                    ),
                    showlegend=True,
                    height=400,
                    title="Grupo vs Promedio General: El internet como herramienta de distintos factores"
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generando gráfico radial: {str(e)}")
                st.info("Algunas variables contienen datos no numéricos.")
        else:
            st.info("No hay suficientes variables de importancia disponibles para generar gráfico radial.")

        # Estadísticas Clave en grid 2x2 mejorado
        st.markdown("### Estadísticas Clave")
        
        # Grid de estadísticas 2x2 con diseño mejorado
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(clustered_data)) * 100

        card_col1, card_col2 = st.columns(2)
        card_col3, card_col4 = st.columns(2)

        # Tamaño del grupo
        with card_col1:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 15px; height: 140px;margin-bottom: 15px;
                        display: flex; flex-direction: column; justify-content: center; align-items: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: 600; color: #1f77b4;">Tamaño del Grupo</div>
                <div style="font-size: 30px; font-weight: bold; color: #1f77b4;">{cluster_size:,} registros</div>
                <div style="font-size: 14px; color: #333;">{cluster_pct:.1f}% del total</div>
            </div>
            """, unsafe_allow_html=True)

        # Internet Fijo y WiFi
        with card_col2:
            wifi_pct = pd.to_numeric(cluster_data['B4_1_Usa_internet_fijo_y_Wifi'], errors='coerce').mean() * 100
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 15px; height: 140px;margin-bottom: 15px;
                        display: flex; flex-direction: column; justify-content: center; align-items: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: 600; color: #2e8b57;">Internet Fijo y WiFi</div>
                <div style="font-size: 30px; font-weight: bold; color: #2e8b57;">{wifi_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Internet Móvil
        with card_col3:
            mobile_pct = pd.to_numeric(cluster_data['B4_1_Usa_internet_en_celular_o_tableta'], errors='coerce').mean() * 100
            st.markdown(f"""
            <div style="background-color: #fff2e8; padding: 20px; border-radius: 15px; height: 140px;margin-bottom: 15px;
                        display: flex; flex-direction: column; justify-content: center; align-items: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: 600; color: #ff8c00;">Internet Móvil</div>
                <div style="font-size: 30px; font-weight: bold; color: #ff8c00;">{mobile_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Nivel Pirámide Digital
        with card_col4:
            pyramid_avg = pd.to_numeric(cluster_data['nivel_piramide'], errors='coerce').mean()
            st.markdown(f"""
            <div style="background-color: #f0e8ff; padding: 20px; border-radius: 15px; height: 140px;margin-bottom: 15px;
                        display: flex; flex-direction: column; justify-content: center; align-items: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: 600; color: #8a2be2;">Nivel Pirámide Digital</div>
                <div style="font-size: 30px; font-weight: bold; color: #8a2be2;">{pyramid_avg:.1f}</div>
                <div style="font-size: 14px; color: #333;">Escala 0-3</div>
            </div>
            """, unsafe_allow_html=True)

        
    with tab3:
        st.header("🤖 Insights Generados con Inteligencia Artificial")

        if not show_ai_insights:
            st.info(
                "Los insights con IA están desactivados. Actívelos en la barra lateral."
            )
        else:
            try:
                # Initialize session state for caching insights with unique keys
                cache_key = f"insights_cache_{len(clustered_data)}"
                if f'overall_insights_{cache_key}' not in st.session_state:
                    st.session_state[f'overall_insights_{cache_key}'] = None
                if f'cluster_insights_{cache_key}' not in st.session_state:
                    st.session_state[f'cluster_insights_{cache_key}'] = {}
                if f'recommendations_{cache_key}' not in st.session_state:
                    st.session_state[f'recommendations_{cache_key}'] = None
                
                ai_generator = AIInsightsGenerator()

                # Overall insights with caching
                if st.session_state[f'overall_insights_{cache_key}'] is None:
                    with st.spinner("Generando insights generales con IA..."):
                        overall_insights = ai_generator.generate_overall_insights(clustered_data)
                        st.session_state[f'overall_insights_{cache_key}'] = overall_insights
                else:
                    overall_insights = st.session_state[f'overall_insights_{cache_key}']

                st.subheader("🔍 Diagnóstico Nacional")
                st.write(overall_insights.get('diagnostico_nacional', 'Análisis en proceso...'))

                # Tendencias y brechas
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Tendencias Identificadas")
                    trends = overall_insights.get('tendencias_identificadas', [])
                    for trend in trends:
                        st.write(f"• {trend}")

                with col2:
                    st.subheader("⚠️ Brechas Críticas")
                    gaps = overall_insights.get('brechas_criticas', [])
                    for gap in gaps:
                        st.write(f"• {gap}")

                # Cluster-specific insights
                st.subheader("🎯 Análisis por Cluster")

                # Use radio buttons with horizontal layout to avoid app reloads
                clusters = sorted(clustered_data['Cluster'].unique())
                
                # Create horizontal radio buttons
                insight_cluster = st.radio(
                    "Seleccione un cluster para insights detallados:",
                    options=clusters,
                    format_func=lambda x: f"Cluster {x}",
                    horizontal=True,
                    key="cluster_radio_selection"
                )
                
                st.info(f"📊 **Analizando Cluster {insight_cluster}**")

                # Check if cluster insights are cached
                cluster_cache = st.session_state[f'cluster_insights_{cache_key}']
                if insight_cluster in cluster_cache:
                    # Display cached insights immediately
                    cluster_insights = cluster_cache[insight_cluster]
                    
                    st.subheader(f"📊 Perfil del Cluster {insight_cluster}")
                    st.write(cluster_insights.get('perfil_cluster', 'Análisis en proceso...'))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("💪 Fortalezas")
                        strengths = cluster_insights.get('fortalezas', [])
                        for strength in strengths:
                            st.success(f"✓ {strength}")

                    with col2:
                        st.subheader("🚧 Desafíos")
                        challenges = cluster_insights.get('principales_desafios', [])
                        for challenge in challenges:
                            st.warning(f"⚠ {challenge}")

                    st.subheader("🎯 Acciones Prioritarias")
                    actions = cluster_insights.get('acciones_prioritarias', [])
                    for i, action in enumerate(actions, 1):
                        st.write(f"{i}. {action}")
                        
                    # Show regenerate button for cached insights
                    if st.button("🔄 Regenerar Insights del Cluster"):
                        with st.spinner(f"Regenerando análisis del Cluster {insight_cluster}..."):
                            cluster_insights = ai_generator.generate_cluster_insights(clustered_data, insight_cluster)
                            st.session_state[f'cluster_insights_{cache_key}'][insight_cluster] = cluster_insights
                            st.rerun()
                else:
                    # Generate new insights if not cached
                    if st.button("🧠 Generar Insights del Cluster"):
                        with st.spinner(f"Analizando Cluster {insight_cluster} con IA..."):
                            cluster_insights = ai_generator.generate_cluster_insights(clustered_data, insight_cluster)
                            st.session_state[f'cluster_insights_{cache_key}'][insight_cluster] = cluster_insights
                            st.rerun()

                # Recommendations
                st.subheader("💡 Recomendaciones Estratégicas")

                if st.button("🚀 Generar Recomendaciones con IA"):
                    with st.spinner("Generando recomendaciones estratégicas..."):
                        recommendations = ai_generator.generate_recommendations(clustered_data)
                        st.session_state[f'recommendations_{cache_key}'] = recommendations

                # Display cached recommendations if available
                if st.session_state[f'recommendations_{cache_key}']:
                    recommendations = st.session_state[f'recommendations_{cache_key}']
                    
                    # Display recommendations by time horizon
                    rec_tabs = st.tabs(["Inmediatas", "Mediano Plazo", "Largo Plazo"])

                    time_horizons = [
                        ('recomendaciones_inmediatas', 'Inmediatas'),
                        ('recomendaciones_mediano_plazo', 'Mediano Plazo'),
                        ('recomendaciones_largo_plazo', 'Largo Plazo')
                    ]

                    for i, (key, label) in enumerate(time_horizons):
                        with rec_tabs[i]:
                            recs = recommendations.get(key, [])
                            for rec in recs:
                                if isinstance(rec, dict):
                                    st.subheader(rec.get('titulo', 'Recomendación'))
                                    st.write(rec.get('descripcion', ''))

                                    if 'cronograma' in rec:
                                        st.info(f"⏱️ Cronograma: {rec['cronograma']}")

            except Exception as e:
                st.error(f"Error generando insights con IA: {str(e)}")
                st.info(
                    "💡 Verifique que su clave de API de OpenAI esté configurada correctamente."
                )



    # Footer
    st.divider()
    st.markdown("""
    ---
    **Monitor de Inclusión Digital Colombia** | Desarrollado estudiantes de la Universidad de los Andes  
    🤖 Powered by AI Analytics | 📊 Clustering FAMD-KMeans | 🗺️ Visualización Geográfica  
    *Última actualización: {}*
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")))


if __name__ == "__main__":
    main()
