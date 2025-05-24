"""
FAMD Clustering Implementation - Metodología de Marlon
Implementación exacta del enfoque FAMD del notebook de clustering
"""
import pandas as pd
import numpy as np
import prince
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from typing import Dict, Any, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAMDClustering:
    """
    Implementación exacta de la metodología FAMD de Marlon
    Basada en el notebook CLUSTERING MARLON.ipynb
    """
    
    def __init__(self):
        self.famd_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.processed_data = None
        
    def load_and_preprocess_survey_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga y preprocesa los datos de la encuesta siguiendo la metodología de Marlon
        
        Args:
            file_path: Ruta al archivo Excel de la encuesta
            
        Returns:
            DataFrame procesado listo para FAMD
        """
        logger.info("Cargando datos de encuesta CNC 2023...")
        
        # Cargar datos originales
        df_encuesta = pd.read_excel(file_path)
        logger.info(f"Datos cargados: {len(df_encuesta)} registros, {len(df_encuesta.columns)} columnas")
        
        # Selección de variables según la metodología de Marlon
        # Estas son las 296 variables seleccionadas en el notebook
        selected_columns = self._get_selected_variables(df_encuesta)
        
        df_encuesta2 = df_encuesta[selected_columns].copy()
        logger.info(f"Variables seleccionadas: {len(df_encuesta2.columns)} de {len(df_encuesta.columns)}")
        
        # Análisis de completitud y limpieza según metodología
        df_clean = self._apply_completeness_filter(df_encuesta2)
        
        # Preparar datos para FAMD
        df_famd_ready = self._prepare_for_famd(df_clean)
        
        self.processed_data = df_famd_ready
        logger.info(f"Datos procesados para FAMD: {df_famd_ready.shape}")
        
        return df_famd_ready
    
    def _get_selected_variables(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las variables de interés específicas según la metodología de Marlon
        
        Args:
            df: DataFrame original
            
        Returns:
            Lista de nombres de columnas seleccionadas
        """
        # Variables específicas de interés según metodología de Marlon
        selected_variables = [
            'REGISTRO', 'N_ENCUESTA', 'REGIONAL', 'PB1', 'SECTOR', 'REGION',
            'MUNICIPIO', 'PDET', 'PERSONAS', 'GENERO', 'PERSONAS_GEN',
            'PERSONA_SELECCIONADA', 'EDAD', 'REDAD', 'B3_EDAD_1', 'B3_2', 'ST_DEC',
            'ESTRATO', 'ST_GR', 'B4_1', 'B4_2', 'B8_1_1', 'B8_1_2', 'B8_1_3',
            'B8_1_4', 'B8_1_5', 'B8_1_6', 'B8_1_7', 'B8_1_8', 'B8_1_9', 'B9_1',
            'B9_7', 'B10_11_2', 'B10_11_3', 'B10_11_4', 'B10_11_5', 'B10_11_6',
            'B10_11_7', 'B10_11_8', 'B10_11_9', 'B10_11_10', 'B10_11_19',
            'B10_11_20', 'B10_11_21', 'B10_11_22', 'B10_11_24', 'B10_11_25',
            'B13_1', 'B13_2', 'B14_1', 'B14_2', 'ESTRATO_B26_1', 'B26_2', 'B26_11',
            'B26_12', 'LATITUD_FINAL', 'LONGITUD_FINAL', 'nivel_piramide'
        ]
        
        # Filtrar solo las variables que existen en el DataFrame
        existing_variables = [col for col in selected_variables if col in df.columns]
        
        logger.info(f"Variables de interés seleccionadas: {len(existing_variables)} de {len(selected_variables)} especificadas")
        return existing_variables
    
    def _apply_completeness_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica filtro de completitud según la metodología de Marlon
        
        Args:
            df: DataFrame con variables seleccionadas
            
        Returns:
            DataFrame filtrado por completitud
        """
        logger.info("Aplicando filtro de completitud...")
        
        # Calcular porcentaje de completitud por columna
        completitud = df.notna().mean() * 100
        
        # Filtrar columnas con completitud >= 70% (según metodología de Marlon)
        columnas_buena_completitud = completitud[completitud >= 70].index.tolist()
        
        # Mantener siempre las variables clave
        variables_clave = ['REGISTRO', 'REGIONAL', 'MUNICIPIO', 'GENERO', 
                          'LATITUD_FINAL', 'LONGITUD_FINAL', 'nivel_piramide']
        
        for var in variables_clave:
            if var in df.columns and var not in columnas_buena_completitud:
                columnas_buena_completitud.append(var)
        
        df_filtered = df[columnas_buena_completitud].copy()
        
        logger.info(f"Columnas después del filtro de completitud: {len(df_filtered.columns)}")
        
        # Aplicar filtros adicionales de calidad
        # Eliminar filas con muchos valores faltantes
        threshold_rows = 0.7  # Al menos 70% de datos no faltantes por fila
        df_filtered = df_filtered.dropna(thresh=int(threshold_rows * len(df_filtered.columns)))
        
        logger.info(f"Registros después del filtro: {len(df_filtered)}")
        
        return df_filtered
    
    def _prepare_for_famd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos para FAMD siguiendo exactamente la metodología de Marlon
        
        Args:
            df: DataFrame filtrado
            
        Returns:
            DataFrame preparado para reducción de dimensionalidad
        """
        logger.info("Aplicando limpieza y feature engineering para reducción de dimensionalidad...")
        
        # Guardar coordenadas geográficas antes de eliminarlas para clustering
        self.geographic_data = None
        if 'LATITUD_FINAL' in df.columns and 'LONGITUD_FINAL' in df.columns:
            self.geographic_data = df[['LATITUD_FINAL', 'LONGITUD_FINAL']].copy()
        
        # Primero selecciono solo las variables que serán de interés para el clustering
        df_red_dim = df.drop(columns=['REGISTRO', 'N_ENCUESTA', 'REDAD', 'B3_EDAD_1', 'ST_DEC', 
                                     'ST_GR', 'PERSONAS_GEN', 'PERSONA_SELECCIONADA', 
                                     'ESTRATO_B26_1', 'LATITUD_FINAL', 'LONGITUD_FINAL'], errors='ignore')
        
        # Elimino variables con muchas categorías
        df_red_dim = df_red_dim.drop(columns=['SECTOR', 'MUNICIPIO'], errors='ignore')
        
        # Elimino variables redundantes o poco interesantes
        drop_cols = ['REGIONAL', 'REGION', 'B9_1', 'B9_7', 'B13_1', 'B10_11_2', 'B10_11_3', 
                    'B10_11_4', 'B10_11_5', 'B10_11_6', 'B10_11_7', 'B10_11_8', 'B10_11_9', 
                    'B10_11_10', 'B10_11_19', 'B10_11_20', 'B10_11_21', 'B10_11_22', 
                    'B10_11_24', 'B10_11_25', 'B3_2', 'B14_2', 'B13_2', 'B26_2', 'B26_12']
        df_red_dim = df_red_dim.drop(columns=drop_cols, errors='ignore')
        
        # Transformo en categorías con sus respectivas descripciones
        
        # Tipo población
        if 'PB1' in df_red_dim.columns:
            mapping_pb1 = {1: "Urbana", 2: "Rural"}
            df_red_dim["PB1"] = df_red_dim["PB1"].map(mapping_pb1)
        
        # Nivel educativo propio
        if 'B26_11' in df_red_dim.columns:
            mapping_B26_11 = {1: "Primaria", 2: "Bachillerato", 3: "Técnico", 4: "Tecnológico", 
                              5: "Universitario", 6: "Postgrado", 7: "Doctorado", 8: 'Ninguno', 9: 'No responde'}
            df_red_dim["B26_11"] = df_red_dim["B26_11"].map(mapping_B26_11)
        
        # Transformo a categoría el resto de variables categóricas
        categorical_vars = ['nivel_piramide', 'GENERO']
        for var in categorical_vars:
            if var in df_red_dim.columns:
                df_red_dim[var] = df_red_dim[var].astype(str)
        
        # Transformo las numéricas de opción múltiple
        # B4_1 (cantidad de servicios de internet que usa)
        if 'B4_1' in df_red_dim.columns:
            dict_b4_1 = {1: "Usa teléfono celular", 2: "Usa internet fijo y Wifi", 
                        3: "Usa internet en celular o tableta", 4: "Usa internet móvil", 
                        88: "No usa servicio telecomunicaciones"}
            b4_1_dummies = self._expand_multichoice(df, 'B4_1', dict_b4_1)
            df_red_dim = df_red_dim.drop(columns=['B4_1'], errors='ignore')
            df_red_dim = df_red_dim.join(b4_1_dummies)
        
        # B4_2 (cantidad de servicios de internet que tiene)
        if 'B4_2' in df_red_dim.columns:
            dict_b4_2 = {1: "Tiene teléfono celular", 2: "Tiene internet fijo y Wifi", 
                        3: "Tiene internet en celular o tableta", 4: "Tiene internet móvil", 
                        88: "No tiene servicio telecomunicaciones"}
            b4_2_dummies = self._expand_multichoice(df, 'B4_2', dict_b4_2)
            df_red_dim = df_red_dim.drop(columns=['B4_2'], errors='ignore')
            df_red_dim = df_red_dim.join(b4_2_dummies)
        
        # Transformo los enteros a float
        num_cols = df_red_dim.select_dtypes(include=['int', 'float']).columns
        df_red_dim[num_cols] = df_red_dim[num_cols].astype(float)
        
        # Identificar tipos de variables para FAMD
        numerical_cols = list(df_red_dim.select_dtypes(include=['int', 'float']).columns)
        categorical_cols = list(df_red_dim.select_dtypes(include=['object']).columns)
        
        # Tratar valores faltantes
        # Para variables categóricas, crear categoría "No_Respuesta"
        for col in categorical_cols:
            df_red_dim[col] = df_red_dim[col].fillna('No_Respuesta')
        
        # Para variables numéricas, usar mediana
        for col in numerical_cols:
            df_red_dim[col] = df_red_dim[col].fillna(df_red_dim[col].median())
        
        logger.info(f"Variables numéricas: {len(numerical_cols)}")
        logger.info(f"Variables categóricas: {len(categorical_cols)}")
        logger.info(f"Forma final del dataset para FAMD: {df_red_dim.shape}")
        
        # Guardar información de tipos de variables
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        return df_red_dim
    
    def _expand_multichoice(self, df: pd.DataFrame, column: str, mapping_dict: Dict) -> pd.DataFrame:
        """
        Expande variables de opción múltiple en variables dummy
        
        Args:
            df: DataFrame original
            column: Nombre de la columna a expandir
            mapping_dict: Diccionario de mapeo de valores
            
        Returns:
            DataFrame con variables dummy
        """
        if column not in df.columns:
            return pd.DataFrame()
        
        # Crear DataFrame dummy para cada opción
        dummy_data = {}
        
        for key, value in mapping_dict.items():
            # Crear nombre de columna limpio
            col_name = f"{column}_{value.replace(' ', '_').replace(',', '').replace('é', 'e').replace('ó', 'o')}"
            
            # Crear variable dummy (1 si contiene ese valor, 0 si no)
            dummy_data[col_name] = (df[column] == key).astype(int)
        
        return pd.DataFrame(dummy_data, index=df.index)
    
    def perform_famd_analysis(self, df: pd.DataFrame, n_components: int = 10) -> Dict[str, Any]:
        """
        Ejecuta el análisis FAMD siguiendo la metodología de Marlon
        
        Args:
            df: DataFrame preparado para FAMD
            n_components: Número de componentes FAMD
            
        Returns:
            Diccionario con resultados FAMD
        """
        logger.info(f"Ejecutando FAMD con {n_components} componentes...")
        
        try:
            # Crear y ajustar modelo FAMD
            self.famd_model = prince.FAMD(
                n_components=n_components,
                n_iter=10,
                copy=True,
                check_input=True,
                engine='sklearn'
            )
            
            # Ajustar el modelo
            self.famd_model.fit(df)
            
            # Transformar datos
            coordinates = self.famd_model.transform(df)
            
            # Obtener información de componentes
            summary_df = self.famd_model.eigenvalues_summary
            eigenvalues = summary_df["eigenvalue"].values
            explained_variance = summary_df["% of variance"].values
            explained_variance = np.array([float(p.strip('%')) for p in explained_variance])
            cumulative_variance_total = summary_df["% of variance (cumulative)"].iloc[-1]
            
            # Resultados
            famd_results = {
                'model': self.famd_model,
                'coordinates': coordinates,
                'eigenvalues': eigenvalues,
                'explained_variance': explained_variance,
                'n_components': n_components
            }
            
            logger.info(f"FAMD completado. Varianza explicada total: {explained_variance.sum():.3f}")
            logger.info(f"Primeras 3 componentes explican: {explained_variance[:3].sum():.3f}")
            
            return famd_results
            
        except Exception as e:
            logger.error(f"Error en FAMD: {str(e)}")
            raise
    
    
    
    def perform_clustering(self, coordinates: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """
        Ejecuta clustering final con KMeans según metodología FAMD de Marlon
        
        Args:
            coordinates: Coordenadas FAMD
            n_clusters: Número de clusters
            
        Returns:
            Diccionario con resultados de clustering
        """
        logger.info(f"Ejecutando clustering final KMeans con {n_clusters} clusters...")
        
        # Usar KMeans como en la metodología de Marlon
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
        final_silhouette = silhouette_score(coordinates, cluster_labels)
        
        self.clustering_model = kmeans
        logger.info(f"KMeans completado (Silhouette: {final_silhouette:.3f})")
        
        # Crear DataFrame con resultados
        result_data = self.processed_data.copy()
        result_data['Cluster'] = cluster_labels
        
        # Agregar coordenadas geográficas de vuelta si están disponibles
        if self.geographic_data is not None:
            result_data = result_data.join(self.geographic_data, how='left')
        
        # Agregar coordenadas FAMD (corregir indexación)
        coordinates_df = pd.DataFrame(coordinates, index=result_data.index)
        for i in range(min(5, coordinates.shape[1])):
            result_data[f'FAMD_Dim_{i+1}'] = coordinates_df.iloc[:, i]
        
        clustering_results = {
            'cluster_data': result_data,
            'cluster_labels': cluster_labels,
            'silhouette_score': final_silhouette,
            'n_clusters': n_clusters,
            'model': self.clustering_model,
            'coordinates': coordinates
        }
        
        logger.info(f"Clustering KMeans completado. Silhouette final: {final_silhouette:.3f}")
        
        return clustering_results
    
    def generate_cluster_profiles(self, cluster_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Genera perfiles detallados de cada cluster
        
        Args:
            cluster_data: DataFrame con asignaciones de cluster
            
        Returns:
            Diccionario con perfiles de cluster
        """
        logger.info("Generando perfiles de clusters...")
        
        profiles = {}

        
        # Variables numéricas para análisis
        numerical_vars = [col for col in self.numerical_cols if col in cluster_data.columns]
        categorical_vars = [col for col in self.categorical_cols if col in cluster_data.columns]
        
        for cluster_id in sorted(cluster_data['Cluster'].unique()):
            cluster_subset = cluster_data[cluster_data['Cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_subset),
                'percentage': len(cluster_subset) / len(cluster_data) * 100,
                'numerical_stats': {},
                'categorical_stats': {},
                'geographic_distribution': {}
            }
            
            # Estadísticas de variables numéricas
            for var in numerical_vars:
                if var in cluster_subset.columns:
                    profile['numerical_stats'][var] = {
                        'mean': float(cluster_subset[var].mean()),
                        'std': float(cluster_subset[var].std()),
                        'median': float(cluster_subset[var].median()),
                        'min': float(cluster_subset[var].min()),
                        'max': float(cluster_subset[var].max())
                    }
            
            # Distribuciones de variables categóricas
            for var in categorical_vars[:10]:  # Limitar a 10 variables más importantes
                if var in cluster_subset.columns:
                    value_counts = cluster_subset[var].value_counts()
                    profile['categorical_stats'][var] = value_counts.head().to_dict()
            
            # Distribución geográfica
            if 'REGIONAL' in cluster_subset.columns:
                regional_dist = cluster_subset['REGIONAL'].value_counts()
                profile['geographic_distribution']['regional'] = regional_dist.to_dict()
            
            if 'MUNICIPIO' in cluster_subset.columns:
                municipio_dist = cluster_subset['MUNICIPIO'].value_counts()
                profile['geographic_distribution']['municipio'] = municipio_dist.head().to_dict()
            
            # Nivel de pirámide digital
            if 'nivel_piramide' in cluster_subset.columns:
                piramide_dist = cluster_subset['nivel_piramide'].value_counts()
                profile['digital_level'] = piramide_dist.to_dict()
                
                # Calcular nivel promedio de adopción digital
                avg_level = cluster_subset['nivel_piramide'].astype('float').mean()
                if avg_level >= 2.5:
                    profile['adoption_level'] = 'Alto'
                elif avg_level >= 1.5:
                    profile['adoption_level'] = 'Medio'
                else:
                    profile['adoption_level'] = 'Bajo'
            
            profiles[cluster_id] = profile
        
        logger.info(f"Perfiles generados para {len(profiles)} clusters")
        return profiles
    
    def run_complete_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo siguiendo la metodología de Marlon
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            Diccionario con todos los resultados
        """
        logger.info("Iniciando análisis completo FAMD-Clustering...")
        
        # 1. Cargar y preprocesar datos
        df_processed = self.load_and_preprocess_survey_data(file_path)
        
        # 2. Ejecutar FAMD
        famd_results = self.perform_famd_analysis(df_processed, n_components=18)
        
        # 3. Ejecutar clustering final
        optimal_clusters = 7
        clustering_results = self.perform_clustering(famd_results['coordinates'], optimal_clusters)
        
        # 4. Generar perfiles de clusters
        cluster_profiles = self.generate_cluster_profiles(clustering_results['cluster_data'])
        
        # Compilar resultados completos
        complete_results = {
            'preprocessing': {
                'original_shape': df_processed.shape,
                'numerical_vars': self.numerical_cols,
                'categorical_vars': self.categorical_cols
            },
            'famd_results': famd_results,
            'clustering_results': clustering_results,
            'cluster_profiles': cluster_profiles,
            'optimal_n_clusters': optimal_clusters,
            'methodology': 'FAMD-Clustering-Marlon'
        }
        
        logger.info("Análisis completo terminado exitosamente")
        
        return complete_results