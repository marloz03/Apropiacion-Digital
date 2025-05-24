"""
AI-Powered Insights Generation Module
Uses OpenAI GPT-4o to generate automatic insights and recommendations
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIInsightsGenerator:
    """Class to generate AI-powered insights using OpenAI GPT-4o"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"
    
    def generate_cluster_insights(self, cluster_data: pd.DataFrame, cluster_id: int) -> Dict[str, str]:
        """
        Generate AI insights for a specific cluster
        
        Args:
            cluster_data: DataFrame with clustering results
            cluster_id: ID of the cluster to analyze
            
        Returns:
            Dictionary with AI-generated insights
        """
        try:
            # Filter data for the specific cluster
            cluster_subset = cluster_data[cluster_data['Cluster'] == cluster_id]
            
            # Calculate key statistics
            stats = self._calculate_cluster_stats(cluster_subset)
            
            # Generate insights using AI
            prompt = self._create_cluster_analysis_prompt(cluster_id, stats)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto analista en inclusión digital y políticas públicas para Colombia. "
                        "Proporciona análisis profundos y recomendaciones accionables basadas en datos de clustering "
                        "de adopción digital. Responde en español de manera profesional y técnica, pero accesible para "
                        "tomadores de decisiones en gobierno."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                logger.info(f"Generated AI insights for cluster {cluster_id}")
                return result
            else:
                return self._get_fallback_insights(cluster_id)
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return self._get_fallback_insights(cluster_id)
    
    def generate_overall_insights(self, cluster_data: pd.DataFrame) -> Dict[str, str]:
        """
        Generate overall AI insights for all clusters
        
        Args:
            cluster_data: DataFrame with clustering results
            
        Returns:
            Dictionary with overall insights
        """
        try:
            # Calculate overall statistics
            overall_stats = self._calculate_overall_stats(cluster_data)
            
            prompt = self._create_overall_analysis_prompt(overall_stats)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto consultor en transformación digital y políticas de inclusión "
                        "digital para Colombia. Analiza patrones generales en datos de clustering y proporciona "
                        "recomendaciones estratégicas para cerrar la brecha digital. Responde en español con "
                        "enfoque en política pública y estrategias nacionales."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                logger.info("Generated overall AI insights")
                return result
            else:
                return self._get_fallback_overall_insights()
            
        except Exception as e:
            logger.error(f"Error generating overall insights: {str(e)}")
            return self._get_fallback_overall_insights()
    
    def generate_recommendations(self, cluster_data: pd.DataFrame, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Generate AI-powered policy recommendations
        
        Args:
            cluster_data: DataFrame with clustering results
            focus_areas: List of specific areas to focus on
            
        Returns:
            Dictionary with recommendations
        """
        try:
            # Identify priority clusters (lowest digital adoption)
            priority_analysis = self._identify_priority_clusters(cluster_data)
            
            prompt = self._create_recommendations_prompt(priority_analysis, focus_areas)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un especialista en diseño de políticas públicas para inclusión digital "
                        "en Colombia. Genera recomendaciones específicas, viables y priorizadas para cerrar la "
                        "brecha digital basándote en análisis de clustering. Incluye cronogramas, recursos necesarios "
                        "y métricas de éxito. Responde en español."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                logger.info("Generated AI recommendations")
                return result
            else:
                return self._get_fallback_recommendations()
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _calculate_cluster_stats(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key statistics for a cluster"""
        # Key variables from CNC dataset
        key_numerical_vars = [
            'ESTRATO', 'B8_1_1', 'B8_1_2', 'B8_1_3', 'B8_1_4', 'B8_1_5', 
            'B8_1_6', 'B8_1_7', 'B8_1_8', 'B8_1_9', 'B14_1', 'B26_11'
        ]
        
        key_boolean_vars = [
            'B4_1_Usa_telefono_celular', 'B4_1_Usa_internet_fijo_y_Wifi',
            'B4_1_Usa_internet_en_celular_o_tableta', 'B4_1_Usa_internet_movil',
            'B4_1_No_usa_servicio_telecomunicaciones', 'B4_2_Tiene_telefono_celular',
            'B4_2_Tiene_internet_fijo_y_Wifi', 'B4_2_Tiene_internet_en_celular_o_tableta',
            'B4_2_Tiene_internet_movil', 'B4_2_No_tiene_servicio_telecomunicaciones'
        ]
        
        key_categorical_vars = ['nivel_piramide']
        
        stats = {
            'size': len(cluster_data),
            'avg_metrics': {},
            'usage_rates': {},
            'categorical_distributions': {}
        }
        
        # Add geographic info if available
        if 'PB1' in cluster_data.columns:
            stats['areas'] = cluster_data['PB1'].value_counts().to_dict()
        
        # Calculate stats for numerical variables
        for col in key_numerical_vars:
            if col in cluster_data.columns and cluster_data[col].dtype in ['float64', 'int64']:
                stats['avg_metrics'][col] = {
                    'mean': float(cluster_data[col].mean()),
                    'std': float(cluster_data[col].std()),
                    'min': float(cluster_data[col].min()),
                    'max': float(cluster_data[col].max())
                }
        
        # Calculate usage rates for boolean variables
        for col in key_boolean_vars:
            if col in cluster_data.columns:
                if cluster_data[col].dtype == 'bool':
                    stats['usage_rates'][col] = float(cluster_data[col].mean())
                elif cluster_data[col].dtype in ['int64', 'float64']:
                    stats['usage_rates'][col] = float((cluster_data[col] == 1).mean())
        
        # Calculate distributions for categorical variables
        for col in key_categorical_vars:
            if col in cluster_data.columns:
                stats['categorical_distributions'][col] = cluster_data[col].value_counts().to_dict()
        
        return stats
    
    def _calculate_overall_stats(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics across all clusters"""
        # Key variables from CNC dataset
        key_numerical_vars = [
            'ESTRATO', 'B8_1_1', 'B8_1_2', 'B8_1_3', 'B8_1_4', 'B8_1_5', 
            'B8_1_6', 'B8_1_7', 'B8_1_8', 'B8_1_9', 'B14_1', 'B26_11'
        ]
        
        key_boolean_vars = [
            'B4_1_Usa_telefono_celular', 'B4_1_Usa_internet_fijo_y_Wifi',
            'B4_1_Usa_internet_en_celular_o_tableta', 'B4_1_Usa_internet_movil',
            'B4_2_Tiene_telefono_celular', 'B4_2_Tiene_internet_fijo_y_Wifi',
            'B4_2_Tiene_internet_en_celular_o_tableta', 'B4_2_Tiene_internet_movil'
        ]
        
        stats = {
            'total_records': len(cluster_data),
            'num_clusters': cluster_data['Cluster'].nunique(),
            'cluster_distribution': cluster_data['Cluster'].value_counts().to_dict(),
            'national_averages': {},
            'national_usage_rates': {}
        }
        
        # Calculate national averages for numerical variables
        for col in key_numerical_vars:
            if col in cluster_data.columns and cluster_data[col].dtype in ['float64', 'int64']:
                stats['national_averages'][col] = float(cluster_data[col].mean())
        
        # Calculate national usage rates for boolean variables
        for col in key_boolean_vars:
            if col in cluster_data.columns:
                if cluster_data[col].dtype == 'bool':
                    stats['national_usage_rates'][col] = float(cluster_data[col].mean())
                elif cluster_data[col].dtype in ['int64', 'float64']:
                    stats['national_usage_rates'][col] = float((cluster_data[col] == 1).mean())
        
        return stats
    
    def _identify_priority_clusters(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify priority clusters based on digital adoption levels"""
        # Use a representative digital adoption variable if available
        adoption_col = None
        for col in cluster_data.columns:
            if col.startswith('B8_1_') and cluster_data[col].dtype in ['float64', 'int64']:
                adoption_col = col
                break
        
        if adoption_col:
            cluster_scores = cluster_data.groupby('Cluster')[adoption_col].mean().sort_values()
            return {
                'lowest_adoption_cluster': int(cluster_scores.index[0]),
                'highest_adoption_cluster': int(cluster_scores.index[-1]),
                'cluster_scores': cluster_scores.to_dict(),
                'digital_divide_gap': float(cluster_scores.iloc[-1] - cluster_scores.iloc[0])
            }
        else:
            # Fallback if no suitable column found
            cluster_counts = cluster_data['Cluster'].value_counts()
            return {
                'lowest_adoption_cluster': int(cluster_counts.index[-1]),  # Smallest cluster
                'highest_adoption_cluster': int(cluster_counts.index[0]),  # Largest cluster
                'cluster_scores': cluster_counts.to_dict(),
                'digital_divide_gap': 0.0
            }
    
    def _create_cluster_analysis_prompt(self, cluster_id: int, stats: Dict[str, Any]) -> str:
        """Create prompt for cluster-specific analysis"""
        # Build geographic info if available
        geo_info = ""
        if 'areas' in stats:
            geo_info = f"- Distribución geográfica: {json.dumps(stats['areas'])}"
        
        # Build usage rates info if available
        usage_info = ""
        if 'usage_rates' in stats and stats['usage_rates']:
            usage_info = f"- Tasas de uso de tecnología: {json.dumps(stats['usage_rates'], indent=2)}"
        
        # Build categorical distributions if available
        categorical_info = ""
        if 'categorical_distributions' in stats and stats['categorical_distributions']:
            categorical_info = f"- Distribuciones categóricas: {json.dumps(stats['categorical_distributions'])}"
        
        return f"""
        Analiza el siguiente cluster de adopción digital en Colombia:

        CLUSTER {cluster_id}:
        - Tamaño: {stats['size']} registros
        {geo_info}
        - Métricas promedio: {json.dumps(stats['avg_metrics'], indent=2)}
        {usage_info}
        {categorical_info}

        Proporciona un análisis en formato JSON con las siguientes claves:
        {{
            "perfil_cluster": "Descripción del perfil característico de este cluster",
            "nivel_adopcion_digital": "Alto/Medio/Bajo con justificación",
            "principales_desafios": ["Lista de principales desafíos identificados"],
            "fortalezas": ["Lista de fortalezas del cluster"],
            "acciones_prioritarias": ["Lista de 3-5 acciones prioritarias específicas"],
            "indicadores_clave": "Métricas más relevantes para monitoreo",
            "poblacion_objetivo": "Descripción de la población objetivo de este cluster"
        }}
        """
    
    def _create_overall_analysis_prompt(self, stats: Dict[str, Any]) -> str:
        """Create prompt for overall analysis"""
        # Build usage rates info if available
        usage_rates_info = ""
        if 'national_usage_rates' in stats and stats['national_usage_rates']:
            usage_rates_info = f"- Tasas nacionales de uso de tecnología: {json.dumps(stats['national_usage_rates'], indent=2)}"
        
        return f"""
        Analiza el panorama general de adopción digital en Colombia basado en clustering:

        ESTADÍSTICAS GENERALES:
        - Total de registros: {stats['total_records']}
        - Número de clusters identificados: {stats['num_clusters']}
        - Distribución por cluster: {json.dumps(stats['cluster_distribution'])}
        - Promedios nacionales: {json.dumps(stats['national_averages'], indent=2)}
        {usage_rates_info}

        Proporciona un análisis estratégico en formato JSON:
        {{
            "diagnostico_nacional": "Evaluación general del estado de la digitalización",
            "tendencias_identificadas": ["Lista de tendencias principales"],
            "brechas_criticas": ["Brechas más urgentes identificadas"],
            "oportunidades_estrategicas": ["Oportunidades de mayor impacto"],
            "recomendaciones_transversales": ["Recomendaciones que aplican a todos los clusters"],
            "priorizacion_geografica": "Sugerencias de priorización geográfica",
            "impacto_estimado": "Estimación del impacto potencial de las intervenciones"
        }}
        """
    
    def _create_recommendations_prompt(self, priority_analysis: Dict[str, Any], focus_areas: List[str] = None) -> str:
        """Create prompt for generating recommendations"""
        focus_text = f"Enfócate especialmente en: {', '.join(focus_areas)}" if focus_areas else ""
        
        return f"""
        Genera recomendaciones específicas de política pública para Colombia basadas en:

        ANÁLISIS DE PRIORIDADES:
        {json.dumps(priority_analysis, indent=2)}

        {focus_text}

        Proporciona recomendaciones detalladas en formato JSON:
        {{
            "recomendaciones_inmediatas": [
                {{
                    "titulo": "Nombre de la recomendación",
                    "descripcion": "Descripción detallada",
                    "recursos_necesarios": "Descripción de recursos",
                    "cronograma": "Tiempo estimado de implementación",
                    "indicadores_exito": ["Métricas para medir éxito"],
                    "responsables": ["Entidades responsables"]
                }}
            ],
            "recomendaciones_mediano_plazo": [...],
            "recomendaciones_largo_plazo": [...],
            "presupuesto_estimado": "Estimación de inversión necesaria",
            "marco_legal_necesario": "Cambios normativos requeridos",
            "alianzas_estrategicas": ["Socios clave para implementación"]
        }}
        """
    
    def _get_fallback_insights(self, cluster_id: int) -> Dict[str, Any]:
        """Provide fallback insights when AI is not available"""
        return {
            "perfil_cluster": f"Cluster {cluster_id} requiere análisis detallado de características de adopción digital",
            "nivel_adopcion_digital": "Pendiente de análisis con IA",
            "principales_desafios": ["Conectividad", "Alfabetización digital", "Acceso a dispositivos"],
            "acciones_prioritarias": ["Mejorar infraestructura", "Programas de capacitación", "Subsidios para acceso"],
            "indicadores_clave": "Tasa de acceso a internet, nivel de alfabetización digital"
        }
    
    def _get_fallback_overall_insights(self) -> Dict[str, Any]:
        """Provide fallback overall insights"""
        return {
            "diagnostico_nacional": "Se requiere análisis con IA para diagnóstico completo",
            "tendencias_identificadas": ["Brecha urbano-rural", "Diferencias generacionales"],
            "brechas_criticas": ["Acceso en zonas rurales", "Habilidades digitales"],
            "recomendaciones_transversales": ["Infraestructura", "Educación", "Políticas de inclusión"]
        }
    
    def _get_fallback_recommendations(self) -> Dict[str, Any]:
        """Provide fallback recommendations"""
        return {
            "recomendaciones_inmediatas": [
                {
                    "titulo": "Mejora de conectividad básica",
                    "descripcion": "Expansión de infraestructura de telecomunicaciones",
                    "cronograma": "12-18 meses"
                }
            ],
            "presupuesto_estimado": "Requiere análisis detallado con IA"
        }