"""
Simplified Colombia Geographic Visualization Module
Works directly with LATITUD_FINAL and LONGITUD_FINAL from real survey data
"""
import pandas as pd
import folium
import numpy as np

def create_colombia_cluster_map(data: pd.DataFrame) -> folium.Map:
    """
    Create a simple map of Colombia showing cluster distribution using real coordinates
    
    Args:
        data: DataFrame with LATITUD_FINAL, LONGITUD_FINAL and Cluster columns
        
    Returns:
        Folium map object
    """
    # Center map on Colombia
    colombia_center = [4.5709, -74.2973]  # Bogotá coordinates
    m = folium.Map(location=colombia_center, zoom_start=6)
    
    # Define colors for clusters
    cluster_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    # Add points for each record with coordinates
    for idx, row in data.iterrows():
        if pd.isna(row.get('LATITUD_FINAL')) or pd.isna(row.get('LONGITUD_FINAL')):
            continue
            
        lat = row['LATITUD_FINAL']
        lon = row['LONGITUD_FINAL']
        cluster = int(row['Cluster'])
        
        # Get cluster color
        color = cluster_colors[cluster % len(cluster_colors)]
        
        # Create popup with cluster info
        popup_text = f"""
        <div style="width:200px">
            <b>Cluster:</b> {cluster}<br>
            <b>Área:</b> {row.get('PB1', 'N/A')}<br>
            <b>Estrato:</b> {row.get('ESTRATO', 'N/A')}<br>
            <b>Nivel Pirámide:</b> {row.get('nivel_piramide', 'N/A')}<br>
            <b>Edad:</b> {row.get('EDAD', 'N/A')}
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, -lon],
            radius=6,
            popup=popup_text,
            color=color,
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    # Add improved legend with better positioning and responsive design
    unique_clusters = sorted(data['Cluster'].unique())
    num_clusters = len(unique_clusters)
    
    # Calculate dynamic height based on number of clusters
    legend_height = max(120, 30 + (num_clusters * 25))
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 60px; right: 20px; width: 160px; height: {legend_height}px; 
                background-color: rgba(255, 255, 255, 0.98); 
                border: 1px solid #ddd; 
                border-radius: 6px;
                z-index: 1000; 
                font-size: 8px; 
                padding: 6px; 
                box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                overflow: hidden;">
    <div style="margin-bottom: 6px; font-weight: bold; color: #333; border-bottom: 1px solid #eee; padding-bottom: 3px; font-size: 12px;">
        📊 Segmentos
    </div>
    '''
    
    for cluster in unique_clusters:
        color = cluster_colors[cluster % len(cluster_colors)]
        count = len(data[data['Cluster'] == cluster])
        legend_html += f'''
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <span style="color: {color}; font-size: 10px; margin-right: 8px;">●</span>
            <span style="color: #333; font-size: 11px;">Cluster {cluster} ({count:,})</span>
        </div>
        '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m