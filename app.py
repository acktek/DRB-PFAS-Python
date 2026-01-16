################################################################################
#### DELAWARE RIVER BASIN PFAS APP - COMPLETE PYTHON/STREAMLIT PORT
################################################################################

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import Point

# Page configuration
st.set_page_config(
    page_title="PFAS in the Delaware River Basin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.8);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #1e3c72;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .note-box {
        background: rgba(255,255,255,0.9);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess all PFAS data"""

    def simplify_data(df, tissue=False):
        """Simplify and aggregate data"""

        def map_agency(org_name):
            if pd.isna(org_name):
                return "Other"
            org_name = str(org_name)
            if "DRBC" in org_name or "Delaware River Basin Commission" in org_name:
                return "DRBC"
            elif "USGS" in org_name or "U.S. Geological Survey" in org_name:
                return "USGS"
            elif "USEPA" in org_name:
                return "USEPA"
            elif "NJDEP" in org_name or "New Jersey Department of Environmental Protection" in org_name:
                return "NJDEP"
            else:
                return "Other"

        if tissue:
            simple_df = pd.DataFrame({
                'agency': df['OrganizationFormalName'].apply(map_agency),
                'loc': df['Location.Name'],
                'lat': df['Latitude'].round(3),
                'lon': df['Longitude'].round(3),
                'yr': df['Year'],
                'conc': df['Result.Measure.Value..ppt.'],
                'chem': df['PFAS.Chemical.Name'],
                'abbrev': df['PFAS.abbrev'],
                'group': df['PFAS.group'],
                'species': df['Fish.Species.Common']
            })
            cols = ['agency', 'loc', 'yr', 'conc', 'chem', 'abbrev', 'group', 'species']
            simple_df = simple_df.drop_duplicates(subset=cols)
            raw = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'abbrev', 'group', 'agency', 'species'])['conc'].mean().reset_index()
            grouped = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'group', 'agency', 'species'])['conc'].sum().reset_index()
            all_pfas = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'agency', 'species'])['conc'].sum().reset_index()
        else:
            simple_df = pd.DataFrame({
                'agency': df['OrganizationFormalName'].apply(map_agency),
                'loc': df['Location.Name'],
                'lat': df['Latitude'].round(3),
                'lon': df['Longitude'].round(3),
                'yr': df['Year'],
                'conc': df['Result.Measure.Value..ppt.'],
                'chem': df['PFAS.Chemical.Name'],
                'abbrev': df['PFAS.abbrev'],
                'group': df['PFAS.group']
            })
            cols = ['agency', 'loc', 'yr', 'conc', 'chem', 'abbrev', 'group']
            simple_df = simple_df.drop_duplicates(subset=cols)
            raw = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'abbrev', 'group', 'agency'])['conc'].mean().reset_index()
            grouped = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'group', 'agency'])['conc'].sum().reset_index()
            all_pfas = simple_df.groupby(['loc', 'lat', 'lon', 'yr', 'agency'])['conc'].sum().reset_index()

        return {'raw': raw, 'grouped': grouped, 'all': all_pfas}

    data_path = Path(__file__).parent
    water_df = pd.read_csv(data_path / "PFAS_water_data_DRB_20260112.csv")
    gw_df = pd.read_csv(data_path / "PFAS_ground_water_data_DRB_20260112.csv")
    sed_df = pd.read_csv(data_path / "PFAS_sediment_data_DRB_20260112.csv")
    tissue_df = pd.read_csv(data_path / "PFAS_tissue_data_DRB_20260112.csv")

    sed_df['Result.Measure.Value..ppt.'] = sed_df['Result.Measure.Value..ppt.'] / 1000
    tissue_df['Result.Measure.Value..ppt.'] = tissue_df['Result.Measure.Value..ppt.'] / 1000

    return {
        'Water': simplify_data(water_df),
        'GW': simplify_data(gw_df),
        'Sediment': simplify_data(sed_df),
        'Tissue': simplify_data(tissue_df, tissue=True)
    }

@st.cache_data
def load_spatial_data():
    """Load spatial/shapefile data"""
    data_path = Path(__file__).parent
    drbbnd = gpd.read_file(data_path / "drb_bnd_arc.shp").to_crs(epsg=4326)
    huc = gpd.read_file(data_path / "drbhuc12.shp").to_crs(epsg=4326)
    rm = gpd.read_file(data_path / "delrivRM10th.shp").to_crs(epsg=4326)
    delrivbay = gpd.read_file(data_path / "delrivbay.shp").to_crs(epsg=4326)

    return {'drbbnd': drbbnd, 'huc': huc, 'rm': rm, 'delrivbay': delrivbay}

# Legend configuration
LEGEND_CONFIG = {
    'Water': {
        'breaks': [0, 1, 2, 5, 10, 25, 50, 100, 1000, np.inf],
        'labels': ["B.D. - 1", ">1 - 2", ">2 - 5", ">5 - 10", ">10 - 25",
                   ">25 - 50", ">50 - 100", ">100 - 1000", ">1000"],
        'colors': ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde724']
    },
    'GW': {
        'breaks': [0, 1, 2, 5, 10, 25, 50, 100, 1000, np.inf],
        'labels': ["B.D. - 1", ">1 - 2", ">2 - 5", ">5 - 10", ">10 - 25",
                   ">25 - 50", ">50 - 100", ">100 - 1000", ">1000"],
        'colors': ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde724']
    },
    'Sediment': {
        'breaks': [0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.50, 5.00, np.inf],
        'labels': ["B.D. - 0.01", ">0.01 - 0.05", ">0.05 - 0.10", ">0.10 - 0.25",
                   ">0.25 - 0.50", ">0.50 - 1.00", ">1.00 - 2.50", ">2.50 - 5.00", ">5.00"],
        'colors': ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde724']
    },
    'Tissue': {
        'breaks': [0, 0.10, 0.50, 1.00, 5.00, 10.0, 25.0, 50.0, 100.0, np.inf],
        'labels': ["B.D. - 0.10", ">0.10 - 0.50", ">0.50 - 1.00", ">1.00 - 5.00",
                   ">5.00 - 10.0", ">10.0 - 25.0", ">25.0 - 50.0", ">50.0 - 100.0", ">100.0"],
        'colors': ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde724']
    }
}

def get_color_for_concentration(conc, dataset_key):
    """Get color based on concentration"""
    breaks = LEGEND_CONFIG[dataset_key]['breaks']
    colors = LEGEND_CONFIG[dataset_key]['colors']

    for i in range(len(breaks) - 1):
        if conc >= breaks[i] and conc < breaks[i + 1]:
            return colors[min(i, len(colors) - 1)]
    return colors[-1]

def aggregate_coords(df, method="Most Recent"):
    """Aggregate data by coordinates"""
    if df.empty:
        return df

    if method == "Most Recent":
        result = df.sort_values('yr', ascending=False).groupby(['lat', 'lon']).first().reset_index()
    elif method == "Maximum":
        result = df.groupby(['lat', 'lon'])['conc'].max().reset_index()
        result = df.groupby(['lat', 'lon']).first().reset_index().merge(result[['lat', 'lon', 'conc']], on=['lat', 'lon'], suffixes=('_drop', ''))
        result = result[[c for c in result.columns if not c.endswith('_drop')]]
    elif method == "Average":
        result = df.groupby(['lat', 'lon'])['conc'].mean().reset_index()
        result = df.groupby(['lat', 'lon']).first().reset_index().merge(result[['lat', 'lon', 'conc']], on=['lat', 'lon'], suffixes=('_drop', ''))
        result = result[[c for c in result.columns if not c.endswith('_drop')]]
    elif method == "Minimum":
        result = df.groupby(['lat', 'lon'])['conc'].min().reset_index()
        result = df.groupby(['lat', 'lon']).first().reset_index().merge(result[['lat', 'lon', 'conc']], on=['lat', 'lon'], suffixes=('_drop', ''))
        result = result[[c for c in result.columns if not c.endswith('_drop')]]
    else:
        result = df

    return result

# Load data
all_data = load_data()
spatial_data = load_spatial_data()

agencies = sorted(set().union(*[
    set(all_data[media]['raw']['agency'].unique())
    for media in all_data.keys()
]))

# HEADER
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 class="header-title">🌊 PFAS in the Delaware River Basin</h1>
    </div>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem;">
        Interactive Data Visualization & Analysis Tool
    </p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### 🔧 Filters & Controls")
    st.markdown("---")

    media_map = {
        "💧 Surface Water": "Water",
        "🏞️ Groundwater": "GW",
        "🪨 Sediment": "Sediment",
        "🐟 Tissue": "Tissue"
    }

    dataset = st.selectbox("Media Type:", list(media_map.keys()))
    dataset_key = media_map[dataset]

    data_type = st.radio("PFAS Data Type:", ["ΣPFAS", "Compounds", "Groups"])
    data_type_key = {"ΣPFAS": "all", "Compounds": "raw", "Groups": "grouped"}[data_type]

    agency_filter = st.selectbox("Agency:", ["All"] + agencies)

    current_data = all_data[dataset_key][data_type_key].copy()

    if data_type == "Compounds":
        chem_filter = st.selectbox("Compound:",
                                   sorted(all_data[dataset_key]['raw']['abbrev'].unique()))
        current_data = current_data[current_data['abbrev'] == chem_filter]
        filter_name = chem_filter
    elif data_type == "Groups":
        group_filter = st.selectbox("Group:",
                                    sorted(all_data[dataset_key]['grouped']['group'].unique()))
        current_data = current_data[current_data['group'] == group_filter]
        filter_name = group_filter
    else:
        filter_name = "ΣPFAS"

    if dataset_key == "Tissue" and 'species' in current_data.columns:
        species_list = ["All"] + sorted(current_data['species'].dropna().unique().tolist())
        species_filter = st.selectbox("Species:", species_list)
        if species_filter != "All":
            current_data = current_data[current_data['species'] == species_filter]

    if 'yr' in current_data.columns and not current_data['yr'].isna().all():
        year_min, year_max = int(current_data['yr'].min()), int(current_data['yr'].max())
        year_range = st.slider("Years:", year_min, year_max, (year_min, year_max))
        current_data = current_data[(current_data['yr'] >= year_range[0]) &
                                    (current_data['yr'] <= year_range[1])]

    if agency_filter != "All":
        current_data = current_data[current_data['agency'] == agency_filter]

    st.markdown("---")

    mapped_sample = st.selectbox(
        "Display value:",
        ["Most Recent", "Maximum", "Average", "Minimum"],
        help="For locations with multiple years"
    )

    show_huc = st.checkbox("📊 Show HUC12 Averages", value=False)
    show_rm = st.checkbox("📏 Show River Miles", value=False)

    st.markdown("---")
    st.markdown("""
    <div class="note-box">
    <strong>📌 Note:</strong><br>
    • Data averaged for multiple samples<br>
    • ΣPFAS = Sum of detected compounds<br>
    • B.D. = Below detection limits
    </div>
    """, unsafe_allow_html=True)

# DATA SUMMARY METRICS
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_samples = len(current_data)
    st.metric("Total Samples", f"{total_samples:,}")

with col2:
    unique_locations = current_data[['lat', 'lon']].drop_duplicates().shape[0]
    st.metric("Locations", f"{unique_locations:,}")

with col3:
    if 'conc' in current_data.columns:
        detected = (current_data['conc'] > 0).sum()
        st.metric("Detections", f"{detected:,}")
    else:
        st.metric("Detections", "N/A")

with col4:
    if 'yr' in current_data.columns and not current_data['yr'].isna().all():
        year_span = f"{int(current_data['yr'].min())}-{int(current_data['yr'].max())}"
        st.metric("Year Range", year_span)
    else:
        st.metric("Year Range", "N/A")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Map", "📈 Estuary Analysis", "📋 Criteria", "📚 History", "ℹ️ About"
])

with tab1:
    st.markdown('<div class="info-box"><h3 style="margin:0;">Interactive Map Visualization</h3><p style="margin:0.5rem 0 0 0;">Explore PFAS concentrations across the Delaware River Basin</p></div>', unsafe_allow_html=True)

    col_map, col_plots = st.columns([2, 1])

    with col_map:
        # Create map
        m = folium.Map(
            location=[40.61678, -75.22281],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # Add DRB boundary
        folium.GeoJson(
            spatial_data['drbbnd'],
            style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0}
        ).add_to(m)

        # Aggregate data by coordinates
        map_data = aggregate_coords(current_data, mapped_sample)

        # Add markers
        unit = "ng/g" if dataset_key in ["Sediment", "Tissue"] else "ng/L"

        for idx, row in map_data.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                conc = row['conc']
                color = get_color_for_concentration(conc, dataset_key)

                conc_text = "B.D." if conc == 0 else f"{conc:.2f} {unit}"

                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <b>{filter_name}</b><br>
                    <b>Concentration:</b> {conc_text}<br>
                    <b>Agency:</b> {row.get('agency', 'N/A')}<br>
                    <b>Year:</b> {int(row.get('yr', 0)) if pd.notna(row.get('yr')) else 'N/A'}
                </div>
                """

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=250),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=0
                ).add_to(m)

        # Display map
        st_folium(m, width=700, height=600)

    with col_plots:
        # Histogram
        st.markdown("#### Concentration Distribution")

        if not current_data.empty and 'conc' in current_data.columns:
            vals = current_data['conc'].dropna()
            vals_pos = vals[vals > 0]

            if len(vals_pos) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=vals_pos,
                    nbinsx=25,
                    marker_color='#5D6B8D',
                    name='Concentration'
                ))
                fig.update_layout(
                    xaxis_title=f"{filter_name} ({unit})",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                st.markdown(f"""
                **Statistics:**
                - N: {len(vals)}
                - B.D.: {(vals == 0).sum()}
                - Mean: {vals_pos.mean():.2f}
                - Median: {vals_pos.median():.2f}
                - Max: {vals_pos.max():.2f}
                """)
            else:
                st.info("No detections in current selection")

        # Year histogram
        st.markdown("#### Sampling Years")

        if not current_data.empty and 'yr' in current_data.columns:
            yrs = current_data['yr'].dropna()

            if len(yrs) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=yrs,
                    marker_color='#3A6B35',
                    name='Year'
                ))
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="info-box"><h3 style="margin:0;">Delaware River Estuary Analysis</h3><p style="margin:0.5rem 0 0 0;">PFAS trends along river miles</p></div>', unsafe_allow_html=True)

    if not current_data.empty:
        # Filter points within Delaware River Bay
        pts_gdf = gpd.GeoDataFrame(
            current_data,
            geometry=[Point(xy) for xy in zip(current_data['lon'], current_data['lat'])],
            crs="EPSG:4326"
        )

        within_bay = pts_gdf[pts_gdf.geometry.within(spatial_data['delrivbay'].unary_union)]

        if not within_bay.empty:
            # Find nearest river mile
            rm_points = spatial_data['rm'].copy()

            nearest_rm = []
            for idx, point in within_bay.iterrows():
                distances = rm_points.distance(point.geometry)
                nearest_idx = distances.idxmin()
                nearest_rm.append(rm_points.loc[nearest_idx, 'RM'])

            within_bay['RM'] = nearest_rm

            # Create plot
            fig = go.Figure()

            # Get unique years and assign colors
            years = sorted(within_bay['yr'].dropna().unique())
            colors = px.colors.sample_colorscale("Spectral", [i/(len(years)-1) if len(years) > 1 else 0 for i in range(len(years))])

            for i, year in enumerate(years):
                year_data = within_bay[within_bay['yr'] == year]
                conc_vals = year_data['conc'].replace(0, np.nan)  # Replace 0 with NaN for log scale

                fig.add_trace(go.Scatter(
                    x=year_data['RM'],
                    y=conc_vals,
                    mode='markers',
                    name=str(int(year)),
                    marker=dict(size=10, color=colors[i], opacity=0.7),
                    text=[f"RM: {rm}<br>Conc: {c:.2f}<br>Year: {int(y)}"
                          for rm, c, y in zip(year_data['RM'], year_data['conc'], year_data['yr'])],
                    hovertemplate='%{text}<extra></extra>'
                ))

            fig.update_layout(
                title=f"{filter_name} Along Delaware River Estuary",
                xaxis_title="River Mile",
                yaxis_title=f"{filter_name} ({unit})",
                yaxis_type="log",
                xaxis=dict(autorange="reversed"),
                height=600,
                legend_title="Year",
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data points found within the Delaware River Estuary for current selection")
    else:
        st.info("No data available for current selection")

with tab3:
    st.markdown("### EPA Drinking Water Maximum Contaminant Level (MCL) for PFAS")

    mcl_data = {
        "Compound": ["PFOA", "PFOS", "PFHxS", "PFNA", "HFPO-DA (GenX)", "Mixture*"],
        "MCL Goal (ng/L)": ["0", "0", "10", "10", "10", "1 (HI)"],
        "Final MCL (ng/L)": ["4.0", "4.0", "10", "10", "10", "1 (HI)"]
    }
    st.table(pd.DataFrame(mcl_data))
    st.caption("*Mixture of PFHxS, PFNA, HFPO-DA, PFBS (Hazard Index)")

    st.markdown("### EPA Draft Human Health Water Quality Criteria (HHC) for PFAS")

    hhc_data = {
        "Compound": ["PFOA", "PFOS", "PFBS"],
        "Water + Organism HHC (ng/L)": ["0.0009", "0.06", "400"],
        "Organism Only HHC (ng/L)": ["0.0036", "0.07", "500"]
    }
    st.table(pd.DataFrame(hhc_data))

    st.info("Note: These HHC values are draft, non-regulatory criteria. States may adopt their own values.")

with tab4:
    st.markdown("### History of PFAS in the Basin")
    st.markdown("""
    Per- and polyfluoroalkyl substances (PFAS) are a group of more than 13,000 synthetic organic compounds
    characterized by exceptionally strong carbon–fluorine bonds. These bonds make PFAS highly resistant to heat,
    water, and oil, and protect them from natural biological and chemical degradation. As a result, PFAS persist
    for long periods in the environment and are often referred to as 'forever chemicals.'

    The Delaware River Basin holds a unique place in the history of PFAS. It was at DuPont's Chambers Works facility
    in Deepwater, New Jersey, that the first commercially produced PFAS compound—polytetrafluoroethylene (PTFE)—
    was accidentally discovered in 1938, later marketed as Teflon™. This discovery helped establish the region as
    a global center for fluoropolymer research and production.

    This long history of industrial and commercial activity across the Delaware River Basin has left a legacy of
    PFAS pollution that is heavily concentrated in the Lehigh Valley and extends south through the estuary. According
    to the EPA's ECHO database, nearly 3,000 industrial sites in the basin have been identified as potentially involved
    in the production, manufacture, or use of PFAS, with more than 1,500 facilities still active today.
    """)

with tab5:
    st.markdown("### About This Application")
    st.markdown("""
    This interactive application provides basin-wide access to publicly available monitoring data for
    **per- and polyfluoroalkyl substances (PFAS)** in surface water, groundwater, sediment, and biological
    tissue across the Delaware River Basin.

    The app was originally developed by scientists at the **Delaware River Basin Commission (DRBC)** as an
    internal tool to better understand the scope, distribution, and variability of PFAS contamination throughout
    the basin.

    #### Data Sources and Scope
    All data displayed in this application are **publicly available** and were retrieved from:
    - U.S. Environmental Protection Agency's Water Quality Portal
    - U.S. Geological Survey's Water Data API

    The dataset currently includes **over 900 ΣPFAS samples**, most of which are surface and ground water samples
    (>500), in addition to almost 300 tissue samples from 17 different aquatic species.

    #### Contact Information
    - **Jeremy Conkle**, Sr. Chemist/Toxicologist: jeremy.conkle@drbc.gov
    - **Matthew Amato**, Water Resource Scientist: matthew.amato@drbc.gov

    For more information about DRBC's PFAS program, visit:
    [https://www.nj.gov/drbc/programs/quality/pfas.html](https://www.nj.gov/drbc/programs/quality/pfas.html)
    """)

# FOOTER
st.markdown("""
<div class="footer">
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        ⚠️ <strong>WARNING:</strong> Development of this app is ongoing; information is subject to change.
    </div>
    <p style="margin-top: 1rem;">
        Development funded by EPA's Section 106 Water Pollution Control Grant Program (I-98339317-3)
    </p>
    <p style="font-style: italic; margin-top: 0.5rem;">
        App Last Modified: January 16, 2026 | Dataset Last Modified: January 12, 2026
    </p>
</div>
""", unsafe_allow_html=True)
