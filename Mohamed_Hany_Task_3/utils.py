"""
utils.py — Data loading, cleaning, and Plotly visualization helpers
for the Uber Fare Prediction EDA Dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
PRIMARY_COLOR   = '#2563eb'   # Royal Blue
SECONDARY_COLOR = '#10b981'   # Emerald Green
ACCENT_COLOR    = '#f59e0b'   # Amber
DANGER_COLOR    = '#ef4444'   # Rose
DARK_COLOR      = '#475569'   # Slate Gray
palette = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, DANGER_COLOR, DARK_COLOR,
           '#8b5cf6', '#ec4899', '#06b6d4']

# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _apply_style(fig, height=420):
    """Apply clean, consistent styling to any Plotly figure."""
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        title_font=dict(size=15, family='Inter, sans-serif', color='#1e293b'),
        font=dict(family='Inter, sans-serif', size=12, color='#475569'),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1,
        ),
    )
    # Soften grid lines on every axis
    for axis in ('xaxis', 'yaxis'):
        fig.update_layout(**{axis: dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            gridwidth=1,
            linecolor='#e2e8f0',
            tickfont=dict(size=11),
            title_font=dict(size=12),
            zeroline=False,
        )})
    return fig


# ---------------------------------------------------------------------------
# Data Loading & Cleaning (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_preview(n_rows: int = 100) -> pd.DataFrame:
    """Return the first *n_rows* rows of the raw CSV for preview purposes."""
    try:
        return pd.read_csv('data/final_internship_data.csv', nrows=n_rows)
    except FileNotFoundError:
        st.error("❌ Dataset file `data/final_internship_data.csv` not found. Make sure it lives in the data folder.")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cleaned_data_with_steps():
    """
    Execute the full data-cleaning pipeline exactly as defined in the notebook.

    Returns
    -------
    df_clean : pd.DataFrame
        The cleaned and feature-engineered DataFrame.
    steps : list[dict]
        Step-by-step audit trail with row counts and descriptions.
    """
    df = pd.read_csv('data/final_internship_data.csv')

    steps = [{
        'name': 'Initial Dataset Ingestion',
        'desc': 'Raw dataset loaded from data/final_internship_data.csv',
        'rows': len(df), 'cols': df.shape[1], 'removed': 0,
    }]

    def _log(name, desc):
        steps.append({
            'name': name, 'desc': desc,
            'rows': len(df), 'cols': df.shape[1],
            'removed': steps[-1]['rows'] - len(df),
        })

    # Step 1 — Drop irrelevant / PII columns
    df = df.drop(columns=['User ID', 'User Name', 'Driver Name', 'key', 'pickup_datetime'],
                 errors='ignore')
    _log('Drop Irrelevant Columns',
         'Removed User ID, User Name, Driver Name, key, and pickup_datetime '
         '(redundant PII & high-cardinality fields).')

    # Step 2 — Drop rows missing dropoff coordinates
    df = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude'])
    _log('Drop Null Coordinates',
         'Dropped 5 rows where dropoff longitude or latitude coordinates were missing.')

    # Step 3 — Fare amount range [$2.50, $150]
    df = df[(df['fare_amount'] >= 2.5) & (df['fare_amount'] <= 150)]
    _log('Filter Fare Amount [2.5 – 150]',
         'Removed fares below the NYC base fare ($2.50) and extreme outliers (> $150.00).')

    # Step 4 — NYC bounding box (coordinates stored in radians in this dataset)
    lon_min, lon_max = np.deg2rad(-74.30), np.deg2rad(-73.65)
    lat_min, lat_max = np.deg2rad(40.45),  np.deg2rad(40.95)
    df = df[
        df['pickup_longitude'].between(lon_min, lon_max) &
        df['pickup_latitude'].between(lat_min, lat_max) &
        df['dropoff_longitude'].between(lon_min, lon_max) &
        df['dropoff_latitude'].between(lat_min, lat_max)
    ]
    _log('NYC Bounding Box Filter',
         'Restricted all coordinates to metropolitan NYC limits (values already in radians).')

    # Step 5 — Passenger count [1, 4]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 4)]
    _log('Filter Passenger Count [1 – 4]',
         'Removed zero-passenger records (invalid) and capped at 4 (standard UberX sedan).')

    # Step 6 — Distance (0, 100] km
    df = df[(df['distance'] > 0.0) & (df['distance'] <= 100.0)]
    _log('Filter Distance (0 – 100 km]',
         'Removed zero-distance trips (cancellations/GPS errors) and capped at 100 km.')

    # Feature Engineering — Cyclic temporal encodings
    df = df.copy()   # defrag / prevent SettingWithCopyWarning
    df['hour_sin']    = np.sin(2 * np.pi * df['hour']           / 24.0)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour']           / 24.0)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']        / 7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']        / 7.0)
    df['month_sin']   = np.sin(2 * np.pi * (df['month'] - 1)   / 12.0)
    df['month_cos']   = np.cos(2 * np.pi * (df['month'] - 1)   / 12.0)

    return df, steps


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_fare_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram — distribution of fare_amount."""
    fig = px.histogram(
        df, x='fare_amount', nbins=120,
        title='Distribution of Fare Amount ($)',
        labels={'fare_amount': 'Fare Amount ($)', 'count': 'Ride Count'},
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(xaxis_title='Fare Amount ($)', yaxis_title='Ride Count',
                      bargap=0.02)
    return _apply_style(fig)


def plot_passenger_count_rides(df: pd.DataFrame) -> go.Figure:
    """Bar chart — total rides by passenger count."""
    counts = df['passenger_count'].value_counts().sort_index().reset_index()
    counts.columns = ['passenger_count', 'ride_count']
    counts['passenger_count'] = counts['passenger_count'].astype(str)

    fig = px.bar(
        counts, x='passenger_count', y='ride_count',
        color='passenger_count',
        title='Total Rides by Passenger Count',
        labels={'passenger_count': 'Passenger Count', 'ride_count': 'Total Rides'},
        color_discrete_sequence=palette,
    )
    fig.update_layout(xaxis_title='Number of Passengers', yaxis_title='Ride Count',
                      showlegend=False)
    return _apply_style(fig)


def plot_passenger_count_box(df: pd.DataFrame) -> go.Figure:
    """Box plot — fare distribution by passenger count."""
    fig = px.box(
        df,
        x=df['passenger_count'].astype(str),
        y='fare_amount',
        color=df['passenger_count'].astype(str),
        title='Fare Distribution by Passenger Count',
        labels={'x': 'Passenger Count', 'fare_amount': 'Fare Amount ($)'},
        color_discrete_sequence=palette,
    )
    fig.update_layout(xaxis_title='Passenger Count', yaxis_title='Fare Amount ($)',
                      showlegend=False)
    return _apply_style(fig)


def plot_car_condition_avg(df: pd.DataFrame) -> go.Figure:
    """Bar chart — average fare by car condition."""
    avg = df.groupby('Car Condition')['fare_amount'].mean().reset_index()
    avg['fare_amount'] = avg['fare_amount'].round(2)

    fig = px.bar(
        avg, x='Car Condition', y='fare_amount',
        color='Car Condition',
        text='fare_amount',
        title='Average Fare by Car Condition',
        labels={'Car Condition': 'Car Condition', 'fare_amount': 'Average Fare ($)'},
        color_discrete_sequence=palette,
    )
    fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title='Car Condition', yaxis_title='Average Fare ($)',
                      showlegend=False, uniformtext_minsize=10)
    return _apply_style(fig)


def plot_car_condition_box(df: pd.DataFrame) -> go.Figure:
    """Box plot — fare distribution by car condition."""
    fig = px.box(
        df, x='Car Condition', y='fare_amount',
        color='Car Condition',
        title='Fare Distribution by Car Condition',
        labels={'Car Condition': 'Car Condition', 'fare_amount': 'Fare Amount ($)'},
        color_discrete_sequence=palette,
    )
    fig.update_layout(xaxis_title='Car Condition', yaxis_title='Fare Amount ($)',
                      showlegend=False)
    return _apply_style(fig)


def plot_hour_fare_trend(df: pd.DataFrame) -> go.Figure:
    """Area chart — average fare by hour of day."""
    hourly = df.groupby('hour')['fare_amount'].mean().reset_index()

    fig = px.area(
        hourly, x='hour', y='fare_amount',
        title='Average Fare Amount Trend by Hour of Day',
        labels={'hour': 'Hour of Day (24h)', 'fare_amount': 'Average Fare ($)'},
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_traces(line_color=PRIMARY_COLOR, fillcolor='rgba(37,99,235,0.12)')
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, title='Hour of Day (24h)'),
        yaxis_title='Average Fare ($)',
    )
    return _apply_style(fig)


def plot_traffic_violin(df: pd.DataFrame) -> go.Figure:
    """Violin plot — fare distribution by traffic condition."""
    fig = px.violin(
        df,
        x='Traffic Condition', y='fare_amount',
        color='Traffic Condition',
        box=True, points=False,
        title='Fare Distribution by Traffic Condition',
        labels={'Traffic Condition': 'Traffic Condition', 'fare_amount': 'Fare Amount ($)'},
        color_discrete_sequence=palette,
    )
    fig.update_layout(xaxis_title='Traffic Condition', yaxis_title='Fare Amount ($)',
                      showlegend=False)
    return _apply_style(fig)


def plot_distance_scatter(df: pd.DataFrame, sample_size: int) -> go.Figure:
    """
    Scatter plot — trip distance vs fare with a manual OLS regression line
    drawn via numpy (no statsmodels dependency required for the line itself).
    """
    n = min(sample_size, len(df))
    sdf = df.sample(n, random_state=42)

    # Compute OLS coefficients with numpy
    x = sdf['distance'].values
    y = sdf['fare_amount'].values
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coef, x_line)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=3, color=PRIMARY_COLOR, opacity=0.20),
        name='Rides',
        hovertemplate='Distance: %{x:.2f} km<br>Fare: $%{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        line=dict(color=DANGER_COLOR, width=2.5),
        name=f'OLS  (slope={coef[0]:.2f})',
    ))
    fig.update_layout(
        title=f'Trip Distance vs. Fare Amount  (sample = {n:,} rides)',
        xaxis_title='Distance (km)',
        yaxis_title='Fare Amount ($)',
        showlegend=True,
    )
    return _apply_style(fig)


def plot_ride_demand(df: pd.DataFrame) -> go.Figure:
    """Bar chart — total ride demand by hour."""
    counts = df['hour'].value_counts().sort_index().reset_index()
    counts.columns = ['hour', 'ride_count']

    fig = px.bar(
        counts, x='hour', y='ride_count',
        title='Total Ride Demand by Hour of Day',
        labels={'hour': 'Hour of Day (24h)', 'ride_count': 'Total Rides'},
        color='ride_count',
        color_continuous_scale=[[0, '#bfdbfe'], [1, PRIMARY_COLOR]],
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, title='Hour of Day (24h)'),
        yaxis_title='Total Rides',
        coloraxis_showscale=False,
    )
    return _apply_style(fig)


def plot_weather_distance(df: pd.DataFrame) -> go.Figure:
    """Box plot — trip distance by weather condition."""
    fig = px.box(
        df, x='Weather', y='distance',
        color='Weather',
        title='Trip Distance by Weather Condition',
        labels={'Weather': 'Weather Condition', 'distance': 'Trip Distance (km)'},
        color_discrete_sequence=palette,
    )
    fig.update_layout(xaxis_title='Weather Condition', yaxis_title='Trip Distance (km)',
                      showlegend=False)
    return _apply_style(fig)


def plot_jfk_scatter(df: pd.DataFrame, sample_size: int) -> go.Figure:
    """Scatter plot — JFK airport distance vs fare amount."""
    n = min(sample_size, len(df))
    sdf = df.sample(n, random_state=42)

    fig = go.Figure(go.Scatter(
        x=sdf['jfk_dist'], y=sdf['fare_amount'],
        mode='markers',
        marker=dict(size=3, color=SECONDARY_COLOR, opacity=0.25),
        hovertemplate='JFK dist: %{x:.2f} km<br>Fare: $%{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=f'Distance to JFK Airport vs. Fare Amount  (sample = {n:,} rides)',
        xaxis_title='Distance to JFK (km)',
        yaxis_title='Fare Amount ($)',
    )
    return _apply_style(fig)


def plot_geo_pickups(df: pd.DataFrame, sample_size: int) -> go.Figure:
    """Scatter map — NYC pickup locations coloured by fare amount."""
    n = min(sample_size, len(df))
    sdf = df.sample(n, random_state=42).copy()

    # Coordinates stored in radians — convert back to degrees for plotting
    sdf['lat_deg'] = np.rad2deg(sdf['pickup_latitude'])
    sdf['lon_deg'] = np.rad2deg(sdf['pickup_longitude'])

    fig = px.scatter(
        sdf, x='lon_deg', y='lat_deg',
        color='fare_amount',
        color_continuous_scale='Viridis',
        opacity=0.30,
        title=f'NYC Pickup Locations & Fare Amounts  (sample = {n:,} rides)',
        labels={'lon_deg': 'Longitude', 'lat_deg': 'Latitude', 'fare_amount': 'Fare ($)'},
    )
    fig.update_traces(marker_size=2)
    fig.update_layout(
        xaxis_title='Longitude', yaxis_title='Latitude',
        coloraxis_colorbar=dict(title='Fare ($)', len=0.8),
    )
    return _apply_style(fig, height=480)


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Heatmap — Pearson correlation matrix of numeric features."""
    cols = [
        'fare_amount', 'distance', 'bearing',
        'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist',
        'passenger_count', 'hour', 'year',
    ]
    corr = df[cols].corr()

    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Pearson Correlation Matrix — Numeric Features',
        labels={'color': 'Correlation'},
        aspect='auto',
    )
    fig.update_traces(textfont_size=11)
    fig.update_layout(xaxis_title='', yaxis_title='',
                      coloraxis_colorbar=dict(title='r', len=0.8))
    return _apply_style(fig, height=480)
