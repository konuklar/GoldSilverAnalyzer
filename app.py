import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Gold & Silver Futures Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize QuantStats
qs.extend_pandas()

# Constants
RF_RATE = 0.02  # 2% risk-free rate

# Cache data download with better error handling
@st.cache_data(ttl=3600)
def download_futures_data(tickers, start_date='2010-01-01', end_date=None):
    """Download futures data from Yahoo Finance with error handling"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Download data
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Check if we have a single ticker or multiple
        if len(tickers) == 1:
            # Single ticker returns a DataFrame with single-level columns
            if 'Adj Close' in data.columns:
                price_data = data[['Adj Close']].copy()
                price_data.columns = tickers
            else:
                # Try to find adjusted close column
                price_data = data[['Close']].copy()
                price_data.columns = tickers
        else:
            # Multiple tickers returns MultiIndex columns
            if ('Adj Close', tickers[0]) in data.columns:
                price_data = data['Adj Close'].copy()
            else:
                price_data = data.xs('Close', axis=1, level=0).copy()
        
        return price_data.dropna()
    
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

def validate_and_prepare_data(price_data, tickers):
    """Validate and prepare data for analysis"""
    if price_data.empty:
        return pd.DataFrame()
    
    # Check for each ticker
    valid_data = {}
    for ticker in tickers:
        if ticker in price_data.columns:
            series = price_data[ticker].dropna()
            
            # Remove leading/trailing zeros or NaN values
            series = series.replace(0, np.nan).dropna()
            
            if len(series) >= 2:  # Need at least 2 points for returns
                # Forward fill small gaps (up to 5 days)
                series = series.ffill(limit=5)
                
                # Ensure no negative prices (though possible, very rare)
                series = series[series > 0]
                
                if len(series) >= 2:
                    valid_data[ticker] = series
    
    if not valid_data:
        return pd.DataFrame()
    
    # Create DataFrame from valid data
    valid_df = pd.DataFrame(valid_data)
    
    # Align dates (outer join then forward fill)
    valid_df = valid_df.ffill(limit=5).dropna()
    
    # Calculate returns
    returns_df = valid_df.pct_change().dropna()
    
    # Remove extreme outliers (more than 50% daily move)
    returns_df = returns_df[(returns_df.abs() < 0.5).all(axis=1)]
    
    # Ensure we have enough data
    if len(returns_df) < 5:
        return pd.DataFrame()
    
    return returns_df

def get_data_with_validation(tickers, start_date, end_date, period):
    """Get and validate data with period filtering"""
    try:
        # Download data
        price_data = download_futures_data(
            tickers, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Calculate returns
        returns_df = validate_and_prepare_data(price_data, tickers)
        
        if returns_df.empty:
            return pd.DataFrame()
        
        # Apply period filter
        if period != "Full History":
            if period == "YTD":
                current_year = pd.Timestamp.now().year
                filtered_returns = returns_df[returns_df.index.year == current_year]
            elif period == "1Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=1)
                filtered_returns = returns_df[returns_df.index >= cutoff_date]
            elif period == "3Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=3)
                filtered_returns = returns_df[returns_df.index >= cutoff_date]
            elif period == "5Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=5)
                filtered_returns = returns_df[returns_df.index >= cutoff_date]
            else:
                filtered_returns = returns_df
        else:
            filtered_returns = returns_df
        
        # Final validation - ensure we have enough data
        if len(filtered_returns) < 5:
            st.warning(f"Insufficient data for {period} period. Using all available data.")
            filtered_returns = returns_df
        
        return filtered_returns
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def safe_quantstats_calculation(func, returns, *args, **kwargs):
    """Safely calculate QuantStats metrics with error handling"""
    try:
        if len(returns) < 2:
            return np.nan
        
        # Ensure returns is a pandas Series
        if isinstance(returns, pd.DataFrame):
            if len(returns.columns) > 0:
                returns = returns.iloc[:, 0]
            else:
                return np.nan
        
        # Check for NaN or infinite values
        if returns.isna().any() or not np.isfinite(returns).all():
            return np.nan
        
        result = func(returns, *args, **kwargs)
        
        # Handle potential NaN results
        if pd.isna(result):
            return np.nan
        
        return result
        
    except Exception:
        return np.nan

# Performance metrics calculation with error handling
@st.cache_data
def calculate_metrics(returns_df):
    """Calculate comprehensive performance metrics with error handling"""
    metrics = {}
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        
        if len(returns) < 10:  # Minimum data points
            metrics[col] = {metric: np.nan for metric in [
                'Cumulative Return', 'Annual Return', 'Annual Volatility',
                'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
                'Calmar Ratio', 'Omega Ratio', 'VaR (95%)', 'CVaR (95%)',
                'Skewness', 'Kurtosis', 'Win Rate', 'Profit Factor',
                'Tail Ratio', 'Daily Value at Risk', 'Expected Shortfall'
            ]}
            continue
        
        # Calculate each metric with error handling
        metrics[col] = {
            'Cumulative Return': safe_quantstats_calculation(qs.stats.comp, returns) * 100,
            'Annual Return': safe_quantstats_calculation(qs.stats.cagr, returns) * 100,
            'Annual Volatility': safe_quantstats_calculation(qs.stats.volatility, returns) * 100,
            'Sharpe Ratio': safe_quantstats_calculation(qs.stats.sharpe, returns, rf=RF_RATE),
            'Sortino Ratio': safe_quantstats_calculation(qs.stats.sortino, returns, rf=RF_RATE),
            'Max Drawdown': safe_quantstats_calculation(qs.stats.max_drawdown, returns) * 100,
            'Calmar Ratio': safe_quantstats_calculation(qs.stats.calmar, returns),
            'Omega Ratio': safe_quantstats_calculation(qs.stats.omega, returns, rf=RF_RATE),
            'VaR (95%)': safe_quantstats_calculation(qs.stats.value_at_risk, returns) * 100,
            'CVaR (95%)': safe_quantstats_calculation(qs.stats.cvar, returns) * 100,
            'Skewness': safe_quantstats_calculation(qs.stats.skew, returns),
            'Kurtosis': safe_quantstats_calculation(qs.stats.kurtosis, returns),
            'Win Rate': safe_quantstats_calculation(qs.stats.win_rate, returns) * 100,
            'Profit Factor': safe_quantstats_calculation(qs.stats.profit_factor, returns),
            'Tail Ratio': safe_quantstats_calculation(qs.stats.tail_ratio, returns),
            'Daily Value at Risk': safe_quantstats_calculation(qs.stats.value_at_risk, returns) * 100,
            'Expected Shortfall': safe_quantstats_calculation(qs.stats.expected_shortfall, returns) * 100,
        }
    
    return metrics

# Advanced plotting functions (unchanged, but using safe functions)
def create_returns_chart(returns_df):
    """Create cumulative returns chart"""
    fig = go.Figure()
    
    for col in returns_df.columns:
        try:
            cum_returns = (1 + returns_df[col].dropna()).cumprod()
            if len(cum_returns) > 0:
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values * 100,
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
        except Exception:
            continue
    
    fig.update_layout(
        title='Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_drawdown_chart(returns_df):
    """Create drawdown chart"""
    fig = go.Figure()
    
    for col in returns_df.columns:
        try:
            returns = returns_df[col].dropna()
            if len(returns) > 0:
                drawdown = qs.stats.to_drawdown_series(returns) * 100
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name=col,
                    fill='tozeroy',
                    line=dict(width=1)
                ))
        except Exception:
            continue
    
    fig.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_monthly_heatmap(returns_df):
    """Create monthly returns heatmap"""
    if returns_df.empty:
        return go.Figure()
    
    if len(returns_df.columns) > 1:
        returns = returns_df.mean(axis=1)
    else:
        returns = returns_df.iloc[:, 0]
    
    try:
        monthly_returns = qs.stats.monthly_returns(returns) * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns.values,
            x=monthly_returns.columns,
            y=monthly_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=monthly_returns.round(2).values,
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400,
            template='plotly_white'
        )
        
        return fig
    except Exception:
        return go.Figure()

def create_distribution_chart(returns_df):
    """Create returns distribution chart"""
    if returns_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=len(returns_df.columns),
        subplot_titles=returns_df.columns,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(returns_df.columns, 1):
        try:
            returns = returns_df[col].dropna() * 100
            
            if len(returns) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        nbinsx=50,
                        name=col,
                        marker_color='skyblue',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                # Add vertical line for mean
                mean_return = returns.mean()
                fig.add_vline(
                    x=mean_return, 
                    line_dash="dash", 
                    line_color="red",
                    row=1, col=idx
                )
                
                fig.update_xaxes(title_text="Daily Return (%)", row=1, col=idx)
                fig.update_yaxes(title_text="Frequency", row=1, col=idx)
        except Exception:
            continue
    
    fig.update_layout(
        title='Returns Distribution',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_rolling_metrics_chart(returns_df):
    """Create rolling Sharpe and volatility chart"""
    if returns_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Sharpe Ratio (6-month)', 'Rolling Volatility (6-month)'),
        vertical_spacing=0.15
    )
    
    window = 126  # 6 months trading days
    
    for col in returns_df.columns:
        try:
            returns = returns_df[col].dropna()
            
            if len(returns) < window:
                continue
            
            # Rolling Sharpe
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: safe_quantstats_calculation(qs.stats.sharpe, x, rf=RF_RATE),
                raw=False
            ).dropna()
            
            if len(rolling_sharpe) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name=f'{col} - Sharpe',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
            
            # Rolling Volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode='lines',
                        name=f'{col} - Volatility',
                        line=dict(width=1)
                    ),
                    row=2, col=1
                )
        except Exception:
            continue
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def format_metric_value(value, metric_name):
    """Format metric values appropriately"""
    if pd.isna(value):
        return "N/A"
    
    if 'Ratio' in metric_name or 'Rate' in metric_name or metric_name in ['Skewness', 'Kurtosis', 'Profit Factor', 'Tail Ratio']:
        return f"{value:.3f}"
    elif 'Return' in metric_name or 'Volatility' in metric_name or 'Drawdown' in metric_name or 'VaR' in metric_name:
        return f"{value:.2f}%"
    else:
        return f"{value:.3f}"

# Main application
def main():
    st.title("📊 Gold & Silver Futures Performance Analyzer")
    st.markdown("""
    Analyze performance and risk metrics for Gold (GC=F) and Silver (SI=F) futures using QuantStats.
    Risk-free rate is set to 2%.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker selection with full names
    ticker_info = {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "HG=F": "Copper Futures"
    }
    
    ticker_options = list(ticker_info.keys())
    display_names = [f"{ticker} - {ticker_info[ticker]}" for ticker in ticker_options]
    
    selected_display = st.sidebar.multiselect(
        "Select Futures Contracts:",
        display_names,
        default=[display_names[0], display_names[1]]
    )
    
    # Extract ticker symbols
    tickers = [name.split(" - ")[0] for name in selected_display]
    
    # Date range with sensible defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # Default to 5 years ago for better data
        default_start = datetime.now() - timedelta(days=5*365)
        start_date = st.date_input("Start Date", default_start)
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Ensure start date is before end date
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date!")
        return
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        ["Full History", "YTD", "1Y", "3Y", "5Y"]
    )
    
    # Minimum data threshold
    min_days = st.sidebar.slider(
        "Minimum days of data required:",
        min_value=10,
        max_value=100,
        value=30,
        help="Skip analysis if we have fewer than this many data points"
    )
    
    # Download button
    if st.sidebar.button("🔄 Update Data"):
        st.cache_data.clear()
    
    if not tickers:
        st.warning("Please select at least one futures contract.")
        return
    
    # Download and validate data
    with st.spinner("Downloading and validating futures data..."):
        returns_df = get_data_with_validation(tickers, start_date, end_date, period)
        
        if returns_df.empty:
            st.error("""
            No valid data available for the selected criteria. This could be due to:
            1. No trading data for the selected date range
            2. All data points are NaN or zero
            3. Insufficient data points after cleaning
            
            Please try:
            - Selecting a different date range
            - Using the default futures (Gold and Silver)
            - Checking if markets were open during the selected period
            """)
            return
        
        if len(returns_df) < min_days:
            st.warning(f"Only {len(returns_df)} days of data available (minimum requested: {min_days}). Analysis may be limited.")
        
        # Display data info
        st.sidebar.success(f"Data loaded: {len(returns_df)} trading days")
        st.sidebar.info(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📊 Performance Metrics", "📉 Risk Analysis", 
        "🔍 Advanced Charts", "📋 Data & Diagnostics"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        # Display data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Returns Data Preview")
            st.dataframe(returns_df.tail(10).style.format("{:.4%}"))
        
        with col2:
            st.subheader("Data Statistics")
            stats_df = pd.DataFrame({
                'Mean': returns_df.mean() * 100,
                'Std Dev': returns_df.std() * 100,
                'Min': returns_df.min() * 100,
                'Max': returns_df.max() * 100,
                'Count': returns_df.count()
            }).T
            st.dataframe(stats_df.style.format("{:.2f}%", subset=[col for col in stats_df.columns if col != 'Count']))
        
        # Summary statistics
        st.subheader("Quick Metrics")
        cols = st.columns(len(tickers))
        
        for idx, ticker in enumerate(tickers):
            if ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                
                if len(returns) >= 10:
                    cagr = safe_quantstats_calculation(qs.stats.cagr, returns) * 100
                    sharpe = safe_quantstats_calculation(qs.stats.sharpe, returns, rf=RF_RATE)
                    volatility = safe_quantstats_calculation(qs.stats.volatility, returns) * 100
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{ticker} ({ticker_info.get(ticker, ticker)})",
                            value=f"{cagr:.2f}%" if not pd.isna(cagr) else "N/A",
                            delta=f"Sharpe: {sharpe:.2f}" if not pd.isna(sharpe) else "N/A"
                        )
                        st.caption(f"Volatility: {volatility:.2f}%" if not pd.isna(volatility) else "Volatility: N/A")
        
        # Cumulative returns chart
        st.plotly_chart(create_returns_chart(returns_df), use_container_width=True)
        
        # Drawdown chart
        st.plotly_chart(create_drawdown_chart(returns_df), use_container_width=True)
    
    with tab2:
        st.header("Performance Metrics")
        
        # Calculate metrics
        metrics = calculate_metrics(returns_df)
        
        # Display metrics in columns
        for ticker in tickers:
            if ticker in metrics:
                st.subheader(f"{ticker} - {ticker_info.get(ticker, ticker)}")
                
                # Check if we have valid metrics
                if all(pd.isna(v) for v in metrics[ticker].values()):
                    st.warning("Insufficient data to calculate metrics for this instrument.")
                    continue
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                metric_data = metrics[ticker]
                with col1:
                    for key in list(metric_data.keys())[:len(metric_data)//2]:
                        value = metric_data[key]
                        st.metric(key, format_metric_value(value, key))
                
                with col2:
                    for key in list(metric_data.keys())[len(metric_data)//2:]:
                        value = metric_data[key]
                        st.metric(key, format_metric_value(value, key))
                
                st.divider()
    
    with tab3:
        st.header("Risk Analysis")
        
        # Rolling metrics
        st.plotly_chart(create_rolling_metrics_chart(returns_df), use_container_width=True)
        
        # Risk metrics comparison
        st.subheader("Risk Metrics Comparison")
        
        risk_metrics = ['Annual Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Sharpe Ratio']
        
        cols = st.columns(len(tickers))
        for idx, ticker in enumerate(tickers):
            if ticker in metrics:
                with cols[idx]:
                    st.markdown(f"**{ticker}**")
                    for metric in risk_metrics:
                        if metric in metrics[ticker]:
                            value = metrics[ticker][metric]
                            if not pd.isna(value):
                                st.metric(
                                    label=metric,
                                    value=format_metric_value(value, metric)
                                )
    
    with tab4:
        st.header("Advanced Charts")
        
        # Returns distribution
        st.plotly_chart(create_distribution_chart(returns_df), use_container_width=True)
        
        # Monthly heatmap
        st.plotly_chart(create_monthly_heatmap(returns_df), use_container_width=True)
        
        # Additional QuantStats charts
        st.subheader("QuantStats Detailed Analysis")
        
        if len(returns_df.columns) > 0:
            selected_ticker = st.selectbox("Select ticker for detailed analysis:", tickers)
            
            if selected_ticker:
                returns = returns_df[selected_ticker].dropna()
                
                if len(returns) >= 20:  # Minimum for monthly analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Monthly returns distribution
                        st.write("**Monthly Returns Distribution**")
                        try:
                            monthly_table = qs.stats.monthly_returns(returns) * 100
                            st.dataframe(monthly_table.style.format("{:.2f}%").background_gradient(
                                cmap='RdYlGn', axis=None, vmin=-10, vmax=10
                            ))
                        except Exception:
                            st.warning("Could not calculate monthly returns")
                    
                    with col2:
                        # Worst drawdown periods
                        st.write("**Worst Drawdown Periods**")
                        try:
                            worst_dd = qs.stats.top_drawdowns(returns)
                            
                            if len(worst_dd) > 0:
                                dd_data = []
                                for peak, recovery, dd in worst_dd:
                                    dd_data.append({
                                        'Peak': peak.date() if hasattr(peak, 'date') else peak,
                                        'Recovery': recovery.date() if hasattr(recovery, 'date') else recovery,
                                        'Drawdown': f"{dd * 100:.2f}%"
                                    })
                                st.dataframe(pd.DataFrame(dd_data))
                            else:
                                st.info("No significant drawdowns found")
                        except Exception:
                            st.warning("Could not calculate drawdown periods")
                else:
                    st.warning("Insufficient data for detailed analysis")
    
    with tab5:
        st.header("Data & Diagnostics")
        
        # Show raw data
        st.subheader("Raw Price Data")
        
        # Download raw prices for reference
        try:
            price_data = download_futures_data(
                tickers, 
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not price_data.empty:
                st.dataframe(price_data.tail(20))
                
                # Download button for data
                csv = price_data.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Price Data (CSV)",
                    data=csv,
                    file_name="futures_price_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No raw price data available")
        except Exception as e:
            st.error(f"Could not load raw data: {str(e)}")
        
        # Data quality report
        st.subheader("Data Quality Report")
        
        quality_report = []
        for ticker in tickers:
            if ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                
                quality_report.append({
                    'Ticker': ticker,
                    'Days Available': len(returns),
                    'Missing Values': returns.isna().sum(),
                    'Zero Returns': (returns == 0).sum(),
                    'Positive Days': (returns > 0).sum(),
                    'Negative Days': (returns < 0).sum(),
                    'Start Date': returns.index.min().date() if len(returns) > 0 else 'N/A',
                    'End Date': returns.index.max().date() if len(returns) > 0 else 'N/A'
                })
        
        if quality_report:
            st.dataframe(pd.DataFrame(quality_report))
        
        # System information
        st.subheader("System Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Python Version", f"{pd.__version__}")
        with col2:
            st.metric("Pandas Version", f"{pd.__version__}")
        with col3:
            st.metric("QuantStats Version", f"{qs.__version__}")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    **Data Sources:** 
    - Futures data: Yahoo Finance
    - Risk-free rate: 2% (annualized)
    
    **Notes:**
    - All returns are daily returns
    - Missing data is forward-filled up to 5 days
    - Extreme returns (>50% daily) are filtered out
    - Analysis requires minimum 10 data points
    """)

if __name__ == "__main__":
    main()
