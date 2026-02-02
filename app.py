import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
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

# Cache data download to improve performance
@st.cache_data(ttl=3600)
def download_futures_data(tickers, start_date='2010-01-01'):
    """Download futures data from Yahoo Finance"""
    data = yf.download(tickers, start=start_date, progress=False)
    return data['Adj Close']

# Performance metrics calculation
@st.cache_data
def calculate_metrics(returns_df):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        
        if len(returns) == 0:
            continue
            
        # Basic metrics
        metrics[col] = {
            'Cumulative Return': qs.stats.comp(returns) * 100,
            'Annual Return': qs.stats.cagr(returns) * 100,
            'Annual Volatility': qs.stats.volatility(returns) * 100,
            'Sharpe Ratio': qs.stats.sharpe(returns, rf=RF_RATE),
            'Sortino Ratio': qs.stats.sortino(returns, rf=RF_RATE),
            'Max Drawdown': qs.stats.max_drawdown(returns) * 100,
            'Calmar Ratio': qs.stats.calmar(returns),
            'Omega Ratio': qs.stats.omega(returns, rf=RF_RATE),
            'VaR (95%)': qs.stats.value_at_risk(returns) * 100,
            'CVaR (95%)': qs.stats.cvar(returns) * 100,
            'Skewness': qs.stats.skew(returns),
            'Kurtosis': qs.stats.kurtosis(returns),
            'Win Rate': qs.stats.win_rate(returns) * 100,
            'Profit Factor': qs.stats.profit_factor(returns),
            'Tail Ratio': qs.stats.tail_ratio(returns),
            'Daily Value at Risk': qs.stats.value_at_risk(returns) * 100,
            'Expected Shortfall': qs.stats.expected_shortfall(returns) * 100,
        }
    
    return metrics

# Advanced plotting functions
def create_returns_chart(returns_df):
    """Create cumulative returns chart"""
    fig = go.Figure()
    
    for col in returns_df.columns:
        cum_returns = (1 + returns_df[col].dropna()).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values * 100,
            mode='lines',
            name=col,
            line=dict(width=2)
        ))
    
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
        returns = returns_df[col].dropna()
        drawdown = qs.stats.to_drawdown_series(returns) * 100
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(width=1)
        ))
    
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
    if len(returns_df.columns) > 1:
        returns = returns_df.mean(axis=1)
    else:
        returns = returns_df.iloc[:, 0]
    
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

def create_distribution_chart(returns_df):
    """Create returns distribution chart"""
    fig = make_subplots(
        rows=1, cols=len(returns_df.columns),
        subplot_titles=returns_df.columns,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(returns_df.columns, 1):
        returns = returns_df[col].dropna() * 100
        
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
    
    fig.update_layout(
        title='Returns Distribution',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_rolling_metrics_chart(returns_df):
    """Create rolling Sharpe and volatility chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Sharpe Ratio (6-month)', 'Rolling Volatility (6-month)'),
        vertical_spacing=0.15
    )
    
    window = 126  # 6 months trading days
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        
        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: qs.stats.sharpe(x, rf=RF_RATE)
        )
        
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
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Main application
def main():
    st.title("📊 Gold & Silver Futures Performance Analyzer")
    st.markdown("""
    Analyze performance and risk metrics for Gold (GC=F) and Silver (SI=F) futures using QuantStats.
    Risk-free rate is set to 2%.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker selection
    tickers = st.sidebar.multiselect(
        "Select Futures:",
        ["GC=F", "SI=F"],
        default=["GC=F", "SI=F"]
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"))
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        ["Full History", "YTD", "1Y", "3Y", "5Y"]
    )
    
    # Download button
    if st.sidebar.button("🔄 Update Data"):
        st.cache_data.clear()
    
    if not tickers:
        st.warning("Please select at least one futures contract.")
        return
    
    # Download data
    with st.spinner("Downloading futures data..."):
        try:
            data = download_futures_data(tickers, start_date=start_date)
            
            if data.empty:
                st.error("No data available for the selected tickers and date range.")
                return
            
            # Calculate returns
            returns_df = data.pct_change().dropna()
            
            # Apply period filter
            if period != "Full History":
                if period == "YTD":
                    filtered_returns = returns_df[returns_df.index.year == pd.Timestamp.now().year]
                elif period == "1Y":
                    filtered_returns = returns_df.last('1Y')
                elif period == "3Y":
                    filtered_returns = returns_df.last('3Y')
                elif period == "5Y":
                    filtered_returns = returns_df.last('5Y')
                else:
                    filtered_returns = returns_df
            else:
                filtered_returns = returns_df
            
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            return
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📊 Performance Metrics", "📉 Risk Analysis", 
        "🔍 Advanced Charts", "📋 Full Report"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        for idx, ticker in enumerate(tickers):
            if ticker in filtered_returns.columns:
                returns = filtered_returns[ticker].dropna()
                
                cols = [col1, col2, col3, col4][idx]
                
                with cols:
                    st.metric(
                        label=ticker,
                        value=f"{qs.stats.cagr(returns) * 100:.2f}%",
                        delta=f"Sharpe: {qs.stats.sharpe(returns, rf=RF_RATE):.2f}"
                    )
        
        # Cumulative returns chart
        st.plotly_chart(create_returns_chart(filtered_returns), use_container_width=True)
        
        # Drawdown chart
        st.plotly_chart(create_drawdown_chart(filtered_returns), use_container_width=True)
    
    with tab2:
        st.header("Performance Metrics")
        
        # Calculate metrics
        metrics = calculate_metrics(filtered_returns)
        
        # Display metrics in columns
        for ticker in tickers:
            if ticker in metrics:
                st.subheader(f"{ticker} Performance Metrics")
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                metric_data = metrics[ticker]
                with col1:
                    for key in list(metric_data.keys())[:len(metric_data)//2]:
                        if 'Ratio' in key or 'Rate' in key:
                            st.metric(key, f"{metric_data[key]:.3f}")
                        elif 'Return' in key or 'Volatility' in key or 'Drawdown' in key or 'VaR' in key:
                            st.metric(key, f"{metric_data[key]:.2f}%")
                        else:
                            st.metric(key, f"{metric_data[key]:.3f}")
                
                with col2:
                    for key in list(metric_data.keys())[len(metric_data)//2:]:
                        if 'Ratio' in key or 'Rate' in key:
                            st.metric(key, f"{metric_data[key]:.3f}")
                        elif 'Return' in key or 'Volatility' in key or 'Drawdown' in key or 'VaR' in key:
                            st.metric(key, f"{metric_data[key]:.2f}%")
                        else:
                            st.metric(key, f"{metric_data[key]:.3f}")
                
                st.divider()
    
    with tab3:
        st.header("Risk Analysis")
        
        # Rolling metrics
        st.plotly_chart(create_rolling_metrics_chart(filtered_returns), use_container_width=True)
        
        # Risk metrics comparison
        st.subheader("Risk Metrics Comparison")
        
        risk_metrics = ['Annual Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
        
        for metric in risk_metrics:
            cols = st.columns(len(tickers))
            for idx, ticker in enumerate(tickers):
                if ticker in metrics:
                    with cols[idx]:
                        st.metric(
                            label=f"{ticker} - {metric}",
                            value=f"{metrics[ticker][metric]:.2f}%",
                            delta="Lower is better" if metric != 'Sharpe Ratio' else ""
                        )
    
    with tab4:
        st.header("Advanced Charts")
        
        # Returns distribution
        st.plotly_chart(create_distribution_chart(filtered_returns), use_container_width=True)
        
        # Monthly heatmap
        st.plotly_chart(create_monthly_heatmap(filtered_returns), use_container_width=True)
        
        # Additional QuantStats charts
        st.subheader("QuantStats Detailed Analysis")
        
        if len(filtered_returns.columns) > 0:
            selected_ticker = st.selectbox("Select ticker for detailed analysis:", tickers)
            
            if selected_ticker:
                returns = filtered_returns[selected_ticker].dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Monthly returns distribution
                    st.write("**Monthly Returns Distribution**")
                    monthly_table = qs.stats.monthly_returns(returns) * 100
                    st.dataframe(monthly_table.style.format("{:.2f}%").background_gradient(
                        cmap='RdYlGn', axis=None, vmin=-10, vmax=10
                    ))
                
                with col2:
                    # Worst drawdown periods
                    st.write("**Worst Drawdown Periods**")
                    drawdowns = qs.stats.to_drawdown_series(returns)
                    worst_dd = qs.stats.top_drawdowns(returns)
                    
                    if len(worst_dd) > 0:
                        dd_data = []
                        for peak, recovery, dd in worst_dd:
                            dd_data.append({
                                'Peak': peak,
                                'Recovery': recovery,
                                'Drawdown': f"{dd * 100:.2f}%"
                            })
                        st.dataframe(pd.DataFrame(dd_data))
    
    with tab5:
        st.header("Full QuantStats Report")
        
        if len(filtered_returns.columns) > 0:
            selected_ticker = st.selectbox("Generate full report for:", tickers, key="report_ticker")
            
            if selected_ticker:
                returns = filtered_returns[selected_ticker].dropna()
                
                # Generate full HTML report
                st.write("**Comprehensive Performance Report**")
                
                # Display key metrics table
                stats = qs.reports.metrics(returns, rf=RF_RATE, display=False)
                stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                st.dataframe(stats_df, use_container_width=True)
                
                # Option to save report
                if st.button("📥 Generate HTML Report"):
                    with st.spinner("Generating HTML report..."):
                        # Create full report
                        html_report = qs.reports.html(
                            returns,
                            rf=RF_RATE,
                            output='output.html',
                            title=f"{selected_ticker} Performance Report"
                        )
                        
                        # Read and display the report
                        with open('output.html', 'r') as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=800, scrolling=True)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    **Note:** 
    - Data sourced from Yahoo Finance
    - Risk-free rate: 2%
    - Metrics calculated using QuantStats
    - All returns are based on adjusted close prices
    """)

if __name__ == "__main__":
    main()
