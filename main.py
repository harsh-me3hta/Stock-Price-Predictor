import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64

warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60
        self.data = None
        self.symbol = None
        
    def fetch_data(self, symbol, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            if period == 'YTD':
                # Get YTD data
                start_date = datetime(datetime.now().year, 1, 1)
                data = stock.history(start=start_date)
            else:
                # Get 1 year data
                data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            self.data = data
            self.symbol = symbol
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def prepare_data(self, data, target_column='Close'):
        """Prepare data for LSTM training"""
        # Use only the target column
        dataset = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y, scaled_data
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return history
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        if self.model is None or self.data is None:
            raise ValueError("Model not trained or data not available")
        
        # Get last sequence_length days of data
        last_sequence = self.data['Close'].values[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = last_sequence_scaled.reshape(1, self.sequence_length, 1)
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def create_prediction_plot(self, predictions, days=30):
        """Create interactive plot with historical and predicted data"""
        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        # Create subplot with better sizing
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📊 Historical Stock Price', '🔮 Future Price Predictions'),
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35]
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                mode='lines',
                name='Historical Close Price',
                line=dict(color='#1f77b4', width=2.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, color='#ff7f0e', symbol='diamond'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Connect last historical point to first prediction
        fig.add_trace(
            go.Scatter(
                x=[self.data.index[-1], future_dates[0]],
                y=[self.data['Close'].iloc[-1], predictions[0]],
                mode='lines',
                name='Transition',
                line=dict(color='#2ca02c', width=2.5, dash='dot'),
                showlegend=False,
                hovertemplate='<b>Transition Point</b><extra></extra>'
            ),
            row=2, col=1
        )
        
        # Enhanced styling
        fig.update_layout(
            title={
                'text': f'<b>{self.symbol} Stock Price Analysis & Prediction</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f1f1f'}
            },
            height=700,
            width=None,  # Let it be responsive
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="<b>Date</b>", 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_xaxes(
            title_text="<b>Future Dates</b>", 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_yaxes(
            title_text="<b>Price (USD)</b>", 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        fig.update_yaxes(
            title_text="<b>Predicted Price (USD)</b>", 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        return fig

# Initialize predictor
predictor = StockPredictor()

def predict_stock(symbol, period, prediction_days, epochs, batch_size):
    """Main prediction function"""
    try:
        # Update progress
        progress_text = "Fetching data from Yahoo Finance..."
        
        # Fetch data
        data = predictor.fetch_data(symbol.upper(), period)
        
        if len(data) < predictor.sequence_length:
            return None, f"Error: Not enough data. Need at least {predictor.sequence_length} days of data."
        
        progress_text += f"\nFetched {len(data)} days of data"
        
        # Prepare data
        progress_text += "\nPreparing data for training..."
        X, y, scaled_data = predictor.prepare_data(data)
        
        if len(X) == 0:
            return None, "Error: Not enough data to create training sequences."
        
        progress_text += f"\nCreated {len(X)} training sequences"
        
        # Train model
        progress_text += "\nTraining LSTM model..."
        history = predictor.train_model(X, y, epochs=epochs, batch_size=batch_size)
        
        # Make predictions
        progress_text += "\nGenerating predictions..."
        predictions = predictor.predict_future(prediction_days)
        
        # Create plot
        fig = predictor.create_prediction_plot(predictions, prediction_days)
        
        # Create summary
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Determine trend emoji and color
        trend_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
        trend_color = "var(--color-success)" if price_change > 0 else "var(--color-error)" if price_change < 0 else "var(--color-text-secondary)"
        
        summary = f"""
        <div class="summary-header">
            <h2>📊 Prediction Summary for {symbol.upper()}</h2>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>💰 Current Price</h3>
                <div class="metric-value">${current_price:.2f}</div>
            </div>
            
            <div class="metric-card">
                <h3>🔮 Predicted Price ({prediction_days} days)</h3>
                <div class="metric-value">${predicted_price:.2f}</div>
            </div>
            
            <div class="metric-card">
                <h3>{trend_emoji} Expected Change</h3>
                <div class="metric-value" style="color: {trend_color};">${price_change:+.2f} ({price_change_percent:+.2f}%)</div>
            </div>
        </div>
        
        <div class="performance-section">
            <h3>🤖 Model Performance</h3>
            <div class="performance-grid">
                <div class="performance-item">
                    <strong>Training Loss:</strong> <span style="color: var(--color-error);">{history.history['loss'][-1]:.6f}</span>
                </div>
                <div class="performance-item">
                    <strong>Validation Loss:</strong> <span style="color: var(--color-warning);">{history.history['val_loss'][-1]:.6f}</span>
                </div>
                <div class="performance-item">
                    <strong>Training Sequences:</strong> <span style="color: var(--color-primary);">{len(X)}</span>
                </div>
                <div class="performance-item">
                    <strong>Epochs Completed:</strong> <span style="color: var(--color-success);">{epochs}</span>
                </div>
            </div>
        </div>
        
        <div class="disclaimer-section">
            <h4>⚠️ Important Disclaimer</h4>
            <p>
                This prediction is based on historical data and LSTM modeling. Stock prices are highly volatile and 
                influenced by many factors not captured in this model. Use this information for 
                <strong>educational purposes only</strong> and not for actual trading decisions.
            </p>
        </div>
        """
        
        return fig, summary
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Stock Price Predictor with LSTM",
    theme=gr.themes.Soft(),
    css="""
    /* CSS Variables for Theme Adaptation */
    :root {
        --color-primary: #007bff;
        --color-success: #28a745;
        --color-error: #dc3545;
        --color-warning: #fd7e14;
        --color-info: #17a2b8;
        --color-text-primary: #333;
        --color-text-secondary: #666;
        --color-bg-primary: #ffffff;
        --color-bg-secondary: #f8f9fa;
        --color-bg-tertiary: #e9ecef;
        --color-border: #dee2e6;
        --color-shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Dark theme variables */
    [data-theme="dark"] {
        --color-primary: #4dabf7;
        --color-success: #51cf66;
        --color-error: #ff6b6b;
        --color-warning: #ffd43b;
        --color-info: #74c0fc;
        --color-text-primary: #ffffff;
        --color-text-secondary: #ced4da;
        --color-bg-primary: #1a1a1a;
        --color-bg-secondary: #2d2d2d;
        --color-bg-tertiary: #404040;
        --color-border: #495057;
        --color-shadow: rgba(255, 255, 255, 0.1);
    }
    
    /* Auto-detect system theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --color-primary: #4dabf7;
            --color-success: #51cf66;
            --color-error: #ff6b6b;
            --color-warning: #ffd43b;
            --color-info: #74c0fc;
            --color-text-primary: #ffffff;
            --color-text-secondary: #ced4da;
            --color-bg-primary: #1a1a1a;
            --color-bg-secondary: #2d2d2d;
            --color-bg-tertiary: #404040;
            --color-border: #495057;
            --color-shadow: rgba(255, 255, 255, 0.1);
        }
    }
    
    /* Base styles */
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 10px !important;
        background-color: var(--color-bg-primary) !important;
        color: var(--color-text-primary) !important;
    }
    
    .header {
        text-align: center;
        margin-bottom: 20px;
        color: var(--color-text-primary);
    }
    
    .header h1 {
        color: var(--color-text-primary);
        text-shadow: 0 2px 4px var(--color-shadow);
    }
    
    .main-content {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .input-section {
        min-height: 600px;
        padding: 20px;
        background: var(--color-bg-secondary) !important;
        border: 1px solid var(--color-border);
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px var(--color-shadow);
    }
    
    .input-section h3 {
        color: var(--color-text-primary);
        margin-bottom: 20px;
        border-bottom: 2px solid var(--color-primary);
        padding-bottom: 10px;
    }
    
    .results-section {
        width: 100% !important;
        min-height: 800px;
    }
    
    .plot-container {
        width: 100% !important;
        height: 700px !important;
        margin: 20px 0;
        background: var(--color-bg-primary);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px var(--color-shadow);
    }
    
    .summary-container {
        width: 100% !important;
        padding: 20px;
        background: var(--color-bg-secondary) !important;
        border: 1px solid var(--color-border);
        border-radius: 10px;
        box-shadow: 0 2px 8px var(--color-shadow);
    }
    
    /* Summary styling */
    .summary-header {
        background: linear-gradient(135deg, var(--color-primary), var(--color-info));
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        text-align: center;
    }
    
    .summary-header h2 {
        margin: 0;
        color: white;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: var(--color-bg-tertiary);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid var(--color-primary);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--color-shadow);
    }
    
    .metric-card h3 {
        margin: 0;
        color: var(--color-primary);
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        margin: 10px 0;
        font-size: 24px;
        font-weight: bold;
        color: var(--color-text-primary);
    }
    
    .performance-section {
        background: var(--color-bg-tertiary);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid var(--color-border);
    }
    
    .performance-section h3 {
        color: var(--color-text-primary);
        margin-bottom: 15px;
        border-bottom: 2px solid var(--color-primary);
        padding-bottom: 5px;
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    
    .performance-item {
        padding: 10px;
        background: var(--color-bg-secondary);
        border-radius: 5px;
        border: 1px solid var(--color-border);
    }
    
    .performance-item strong {
        color: var(--color-text-primary);
    }
    
    .disclaimer-section {
        background: var(--color-bg-tertiary);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid var(--color-warning);
        border: 1px solid var(--color-border);
    }
    
    .disclaimer-section h4 {
        color: var(--color-warning);
        margin-bottom: 10px;
    }
    
    .disclaimer-section p {
        margin: 0;
        color: var(--color-text-secondary);
        line-height: 1.6;
    }
    
    /* Tips section styling */
    .tips-section {
        margin-top: 20px;
        padding: 15px;
        background: var(--color-bg-tertiary) !important;
        border: 1px solid var(--color-border);
        border-radius: 8px;
    }
    
    .tips-section h4 {
        color: var(--color-text-primary);
        margin-bottom: 10px;
    }
    
    .tips-section ul {
        margin: 10px 0;
        padding-left: 20px;
        color: var(--color-text-secondary);
    }
    
    .tips-section li {
        margin-bottom: 5px;
        line-height: 1.5;
    }
    
    .tips-section strong {
        color: var(--color-text-primary);
    }
    
    /* Button styling */
    .gradio-button {
        background: linear-gradient(135deg, var(--color-primary), var(--color-info)) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .gradio-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px var(--color-shadow) !important;
    }
    
    /* Form controls */
    .gradio-textbox, .gradio-dropdown, .gradio-slider {
        background: var(--color-bg-primary) !important;
        border: 1px solid var(--color-border) !important;
        color: var(--color-text-primary) !important;
    }
    
    .gradio-textbox:focus, .gradio-dropdown:focus {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
    }
    
    /* Labels */
    .gradio-label {
        color: var(--color-text-primary) !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 5px !important;
        }
        
        .input-section {
            min-height: auto;
            padding: 15px;
        }
        
        .plot-container {
            height: 500px !important;
        }
        
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        
        .performance-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Ensure full viewport */
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background-color: var(--color-bg-primary);
        color: var(--color-text-primary);
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="header">
        <h1>📈 Stock Price Predictor with LSTM</h1>
        <p>Predict stock prices using Long Short-Term Memory (LSTM) neural networks</p>
    </div>
    """)
    
    with gr.Column(elem_classes="main-content"):
        # Input Section
        with gr.Row():
            with gr.Column(scale=1, elem_classes="input-section"):
                gr.HTML("<h3>📊 Input Parameters</h3>")
                
                symbol_input = gr.Textbox(
                    label="Stock Symbol",
                    placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
                    value="AAPL",
                    info="Enter the stock ticker symbol"
                )
                
                period_input = gr.Dropdown(
                    label="Data Period",
                    choices=["1y", "YTD"],
                    value="1y",
                    info="Select the period for historical data"
                )
                
                prediction_days = gr.Slider(
                    label="Prediction Days",
                    minimum=1,
                    maximum=60,
                    value=30,
                    step=1,
                    info="Number of days to predict into the future"
                )
                
                epochs = gr.Slider(
                    label="Training Epochs",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10,
                    info="Number of training epochs (more = better accuracy, slower training)"
                )
                
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=16,
                    maximum=128,
                    value=32,
                    step=16,
                    info="Training batch size"
                )
                
                predict_btn = gr.Button(
                    "🔮 Predict Stock Price",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                
                # Tips section
                gr.HTML("""
                <div class="tips-section">
                    <h4>💡 Tips for Better Predictions:</h4>
                    <ul>
                        <li><strong>Popular Stocks:</strong> AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA</li>
                        <li><strong>Training:</strong> More epochs = better accuracy but slower training</li>
                        <li><strong>Data Period:</strong> 1y provides more data for training</li>
                        <li><strong>Prediction Days:</strong> Shorter predictions are generally more accurate</li>
                    </ul>
                </div>
                """)
        
        # Results Section - Full Width
        with gr.Column(elem_classes="results-section"):
            gr.HTML("<h3 style='text-align: center; margin: 20px 0; color: var(--color-text-primary);'>📈 Prediction Results</h3>")
            
            # Plot takes full width
            plot_output = gr.Plot(
                label="Stock Price Prediction",
                show_label=False,
                elem_classes="plot-container"
            )
            
            # Summary below plot
            summary_output = gr.HTML(
                label="Summary",
                show_label=False,
                elem_classes="summary-container"
            )
    
    # Event handlers
    predict_btn.click(
        fn=predict_stock,
        inputs=[symbol_input, period_input, prediction_days, epochs, batch_size],
        outputs=[plot_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )
