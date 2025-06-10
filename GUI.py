import tkinter as tk
from tkinter import messagebox
import pandas as pd
import torch
from SSP2_New import DQNAgent, StockTradingEnv  # Assuming DQNAgent and StockTradingEnv are defined in your project

# Load the trained model
def load_model():
    checkpoint_path = r"C:/Users/ghani/OneDrive/Desktop/vscode/checkpoints/final_model.pt"  # Update path to your model
    agent = DQNAgent(state_size=19, action_size=3)  # The model expects a state size of 19, based on SSP2.py
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")
    return agent

# Get stock data (using the provided CSV file)
def get_stock_data():
    try:
        # Load the stock data from CSV file (always load fresh data)
        df = pd.read_csv(r'C:/Users/ghani/OneDrive/Desktop/vscode/OGDCL_processed.csv', index_col='Date', parse_dates=True)
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Could not fetch data: {e}")
        return None

# Prepare the stock data for prediction
def prepare_data_for_prediction(df, balance, shares_held):
    try:
        # This function will prepare the necessary state to be passed into the model.
        # We need to extract the 19 features for the model (balance, shares_held, cost_basis, price, and technical indicators)

        # Get the last row of the data (for simplicity)
        last_data = df.iloc[-1]  # Use the most recent data

        # Extract the 19 features (technical indicators + basic features)
        price = last_data['Price']

        # Collect technical indicators from the dataset
        technical_indicators = [
            last_data['SMA_5'], last_data['SMA_20'], last_data['SMA_50'], 
            last_data['EMA_12'], last_data['EMA_26'], last_data['MACD'], 
            last_data['MACD_signal'], last_data['MACD_hist'], last_data['BB_middle'], 
            last_data['BB_std'], last_data['BB_upper'], last_data['BB_lower'], 
            last_data['RSI'], last_data['Daily_Return'], last_data['Volatility']
        ]

        # Return the state as a list with 19 features (balance, shares_held, cost_basis, price, and technical indicators)
        state = [balance, shares_held, 0, price] + technical_indicators
        return state

    except Exception as e:
        messagebox.showerror("Error", f"Error preparing data for prediction: {e}")
        return None

# Predict stock action using the trained agent
def predict_stock():
    global balance, shares_held, df, transaction_fee

    stock_symbol = entry_symbol.get().strip()  # Get stock symbol from user input
    if not stock_symbol:
        messagebox.showerror("Input Error", "Please enter a valid stock symbol!")
        return
    
    df = get_stock_data()  # Re-fetch stock data to ensure the stock price is updated
    if df is not None:
        try:
            state = prepare_data_for_prediction(df, balance, shares_held)
            if state is not None:
                state = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor and add batch dimension

                # Predict the action using the trained agent
                action = agent.act(state, evaluate=False)  # Ensure evaluate=False to allow exploration
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
                predicted_action = action_names[action]

                # Dynamically update balance and shares based on the action taken
                stock_price = df.iloc[-1]['Price']
                if predicted_action == 'BUY':
                    balance -= stock_price + transaction_fee  # Decrease balance by stock price + fee
                    shares_held += 1  # Increase shares held
                elif predicted_action == 'SELL' and shares_held > 0:
                    balance += stock_price - transaction_fee  # Increase balance by stock price - fee
                    shares_held -= 1  # Decrease shares held

                # Calculate net worth, share value, and profit/loss
                net_worth = balance + (shares_held * stock_price)
                share_value = shares_held * stock_price
                profit_loss = net_worth - 10000  # Profit or loss based on initial balance
                cash_percentage = (balance / net_worth) * 100
                shares_percentage = (share_value / net_worth) * 100

                # Display the predicted action and additional info
                result_label.config(text=f"Predicted Action: {predicted_action}\n\n"
                                        f"Stock Price: ${stock_price:.2f}\n"
                                        f"Balance: ${balance:.2f}\n"
                                        f"Shares Held: {shares_held}\n"
                                        f"Net Worth: ${net_worth:.2f}\n"
                                        f"Shares Value: ${share_value:.2f}\n"
                                        f"Profit/Loss: ${profit_loss:.2f}\n"
                                        f"Cash: {cash_percentage:.2f}% | Shares: {shares_percentage:.2f}%")
            else:
                messagebox.showerror("Error", "State preparation failed.")
        except Exception as e:
            messagebox.showerror("Error", f"Error predicting stock: {e}")

# GUI Setup
def setup_gui():
    global result_label, entry_symbol, agent, transaction_fee

    # Initialize the window
    window = tk.Tk()
    window.title("Stock Price Predictor")
    window.geometry("600x400")  # Adjust the size to fit more info
    window.config(bg="#f0f0f0")  # Light background color

    # Create a frame for the input and button section
    frame_input = tk.Frame(window, bg="#f0f0f0")
    frame_input.pack(pady=20)

    # Create widgets
    label_prompt = tk.Label(frame_input, text="Enter Stock Symbol (e.g., OGDCL):", font=("Arial", 12), bg="#f0f0f0")
    label_prompt.grid(row=0, column=0, padx=10)

    entry_symbol = tk.Entry(frame_input, font=("Arial", 12), width=20)
    entry_symbol.grid(row=0, column=1, padx=10)

    button_predict = tk.Button(frame_input, text="Predict Action", command=predict_stock, font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
    button_predict.grid(row=1, column=0, columnspan=2, pady=10)

    # Create a frame for displaying results
    frame_results = tk.Frame(window, bg="#f0f0f0", padx=20, pady=20)
    frame_results.pack(pady=20, padx=20)

    result_label = tk.Label(frame_results, text="Predicted Action: N/A", font=("Arial", 14), bg="#f0f0f0")
    result_label.pack()

    window.mainloop()

# Main Function
if __name__ == "__main__":
    balance = 10000  # Initial balance
    shares_held = 0  # Initial shares held
    df = None  # Placeholder for stock data
    transaction_fee = 0.5  # Example transaction fee (could be adjusted)
    agent = load_model()  # Load the trained agent
    setup_gui()  # Set up the GUI
