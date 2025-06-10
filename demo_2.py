import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import time

# DQN Network Architecture
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Action Space (Buy, Sell, Hold)
class StockTradingEnv(gym.Env):
    """Custom Environment for stock trading."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001, verbose=False):
        super(StockTradingEnv, self).__init__()
        
        # Data validation and preprocessing
        self.df = df.copy()
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        self.df = self.df.fillna(0)  # Final fallback
        self.verbose = verbose  # Control verbose output
        
        # Ensure Price column exists and has valid values
        if 'Price' not in self.df.columns:
            raise ValueError("Price column not found in dataframe")
        
        # Remove rows with invalid prices
        self.df = self.df[self.df['Price'] > 0]
        
        if len(self.df) < 10:
            raise ValueError("Not enough valid data points")
        
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_range = (-1, 1)  # Normalized reward range
        self.action_space = spaces.Discrete(3)

        # Indicator Columns
        self.tech_indicator_cols = [col for col in df.columns if col not in ['Price', 'Open', 'High', 'Low']]
        self.num_features = 4 + len(self.tech_indicator_cols)
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(self.num_features,), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()

    def _get_observation(self):
        """Get current observation with proper bounds checking and normalization."""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        current_data = self.df.iloc[self.current_step]
        current_price = float(current_data['Price'])
        
        # Validate current price
        if np.isnan(current_price) or current_price <= 0:
            current_price = self.df['Price'].median()
        
        tech_indicators = current_data[self.tech_indicator_cols].values.astype(float)
        tech_indicators = np.nan_to_num(tech_indicators, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize observation components
        balance_norm = np.clip(self.balance / self.initial_balance, 0, 10)
        shares_norm = np.clip(self.shares_held, 0, 10)
        cost_basis_norm = np.clip(self.cost_basis / current_price if current_price > 0 else 0, 0, 5)
        price_norm = np.clip(current_price / 100, 0, 10)  # Assuming price is in reasonable range
        
        # Clip technical indicators
        tech_indicators = np.clip(tech_indicators, -5, 5)
        
        obs = np.array([
            balance_norm, 
            shares_norm, 
            cost_basis_norm, 
            price_norm, 
            *tech_indicators
        ], dtype=np.float32)
        
        # Final validation
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs

    def _calculate_reward(self, action, previous_net_worth):
        """Calculate reward with proper normalization and bounds."""
        current_price = self.df.iloc[self.current_step]['Price']
        current_net_worth = self.balance + self.shares_held * current_price
        
        # Validate net worth
        if np.isnan(current_net_worth) or np.isinf(current_net_worth):
            current_net_worth = self.initial_balance
        
        # Calculate percentage change
        if previous_net_worth > 0:
            reward = (current_net_worth - previous_net_worth) / previous_net_worth
        else:
            reward = 0
        
        # Add transaction cost penalty
        if action != 0:  # Buy or Sell action
            reward -= self.transaction_fee_percent
        
        # Normalize and clip reward
        reward = np.clip(reward, -1, 1)
        
        return float(reward)

    def step(self, action):
        """Execute one step with robust error handling."""
        
        # Force termination if we've exceeded reasonable bounds
        if self.current_step >= len(self.df) - 1:
            obs = self._get_observation()
            return obs, 0, True, {}
        
        # Get current state
        current_price = float(self.df.iloc[self.current_step]['Price'])
        previous_net_worth = self.balance + self.shares_held * current_price
        
        # Validate price
        if np.isnan(current_price) or current_price <= 0:
            current_price = self.df['Price'].median()
        
        executed_action = action
        action_successful = False
        
        try:
            if action == 1:  # Buy
                # Calculate maximum shares we can buy
                available_funds = self.balance * 0.99  # Leave small buffer
                cost_per_share = current_price * (1 + self.transaction_fee_percent)
                
                if available_funds >= cost_per_share and available_funds > 1:
                    shares_to_buy = available_funds / cost_per_share
                    total_cost = shares_to_buy * cost_per_share
                    
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    self.cost_basis = current_price
                    action_successful = True
                    
                    if self.verbose:
                        print(f"Step {self.current_step}: BUY - Shares: {shares_to_buy:.4f}, Cost: ${total_cost:.2f}")
                else:
                    executed_action = 0  # Convert to hold
                    
            elif action == 2 and self.shares_held > 0.0001:  # Sell (with minimum threshold)
                sales_value = self.shares_held * current_price * (1 - self.transaction_fee_percent)
                sold_shares = self.shares_held
                
                self.balance += sales_value
                self.shares_held = 0
                self.cost_basis = 0
                action_successful = True
                
                if self.verbose:
                    print(f"Step {self.current_step}: SELL - Shares: {sold_shares:.4f}, Value: ${sales_value:.2f}")
            else:
                executed_action = 0  # Hold
                
        except Exception as e:
            if self.verbose:
                print(f"Error in action execution: {e}")
            executed_action = 0
        
        # Calculate reward
        reward = self._calculate_reward(executed_action, previous_net_worth)
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        done = (self.current_step >= len(self.df) - 1)
        
        # Get next observation
        obs = self._get_observation()
        
        # Update history with validation
        current_net_worth = self.balance + self.shares_held * current_price
        if np.isnan(current_net_worth) or np.isinf(current_net_worth):
            current_net_worth = self.initial_balance
            
        self.net_worth_history.append(current_net_worth)
        self.balance_history.append(self.balance)
        self.shares_history.append(self.shares_held)
        self.price_history.append(current_price)
        self.action_history.append(executed_action)
        
        return obs, reward, done, {}

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.cost_basis = 0.0
        
        # Initialize history
        initial_price = float(self.df.iloc[0]['Price'])
        if np.isnan(initial_price) or initial_price <= 0:
            initial_price = self.df['Price'].median()
            
        self.net_worth_history = [self.initial_balance]
        self.balance_history = [self.initial_balance]
        self.shares_history = [0.0]
        self.price_history = [initial_price]
        self.action_history = [0]
        
        return self._get_observation()

    def render(self, mode='human', close=False):
        """Display final results."""
        if len(self.price_history) > 0:
            current_price = self.price_history[-1]
            net_worth = self.balance + self.shares_held * current_price
            profit = net_worth - self.initial_balance
            
            print(f"\n=== Final Results ===")
            print(f"Final balance: ${self.balance:.2f}")
            print(f"Final shares held: {self.shares_held:.4f}")
            print(f"Final share price: ${current_price:.2f}")
            print(f"Final net worth: ${net_worth:.2f}")
            print(f"Profit: ${profit:.2f} ({profit / self.initial_balance * 100:.2f}%)")
        else:
            print("No trading history available")

# DQN Agent with improved stability
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        print(f"DQN Agent initialized on {self.device}")

    def update_target_model(self):
        """Update target model weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, evaluate=False):
        """Choose action using epsilon-greedy policy."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
            return action
        except Exception as e:
            print(f"Error in action selection: {e}")
            return 0  # Default to hold

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0
            
        try:
            batch = random.sample(self.memory, batch_size)
            
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
            
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {e}")
            return 0

    def load(self, name):
        """Load model weights."""
        try:
            if os.path.exists(name):
                state_dict = torch.load(name, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.target_model.load_state_dict(state_dict)
                print(f"Model loaded from {name}")
            else:
                print(f"Model file {name} not found")
        except Exception as e:
            print(f"Error loading model: {e}")

    def save(self, name):
        """Save model weights."""
        try:
            torch.save(self.model.state_dict(), name)
            print(f"Model saved to {name}")
        except Exception as e:
            print(f"Error saving model: {e}")

def train_agent(train_data):
    """Train the DQN agent on training data ONLY."""
    print("=== TRAINING PHASE ===")
    print(f"Training on {len(train_data)} data points (80% of total data)")
    
    # Create training environment (non-verbose)
    env = StockTradingEnv(train_data, verbose=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 10
    batch_size = 32
    max_steps_per_episode = len(train_data) - 1
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        try:
            state = env.reset()
            total_reward = 0
            steps = 0
            
            # Progress tracking
            progress_interval = max(1, max_steps_per_episode // 10)  # Show progress 10 times per episode
            
            # Episode loop with minimal printing
            while steps < max_steps_per_episode:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train every few steps
                if len(agent.memory) > batch_size and steps % 10 == 0:
                    loss = agent.replay(batch_size)
                
                # Show progress only occasionally
                if steps % progress_interval == 0:
                    print(f"Episode {episode+1}: Step {steps}/{max_steps_per_episode} ({steps/max_steps_per_episode*100:.0f}%)")
                
                # Force episode end
                if done or steps >= max_steps_per_episode:
                    break
            
            # Update target network
            if (episode + 1) % 5 == 0:
                agent.update_target_model()
            
            episode_rewards.append(total_reward)
            final_net_worth = env.net_worth_history[-1] if env.net_worth_history else env.initial_balance
            
            print(f"\nEpisode {episode+1}/{episodes} completed:")
            print(f"  Total Reward: {total_reward:.4f}")
            print(f"  Final Net Worth: ${final_net_worth:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
            continue
    
    print(f"\nTraining completed!")
    print(f"Average reward: {np.mean(episode_rewards):.4f}")
    
    return agent

def test_agent_and_save_predictions(agent, test_data, output_file='test_predictions.csv'):
    """Test the agent on test data and save predictions to CSV (silent operation)."""
    print("\n=== TESTING PHASE ===")
    print(f"Testing on {len(test_data)} data points (20% of total data)")
    
    # Create test environment (NON-verbose for silent operation)
    test_env = StockTradingEnv(test_data, verbose=False)
    
    # Reset environment
    state = test_env.reset()
    
    # Lists to store predictions and results
    dates = []
    prices = []
    predictions = []
    actions_taken = []
    balances = []
    shares_held = []
    net_worths = []
    
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    print("Running silent testing (no step-by-step output)...")
    
    step = 0
    while True:
        # Get prediction from agent (evaluation mode - no exploration)
        predicted_action = agent.act(state, evaluate=True)
        
        # Store current state info
        current_date = test_data.index[step]
        current_price = test_data.iloc[step]['Price']
        
        dates.append(current_date)
        prices.append(current_price)
        predictions.append(action_names[predicted_action])
        
        # Execute action (silently)
        next_state, reward, done, _ = test_env.step(predicted_action)
        
        # Store results after action
        actions_taken.append(action_names[test_env.action_history[-1]])
        balances.append(test_env.balance)
        shares_held.append(test_env.shares_held)
        net_worths.append(test_env.net_worth_history[-1])
        
        state = next_state
        step += 1
        
        if done or step >= len(test_data):
            break
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'AI_Prediction': predictions,
        'Action_Taken': actions_taken,
        'Balance': balances,
        'Shares_Held': shares_held,
        'Net_Worth': net_worths
    })
    
    # Set Date as index
    results_df.set_index('Date', inplace=True)
    
    # Save to CSV with error handling for file permissions
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            results_df.to_csv(output_file)
            print(f"Predictions saved to {output_file}")
            break
        except PermissionError:
            if attempt < max_attempts - 1:
                print(f"File {output_file} is in use. Waiting 2 seconds and trying again... (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                # Try alternative filename
                timestamp = int(time.time())
                alt_filename = f"test_predictions_{timestamp}.csv"
                try:
                    results_df.to_csv(alt_filename)
                    print(f"Original file was locked. Predictions saved to {alt_filename}")
                    output_file = alt_filename
                except Exception as e:
                    print(f"Error saving predictions: {e}")
                    return None
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return None
    
    # Show final results
    test_env.render()
    
    # Show prediction summary
    prediction_counts = results_df['AI_Prediction'].value_counts()
    print(f"\nPrediction Summary:")
    for action, count in prediction_counts.items():
        print(f"  {action}: {count} days ({count/len(results_df)*100:.1f}%)")
    
    return results_df

def main():
    """Main function to train and evaluate the agent."""
    print("Starting Stock Trading RL Agent Training and Evaluation")
    
    # Load and validate data
    try:
        print("Loading data...")
        df = pd.read_csv('OGDCL_processed.csv', index_col='Date', parse_dates=True)
        
        # Data validation
        if 'Price' not in df.columns:
            print("Error: 'Price' column not found")
            return None
            
        print(f"Loaded {len(df)} data points")
        print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        if len(df) < 50:
            print("Error: Not enough clean data points")
            return None
            
        # STRICT 80-20 Split - Training data is completely hidden from testing
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size].copy()  # First 80% for training
        test_data = df.iloc[train_size:].copy()   # Last 20% for testing (hidden during training)
        
        print(f"\n=== DATA SPLIT ===")
        print(f"Total data points: {len(df)}")
        print(f"Training data: {len(train_data)} points ({len(train_data)/len(df)*100:.1f}%)")
        print(f"Testing data: {len(test_data)} points ({len(test_data)/len(df)*100:.1f}%)")
        print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Ensure no data leakage
        if len(set(train_data.index).intersection(set(test_data.index))) > 0:
            print("ERROR: Data leakage detected! Training and testing data overlap.")
            return None
        else:
            print("✓ No data leakage - Training and testing data are completely separate")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Train agent ONLY on training data
    agent = train_agent(train_data)
    
    # Save trained model
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(f"{checkpoint_dir}/final_model.pt")
    
    # Test agent on completely unseen testing data
    results_df = test_agent_and_save_predictions(agent, test_data, 'OGDCL_test_predictions.csv')
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Files created:")
    print("1. checkpoints/final_model.pt - Trained model")
    print("2. OGDCL_test_predictions.csv - Test predictions and results")
    print("\nKey Changes Made:")
    print("- Removed verbose output during testing")
    print("- Added file permission error handling")
    print("- Ensured strict 80-20 data split with no leakage")
    print("- Training only uses first 80% of data")
    print("- Testing only uses last 20% of data (completely unseen during training)")
    
    return agent, results_df

if __name__ == "__main__":
    agent, results = main()