# Stock-Price-Predictor
This project implements a **Reinforcement Learning** (RL) agent to predict stock trading actions (Buy, Sell, or Hold) based on historical market data. The agent is trained using a **Deep Q-Network (DQN)** model and optimized using **Particle Swarm Optimization (PSO)** for hyperparameter tuning.

## Project Description

The agent interacts with a simulated stock market environment to decide the best course of action for each given state. The objective of the agent is to maximize its profit by making appropriate stock trading decisions over time.

This repository contains three main scripts:
1. **demo_2.py**: The original version of the agent with basic functionality.
2. **SSP2.py**: The improved version of the agent, optimized using PSO and with various enhancements for performance and efficiency.
3. **GUI.py**: A graphical user interface (GUI) that allows users to visualize the stock trading process and interact with the agent.

## Files

### `demo_2.py`
This is the **original version** of the stock trading RL agent. It implements a basic **Deep Q-Network (DQN)** agent, where the agent learns to make decisions based on stock price data. The code is used for training the agent without any hyperparameter optimization and uses **hardcoded** values for training.

### `SSP2.py`
This is the **optimized version** of the stock trading agent. The improvements include:
- **PSO optimization**: The hyperparameters of the DQN agent, including learning rate, gamma, epsilon, and others, are optimized using **Particle Swarm Optimization (PSO)**.
- **Efficient training process**: The number of episodes for training is reduced, and a **time limit** is set for the fitness function to speed up PSO evaluations.
- **Improved exploration-exploitation balance**: The agent's exploration vs. exploitation is optimized through **epsilon tuning** via PSO.

### `GUI.py`
The **GUI** provides an interface for users to visualize the agent's stock trading actions. It allows users to interact with the trained agent, view predictions, and observe the agent's decisions in real-time.

## How to Run the Code

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/stock-trading-rl-agent.git
    cd stock-trading-rl-agent
    ```

2. **Install Dependencies**:
    Make sure you have **Python 3.x** installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**:

    - To run the original agent (`demo_2.py`):
      ```bash
      python demo_2.py
      ```
    
    - To run the optimized agent (`SSP2.py`):
      ```bash
      python SSP2.py
      ```

    - To run the graphical user interface (`GUI.py`):
      ```bash
      python GUI.py
      ```

## Dependencies

- **NumPy**: For numerical operations.
- **Pandas**: For data handling and manipulation.
- **TensorFlow/PyTorch**: For implementing the neural network (Deep Q-Network).
- **Matplotlib**: For plotting graphs and visualizations.
- **PyQt5**: For building the GUI interface.
- **SciPy**: For PSO and other scientific calculations.

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
stock-trading-rl-agent/
├── demo_2.py           # Original version of the agent
├── SSP2.py             # Optimized version of the agent using PSO
├── GUI.py              # Graphical User Interface for visualizing stock trading decisions
├── requirements.txt    # List of required dependencies
├── README.md           # This file
└── checkpoints/        # Folder for saving model checkpoints
```

## Results

- The **original model** (`demo_2.py`) shows a **positive profit** after testing, but the training process is slow and inefficient.
- The **optimized model** (`SSP2.py`) has **improved training stability** and **faster convergence** through **Particle Swarm Optimization (PSO)** for hyperparameter tuning, but still requires further training to improve its performance.

## Future Improvements

1. **Optimize the reward function** further to improve the agent's decision-making process.
2. **Increase the number of training episodes** for the agent to learn better in real-world environments.
3. **Use more advanced techniques** like **Double Q-Learning** or **Experience Replay** to enhance the model's performance.
4. **Extend the GUI** to show more detailed performance metrics and allow more customization for users.

## Contributors

 Mustafa Ghani (Mustafa-nox123) - *Project lead and developer*
 Aqib Mahar

Feel free to contribute by forking the repository, opening issues, or submitting pull requests.

