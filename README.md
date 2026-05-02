# AI Packet Scheduling using Deep Q-Network (DQN)

## 📌 Description

This project implements an intelligent packet scheduling system using Reinforcement Learning, specifically a Deep Q-Network (DQN). The model learns to optimize packet transmission decisions in a network environment to reduce latency and improve overall efficiency.

Unlike traditional scheduling algorithms such as FIFO and Round Robin, this approach adapts dynamically based on network conditions, making it more efficient in handling varying traffic loads.

## 🎯 Objective

To design a smart scheduling system that minimizes average latency and improves packet delivery performance using machine learning techniques.

## ⚙️ Technologies Used

* Python
* Reinforcement Learning (DQN)
* NumPy / TensorFlow or PyTorch

## 🚀 Key Features

* Adaptive decision-making using RL
* Performance comparison with FIFO and Round Robin
* Improved latency and throughput
* Scalable for different network conditions

## 📊 Results

The model demonstrated better performance compared to traditional scheduling algorithms by reducing average latency and optimizing packet handling efficiency.

## ▶️ How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the project:
   python main.py

## 📁 Project Structure

* model.py / main.py → Core implementation
* environment.py → Simulation environment
* utils.py → Helper functions
* README.md → Documentation

## 🔗 Future Improvements

* Implement PPO for better performance
* Add real-time visualization
* Deploy as a web-based simulation tool
