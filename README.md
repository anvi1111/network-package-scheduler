# AI Packet Scheduling using Deep Q-Network (DQN)

## 📌 Description

This project implements an intelligent packet scheduling system using Reinforcement Learning, specifically a Deep Q-Network (DQN). The model learns to optimize packet transmission decisions in a network environment to reduce latency and improve overall efficiency.

Unlike traditional scheduling algorithms such as FIFO and Round Robin, this approach dynamically adapts to changing network conditions, enabling more efficient handling of varying traffic loads.

---

## 🎯 Objective

To design a smart scheduling system that minimizes average latency and improves packet delivery performance using machine learning techniques.

---

## ⚙️ Technologies Used

* Python
* Reinforcement Learning (DQN)
* NumPy, TensorFlow / PyTorch

---

## 🚀 Key Features

* Adaptive decision-making using Reinforcement Learning
* Performance comparison with FIFO and Round Robin
* Improved latency and throughput
* Scalable for different network conditions

---

## 🧠 How It Works

The model interacts with a simulated network environment and learns optimal scheduling decisions using a reward-based system. Over time, the Deep Q-Network (DQN) improves its policy by minimizing delays and maximizing efficiency.

---

## 📊 Results

The model demonstrated improved performance over traditional scheduling algorithms by:

* Reducing average latency
* Optimizing packet handling efficiency

---

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python main.py
```

---

## 📁 Project Structure

* `main.py` → Entry point of the program
* `dqn_model.py` → Deep Q-Network implementation
* `env.py` → Simulation environment
* `requirements.txt` → Dependencies
* `README.md` → Documentation

---

## 🔗 Future Improvements

* Implement PPO for enhanced performance
* Add real-time visualization of packet flow
* Deploy as a web-based simulation tool
