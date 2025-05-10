# 🧪 Interactive Math Visualizer

A multi-tool web app built using **Streamlit** to visualize and simulate various mathematical and physical models interactively — perfect for educational or exploratory purposes.

## 🚀 Live App

You can try the app directly here:  
👉 [2D Flow Visualizer on Streamlit Cloud](https://modelling-control-of-dynamical-systems---personal-project-5bmt.streamlit.app/)

## 📦 Features

### 1. 📈 Parametric Equation Plotter
- Input any expression in `x` with parameters (e.g., `a*x**2 + b*x + c`)
- Adjust parameters via sliders
- Plot updates in real time
- Displays symbolic derivative and real roots (if no parameters)

### 2. 🧮 Euler Method Visualizer
- Solves and compares:
  \[
  rac{dx}{dt} = -a \cdot x
  \]
- Visualizes exact vs Euler method solutions
- Shows cumulative error

### 3. 🧭 Kuramoto Oscillator Simulation
- Simulates coupled oscillators with:
  - Custom \(\omega_i\), \(	heta_0^i\), and \(K_i\)
  - Euler-based solver
- Animated phase diagram (circular layout)

### 4. 📊 2D Dynamical System Simulator
- Input any 2D system (e.g., dx/dt = r*x - x³ - y)
- Symbolic fixed-point classification (saddle, node, focus, etc.)
- Phase portrait (stream plot)
- Optional animated flow simulation

## 🔧 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run graficador.py
```

> 💡 Make sure your terminal is in the same folder as `graficador.py`.

## 📂 Structure Overview

```text
graficador.py         # Main Streamlit app
README.md             # This file
requirements.txt      # Python dependencies (to be created)
```

## 📚 About

- Author: *Tu Nombre*
- GitHub: [Your GitHub Repo](https://github.com/tu_usuario/tu_repo)

---

Made with ❤️ using [Streamlit](https://streamlit.io/)
