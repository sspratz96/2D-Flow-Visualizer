# Streamlit App for Dynamic Systems

A multi-tool web app built using **Streamlit** to visualize and simulate various mathematical and physical models interactively — perfect for educational or exploratory purposes.

## 🚀 Live App

You can try the app directly here:  
👉 [2D Flow Visualizer on Streamlit Cloud](https://modelling-control-of-dynamical-systems---personal-project-5bmt.streamlit.app/)

## Main App Overview
This Streamlit app is a dynamic system visualization tool that allows the user to interact with different mathematical models and visualize their behavior. The main sections are:

- **Graficador (temp)**: Graphs equations with parameters.
- **Metodo de Euler**: Simulates solutions to differential equations using the Euler method.
- **Osciladores Acoplados**: A model for coupled oscillators.
- **Visualizador 2D**: General 2D dynamic system simulator.
- **Acerca de**: Information about the app.

## Detailed Descriptions

### Graficador (temp)
This section allows you to graph equations with parameters, adjusting their values interactively.

#### Features:
- Input an equation in the form of `a*x**2 + b*x + c`.
- Use sliders to adjust parameter values.
- View both symbolic analysis and graph output.

### Metodo de Euler
Simulate a differential equation of the form `dx/dt = -a*x` and compare the exact solution with the approximation obtained using the **Euler method**.

#### Features:
- Adjustable parameter `a`, time step `dt`, and simulation duration.
- Graph comparing exact solution and Euler approximation.

### Osciladores Acoplados
This part models **Kuramoto coupled oscillators**. You can adjust parameters such as the number of oscillators, their natural frequencies, and coupling constants.

#### Features:
- User can define the number of oscillators, their initial phases, and coupling constants.
- Visualize the evolution of phases and the state of each oscillator over time.
- Generate an animation showing the coupled oscillators.

### Visualizador 2D
A general 2D dynamic system simulator where you can define custom differential equations of the form `dx/dt` and `dy/dt` in the form of symbolic expressions.

#### Features:
- Define custom system equations using symbolic expressions.
- Visualize the vector field of the system.
- Interactive sliders to adjust parameters and zoom in/out.
- Ability to animate the field with multiple particles.

### Acerca de
Information about the app and its creators.

---

## Installation and Usage

To run the app locally:

1. Clone this repository to your local machine:
    ```
    git clone <repository_url>
    ```

2. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```

3. Run the app:
    ```
    streamlit run graficador.py
    ```

Make sure your console is in the correct directory before running the command.

## Features

- Interactive graphing and simulation of dynamic systems.
- Real-time parameter adjustments.
- Symbolic analysis of equations.
- Animation generation for visualizing coupled oscillators and dynamic systems.

## License
This app is released under the MIT License.
