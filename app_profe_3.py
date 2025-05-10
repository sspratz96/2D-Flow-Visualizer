# para correr este codigo (local) en la consola escribir:
# streamlit run graficador.py
# asegurarse de que la consola esta primero en la carpeta correcta !!

import streamlit as st
from sympy import symbols, sympify, lambdify, diff, solveset, S
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pathlib import Path
import tempfile
from PIL import Image
import sympy as sp
import os
import emoji

import streamlit as st

st.set_page_config(page_title="App para Profe xd", layout="wide")
# Menú lateral
st.sidebar.title("Menú Principal")
opcion = st.sidebar.radio("Ir a:", ["🏠 Inicio", "Graficador (temp)", "Metodo de Euler", "🧭 Osciladores Acoplados", "📊 Visualizador 2D", "📚 Acerca de"])

# Mostrar contenido según la selección
if opcion == "🏠 Inicio":
    st.title("Bienvenido 👋")
    st.write("Esta es la página de inicio de la app.")

elif opcion == "Graficador (temp)":
    st.title("📈 Graficador de Ecuaciones con Parámetros")

    input_expr = st.text_input("Introduce una ecuación en x con parámetros (ej: a*x**2 + b*x + c):", "a*x**2 + b*x + c")
    x = symbols('x')

    if input_expr:
        try:
            expr = sympify(input_expr.replace('^', '**'))
            parametros = sorted(expr.free_symbols - {x}, key=lambda s: str(s))

            valores = {}
            st.sidebar.subheader("🎚 Ajusta los parámetros")

            for p in parametros:
                valores[p] = st.sidebar.slider(f"{p}", -10.0, 10.0, 1.0)

            expr_evaluada = expr.subs(valores)
            f = lambdify(x, expr_evaluada, modules=["numpy"])
            x_vals = np.linspace(-10, 10, 400)
            y_vals = f(x_vals)

            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label=f"y = {expr_evaluada}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Gráfico interactivo")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            if not parametros:
                st.subheader("Análisis simbólico")
                derivada = diff(expr, x)
                st.latex(f"\\frac{{dy}}{{dx}} = {derivada}")
                raices = solveset(expr, x, domain=S.Reals)
                st.write("Raíces reales:")
                st.write(raices)

        except Exception as e:
            st.error(f"Error al procesar la expresión: {e}")

elif opcion == "Metodo de Euler":
    st.title("🧪 Comparación entre solución exacta y método de Euler")

    # st.markdown("""
    # Esta herramienta permite resolver la ecuación diferencial:

    # \[
    # \\frac{dx}{dt} = -a \\cdot x
    # \]

    # y comparar la solución exacta con la aproximación numérica usando el **método de Euler**.
    # """)

    # Parámetros ajustables
    a = st.slider("🔧 Parámetro a", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    dt = st.slider("⏱️ Paso de tiempo (dt)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    t_end = st.slider("📏 Tiempo final", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

    # Cálculo
    N = int(t_end / dt)
    t = np.linspace(0, t_end, N)
    x_exact = np.exp(-a * t)

    # Método de Euler
    x_euler = np.zeros_like(t)
    x_euler[0] = 1
    for i in range(1, len(t)):
        x_euler[i] = x_euler[i-1] + (-a * x_euler[i-1]) * dt

    # Gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, x_exact, label="Solución exacta", color="green")
    ax.plot(t, x_euler, label="Método de Euler", linestyle="--", color="blue")
    ax.fill_between(t, x_exact, x_euler, color="orange", alpha=0.3, label="Error acumulado")
    ax.set_title(f"Comparación con a = {a}, dt = {dt}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

elif opcion == "🧭 Osciladores Acoplados":
    st.title("🧭 Modelo de Osciladores Acoplados (Kuramoto) con Kᵢ")
    st.markdown(r"""
    ### 🧮 Ecuaciones del Modelo

    Este simulador implementa el **modelo de Kuramoto** para \( N \) osciladores acoplados con constantes de acoplamiento individuales \( K_i \).  
    Cada oscilador sigue la dinámica:

    $$
    \frac{d\theta_i}{dt} = \omega_i + \frac{K_i}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)
    $$

    donde:

    - \( \theta_i(t) \) es la fase del oscilador \( i \)
    - \( \omega_i \) es su frecuencia natural
    - \( K_i \) es la constante de acoplamiento que modula la influencia de los demás osciladores sobre el oscilador \( i \). Por 'simplicidad' se asumirá
                que todos los osciladores j distintos de i influencian con la misma constante de acoplamiento K, y no es unica por oscilador.
    - \( N \) es el número total de osciladores

    Se resuelve numéricamente mediante el **método de Euler** con paso de integración \( dt \).
    """)

    with st.sidebar:
        st.header("⚙️ Parámetros")
        N = st.slider("Número de osciladores", 2, 5, 3)
        T = st.slider("Tiempo total de simulación", 5, 60, 20)
        dt = st.slider("Paso temporal (dt)", 0.01, 0.2, 0.05, step=0.01)

        st.markdown("---")
        st.markdown("#### Frecuencias Naturales (ωᵢ)")
        omega = []
        for i in range(N):
            omega.append(st.number_input(f"ω{i+1}", value=float(i), key=f"omega_{i}"))

        st.markdown("#### Ángulos Iniciales (θ₀ᵢ)")
        theta0 = []
        for i in range(N):
            theta0.append(st.slider(f"θ₀{i+1} [rad]", 0.0, 2*np.pi, float(i) * 2*np.pi / N, step=0.1, key=f"theta0_{i}"))

        st.markdown("#### Constantes de Acoplamiento (Kᵢ)")
        K_list = []
        for i in range(N):
            K_list.append(st.number_input(f"K{i+1}", value=1.0, min_value=0.0, step=0.1, key=f"K_{i}"))

    if st.button("▶️ Generar Simulación"):
        # Simular y obtener θ(t)
        steps = int(T / dt)
        t = np.linspace(0, T, steps)
        theta = np.zeros((steps, N))
        theta[0, :] = theta0

        for k in range(1, steps):
            dtheta = np.zeros(N)
            for i in range(N):
                coupling = np.sum(np.sin(theta[k-1, :] - theta[k-1, i]))
                dtheta[i] = omega[i] + (K_list[i] / N) * coupling
            theta[k, :] = theta[k-1, :] + dtheta * dt

        # Mostrar gráfico de fases
        st.markdown("### 📈 Evolución de las fases")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        for i in range(N):
            ax1.plot(t, np.mod(theta[:, i], 2*np.pi), label=f"θ{i+1}")
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel("Fase (mod 2π)")
        ax1.set_title("Evolución de fases acopladas")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Generar animación con spinner
        with st.spinner("🎞️ Generando animación..."):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            filename = tmp_file.name
            tmp_file.close()

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_title("Osciladores Acoplados")
            points, = ax.plot([], [], 'o', markersize=10)

            def update(frame):
                angles = theta[frame, :]
                x = np.cos(angles)
                y = np.sin(angles)
                points.set_data(x, y)
                return points,

            ani = FuncAnimation(fig, update, frames=steps, blit=True, interval=30)
            ani.save(filename, writer='pillow')
            plt.close(fig)

            with open(filename, "rb") as f:
                gif_bytes = f.read()
            st.image(gif_bytes, caption="🔄 Evolución de los osciladores", use_column_width=True)
            os.remove(filename)
    
elif opcion == "📊 Visualizador 2D":

    st.title("🧮 Simulador General de Sistemas 2D")

    # Layout: Izquierda (sliders), Derecha (gráfico)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🧾 Define tu sistema dinámico")
        dx_expr = st.text_input("dx/dt =", value="r * x - x**3 - y")
        dy_expr = st.text_input("dy/dt =", value="(1/t) * (x - y)")

        st.markdown("### ⚙️ Parámetros")
        r = st.slider("Parámetro r", -5.0, 5.0, 1.0, 0.1)
        t_param = st.slider("Parámetro t", 0.1, 5.0, 1.0, 0.1)
        zoom = st.slider("Zoom del campo", 1.0, 10.0, 4.0, 0.5)
        density = st.slider("Densidad de flechas", 10, 40, 20, 2)

        st.markdown("### 🔁 Animación")
        show_animation = st.checkbox("🌪️ Mostrar animación de flujo", value=False)

    # Crear funciones del sistema
    def create_system_func(dx_str, dy_str, r, t_param):
        def f(x, y):
            try:
                dx = eval(dx_str, {"np": np}, {"x": x, "y": y, "r": r, "t": t_param})
                dy = eval(dy_str, {"np": np}, {"x": x, "y": y, "r": r, "t": t_param})
            except Exception:
                dx, dy = 0, 0
            return dx, dy
        return f

    # Clasificación de puntos críticos
    def classify_critical_points(dx_expr_str, dy_expr_str, r_val, t_val):
        x, y, r, t = sp.symbols("x y r t")
        dx_expr_sym = sp.sympify(dx_expr_str)
        dy_expr_sym = sp.sympify(dy_expr_str)
        eqs = [dx_expr_sym, dy_expr_sym]
        critical_points = sp.solve(eqs, (x, y), dict=True)
        J = sp.Matrix([dx_expr_sym, dy_expr_sym]).jacobian([x, y])
        results = []

        for pt in critical_points:
            J_eval = J.subs(pt).subs({r: r_val, t: t_val})
            eigs = J_eval.eigenvals()
            eigenvalues = list(eigs.keys())

            if len(eigenvalues) != 2:
                classification = "Indeterminado"
            else:
                l1, l2 = map(sp.N, eigenvalues)
                if sp.re(l1) > 0 and sp.re(l2) > 0 and sp.im(l1) == 0:
                    classification = emoji.emojize('Fuente (inestable)')
                elif sp.re(l1) < 0 and sp.re(l2) < 0 and sp.im(l1) == 0:
                    classification = emoji.emojize('Sumidero (estable)')
                elif sp.re(l1) * sp.re(l2) < 0:
                    classification = "Silla de montar"
                elif sp.re(l1) < 0 and sp.im(l1) != 0:
                    classification = "Foco estable"
                elif sp.re(l1) > 0 and sp.im(l1) != 0:
                    classification = "Foco inestable"
                elif sp.re(l1) == 0 and sp.im(l1) != 0:
                    classification = "Centro"
                else:
                    classification = "Indeterminado"

            results.append((pt, classification))
        return results

    # Graficar campo vectorial
    def plot_phase(f, zoom, density, criticals):
        x_vals = np.linspace(-zoom, zoom, density)
        y_vals = np.linspace(-zoom, zoom, density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U[i, j], V[i, j] = f(X[i, j], Y[i, j])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.streamplot(X, Y, U, V, color='teal', density=1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("🔁 Retrato de Fase")

        for pt, label in criticals:
            try:
                x0 = float(pt[sp.symbols("x")].subs({"r": r, "t": t_param}))
                y0 = float(pt[sp.symbols("y")].subs({"r": r, "t": t_param}))
                ax.plot(x0, y0, 'ro')
                ax.text(x0 + 0.2, y0 + 0.2, label,
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                        )
            except (TypeError, ValueError):
                continue  # Omitir puntos complejos o indefinidos

        st.pyplot(fig)
        plt.close()

    # Animación

    def generate_animation(f, zoom, n_particles=100, n_frames=60, dt=0.1):
        x_particles = np.random.uniform(-zoom, zoom, n_particles)
        y_particles = np.random.uniform(-zoom, zoom, n_particles)
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter(x_particles, y_particles, c='teal', s=30)
        ax.grid()
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_title("Animación del Campo de Flujo")

        def update(frame):
            nonlocal x_particles, y_particles
            for i in range(n_particles):
                dx, dy = f(x_particles[i], y_particles[i])
                x_particles[i] += dx * dt
                y_particles[i] += dy * dt
            scat.set_offsets(np.c_[x_particles, y_particles])
            return scat,

        ani = FuncAnimation(fig, update, frames=n_frames, interval=100)
        tmp_dir = Path(tempfile.gettempdir())
        gif_path = tmp_dir / "animated_flow.gif"
        ani.save(gif_path, writer='pillow', fps=10)
        plt.close()
        return gif_path

    # Mostrar en la derecha
    with col2:
        system_func = create_system_func(dx_expr, dy_expr, r, t_param)
        critical_info = classify_critical_points(dx_expr, dy_expr, r, t_param)

        if show_animation:
            with st.spinner("Generando animación..."):
                gif_path = generate_animation(system_func, zoom)
                st.image(str(gif_path), caption="Animación del Campo de Flujo", use_column_width=True)
        else:
            plot_phase(system_func, zoom, density, critical_info)

        st.markdown("### 🔹 Puntos Críticos")
        for pt, label in critical_info:
            try:
                x_val = pt[sp.symbols("x")].evalf(subs={"r": r, "t": t_param})
                y_val = pt[sp.symbols("y")].evalf(subs={"r": r, "t": t_param})
                if not (sp.im(x_val) or sp.im(y_val)):
                    st.write(f"{label} en (x = {float(x_val):.2f}, y = {float(y_val):.2f})")
            except (TypeError, ValueError):
                continue

elif opcion == "📚 Acerca de":
    st.title("Sobre esta app")
    st.markdown("""
    Esta app fue creada para visualizar sistemas dinámicos 2D en tiempo real.
    
    - Autor: Tu Nombre  
    - Repositorio: [GitHub](https://github.com/tu_usuario/tu_repo)
    """)


    
        

        