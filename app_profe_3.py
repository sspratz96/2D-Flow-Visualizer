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
import emoji

import streamlit as st

st.set_page_config(page_title="App para Profe xd", layout="wide")
# Menú lateral
st.sidebar.title("Menú Principal")
opcion = st.sidebar.radio("Ir a:", ["🏠 Inicio", "Graficador (temp)", "📊 Visualizador 2D", "📚 Acerca de"])

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


    
        

        