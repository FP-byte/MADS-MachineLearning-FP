import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from ray.tune import ExperimentAnalysis
import ray
from logs_loader import load_tunelogs_data

warnings.simplefilter(action="ignore", category=FutureWarning)
ray.init(ignore_reinit_error=True)


def main() -> None:
    """
    Main function for running the Streamlit app.

    This function manages the user interface and renders different plots
    (Scatterplot, Histogram, Boxplot) based on user input. It ensures
    the experiment data is loaded only once.
    """
    # Load dataset into session state
    if "tunelogs" not in st.session_state:
        st.session_state.tunelogs = load_tunelogs_data()

    st.title("Ray Tune Logs Dashboard")

    # Ensure data is not empty
    if st.session_state.tunelogs.empty:
        st.warning("No data available. Check your data directory.")
        return

    # Select plot type
    plot_type = st.radio(
        "Choose a Plot Type", 
        ["Scatterplot", "Histogram", "Boxplot"],
        key="plot_type"
    )

    # Scatterplot
    if plot_type == "Scatterplot":
        x_axis = st.selectbox("Select the x-axis", st.session_state.tunelogs.columns, key="x_axis_scatter")
        y_axis = st.selectbox("Select the y-axis", st.session_state.tunelogs.columns, key="y_axis_scatter")
        color = st.selectbox("Select the color (categorical preferred)", st.session_state.tunelogs.columns, key="color_scatter")

        # Plot scatterplot
        fig, ax = plt.subplots()
        try:
            sns.scatterplot(data=st.session_state.tunelogs, x=x_axis, y=y_axis, hue=color, palette="tab10")
            plt.xticks(rotation=45, ha="right")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to plot scatterplot: {e}")

    # Histogram
    elif plot_type == "Histogram":
        hist_var = st.selectbox("Select a variable for the histogram", st.session_state.tunelogs.columns, key="histogram_var")
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        # Plot histogram
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.tunelogs[hist_var], bins=bins, kde=True)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    # Boxplot
    elif plot_type == "Boxplot":
        y_var = st.selectbox("Select variable for boxplot (Y-axis)", st.session_state.tunelogs.columns, key="boxplot_y")
        x_var = st.selectbox("Select grouping variable (X-axis)", st.session_state.tunelogs.columns, key="boxplot_x")

        # Plot boxplot
        fig, ax = plt.subplots()
        try:
            sns.boxplot(x=x_var, y=y_var, data=st.session_state.tunelogs)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to plot boxplot: {e}")


if __name__ == "__main__":
    main()
