import warnings
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
import streamlit as st
from ray.tune import ExperimentAnalysis
import ray

warnings.simplefilter(action="ignore", category=FutureWarning)
ray.init(ignore_reinit_error=True)

def load_tunelogs_data() -> pd.DataFrame:
    """
    Loads the Ray results from a specified directory and returns it as a list of Pandas DataFrames.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset.
    """
    tune_dir = Path("models/ray").resolve()
    if not tune_dir.exists():
        logger.warning('Model data directory does not exist. Check your tune directory path')
        return

    tunelogs = [d for d in tune_dir.iterdir()]
    tunelogs.sort()
    tune_logs_df = dict()
    for log in tunelogs:
        name = Path(log).name
        analysis = ExperimentAnalysis(log)


    
    print(tune_logs_df.columns)  # Prints column names for debugging purposes
    return tune_logs_df


def main() -> None:
    """
    Main function for running the Streamlit app.

    This function manages the user interface and controls the different plot types
    that can be displayed (Scatterplot, Histogram, Boxplot). The function checks
    the user's plot selection and renders the appropriate visualization.
    It also ensures that the WhatsApp dataset is loaded and cached in the session.

    Returns:
        None
    """
    if "whatsapp" not in st.session_state:
        """
        Caches the WhatsApp dataset in session_state to avoid reloading it every time the user interacts
        with the dashboard. The dataset is loaded only once and reused for all subsequent interactions.
        """
        st.session_state.tunelogs = load_tunelogs_data()

    st.title("Tunelogs Dashboard")

    plot_type = st.radio(
        "Choose a Plot Type", 
        ["Scatterplot", "Histogram", "Boxplot"],
        key="plot_type"  # Unique key for the radio button
    )

    # Scatterplot condition
    if plot_type == "Scatterplot":
        option1 = st.selectbox(
            "Select the x-axis",
            st.session_state.whatsapp.columns,
            index=5,  # Default item is author
            key="x_axis_scatter"  # Unique key for this selectbox
        )
        option2 = st.selectbox(
            "Select the y-axis",
            st.session_state.whatsapp.columns,
            index=13,
            key="y_axis_scatter"  # Unique key for this selectbox
        )
        color = st.selectbox(
            "Select the color", 
            st.session_state.whatsapp.columns, 
            index=9,
            key="color_scatter"  # Unique key for this selectbox
        )

        # Plotting the scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=st.session_state.whatsapp, x=option1, y=option2, hue=color)
        plt.xticks(rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper right')
        st.pyplot(fig)

    # Histogram condition
    elif plot_type == "Histogram":
        option = st.selectbox(
            "Select variable for histogram",
            st.session_state.whatsapp.columns,
            index=4,
            key="histogram_var"  # Unique key for this selectbox
        )
        print(option)  # Debugging output
        # Plotting the histogram
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.whatsapp[option], kde=True)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    # Boxplot condition
    elif plot_type == "Boxplot":
        option = st.selectbox(
            "Select variable for boxplot",
            st.session_state.whatsapp.columns,
            index=11,
            key="boxplot_var"  # Unique key for this selectbox
        )
        # Plotting the boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x="author", y=option, data=st.session_state.whatsapp)
        
        # Rotate the x-axis labels (author names) for better readability
        plt.xticks(rotation=45, ha="right")  # 45 degrees rotation, horizontally aligned to the right
       
        # Display the plot in the Streamlit app
        st.pyplot(fig)


if __name__ == "__main__":
    main()