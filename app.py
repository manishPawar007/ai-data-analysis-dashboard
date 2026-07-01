import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import io

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="AI Powered Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Powered Data Analysis Dashboard")

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Dashboard")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Upload Data",
        "Cleaning",
        "Visualization",
        "SQL Analysis",
        "Machine Learning",
        "Report"
    ]
)

# ---------------- SESSION STATE ----------------

if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.header("📊 Dashboard")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric(
        "Missing",
        int(df.isnull().sum().sum())
    )
    col4.metric(
        "Duplicates",
        int(df.duplicated().sum())
    )
    col5.metric(
        "Memory",
        f"{df.memory_usage().sum()/1024:.2f} KB"
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Search Records")

    search = st.text_input("Search")

    if search:

        filtered = df[
            df.astype(str)
            .apply(
                lambda x:
                x.str.contains(
                    search,
                    case=False,
                    na=False
                )
            )
            .any(axis=1)
        ]

        st.dataframe(filtered)

    st.subheader("AI Insights")

    st.info(
        f"""
Rows : {df.shape[0]}
Columns : {df.shape[1]}
Missing Values : {df.isnull().sum().sum()}
Duplicate Rows : {df.duplicated().sum()}
"""
    )

# ---------------- UPLOAD DATA ----------------

elif page == "Upload Data":

    st.header("📂 Upload Dataset")

    file = st.file_uploader(
        "Upload CSV, Excel or JSON",
        type=["csv", "xlsx", "json"]
    )

    if file:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)

        else:
            df = pd.read_json(file)

        st.session_state.df = df

        st.success("File Uploaded Successfully")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader("Dataset Preview")
        st.dataframe(df)

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Statistical Summary")
        st.write(
            df.describe(
                include="all"
            )
        )

# ---------------- CLEANING ----------------

elif page == "Cleaning":

    st.header("🧹 Data Cleaning")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first.")
        st.stop()

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    if st.button("Remove Null Values"):
        df = df.dropna()
        st.session_state.df = df
        st.success("Null values removed.")

    if st.button("Remove Duplicate Rows"):
        df = df.drop_duplicates()
        st.session_state.df = df
        st.success("Duplicates removed.")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:

        if st.button(
            "Fill Null Values With Mean"
        ):

            df[numeric_cols] = (
                df[numeric_cols]
                .fillna(
                    df[numeric_cols]
                    .mean()
                )
            )

            st.session_state.df = df
            st.success(
                "Null values filled."
            )

    st.subheader("Drop Columns")

    drop_cols = st.multiselect(
        "Select Columns",
        df.columns
    )

    if st.button(
        "Drop Selected Columns"
    ):

        df = df.drop(
            columns=drop_cols
        )

        st.session_state.df = df
        st.success(
            "Columns Removed"
        )

    st.subheader(
        "Rename Column"
    )

    old_col = st.selectbox(
        "Select Column",
        df.columns
    )

    new_col = st.text_input(
        "New Column Name"
    )

    if st.button(
        "Rename Column"
    ):

        if new_col:

            df.rename(
                columns={
                    old_col:
                    new_col
                },
                inplace=True
            )

            st.session_state.df = df

            st.success(
                "Column Renamed"
            )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(df)
# ---------------- VISUALIZATION ----------------

elif page == "Visualization":

    st.header("📈 Visualization")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first.")
        st.stop()

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:

        st.subheader("Numeric Filter")

        num_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        minimum = float(
            df[num_col].min()
        )

        maximum = float(
            df[num_col].max()
        )

        values = st.slider(
            "Select Range",
            minimum,
            maximum,
            (minimum, maximum)
        )

        df = df[
            (df[num_col] >= values[0])
            &
            (df[num_col] <= values[1])
        ]

    chart = st.selectbox(
        "Choose Chart",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Pie Chart",
            "Histogram",
            "Box Plot",
            "Correlation Heatmap"
        ]
    )

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) == 0:
        st.warning(
            "No numeric columns found."
        )
        st.stop()

    if chart == "Bar Chart":

        x = st.selectbox(
            "X Axis",
            df.columns
        )

        y = st.selectbox(
            "Y Axis",
            numeric_cols
        )

        fig = px.bar(
            df,
            x=x,
            y=y
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Line Chart":

        x = st.selectbox(
            "X Axis",
            df.columns
        )

        y = st.selectbox(
            "Y Axis",
            numeric_cols
        )

        fig = px.line(
            df,
            x=x,
            y=y
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Scatter Plot":

        x = st.selectbox(
            "X Axis",
            numeric_cols
        )

        y = st.selectbox(
            "Y Axis",
            numeric_cols
        )

        fig = px.scatter(
            df,
            x=x,
            y=y
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Pie Chart":

        names = st.selectbox(
            "Category",
            df.columns
        )

        values_col = st.selectbox(
            "Values",
            numeric_cols
        )

        fig = px.pie(
            df,
            names=names,
            values=values_col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Histogram":

        col = st.selectbox(
            "Column",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Box Plot":

        col = st.selectbox(
            "Column",
            numeric_cols
        )

        fig = px.box(
            df,
            y=col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart == "Correlation Heatmap":

        corr = df[
            numeric_cols
        ].corr()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

# ---------------- SQL ANALYSIS ----------------

elif page == "SQL Analysis":

    st.header("🗄 SQL Analysis")

    df = st.session_state.df

    if df is None:
        st.warning(
            "Please upload dataset first."
        )
        st.stop()

    conn = sqlite3.connect(
        ":memory:"
    )

    df.to_sql(
        "data",
        conn,
        index=False,
        if_exists="replace"
    )

    st.subheader("Example Queries")

    st.code(
        "SELECT * FROM data LIMIT 5"
    )

    st.code(
        "SELECT COUNT(*) FROM data"
    )

    st.code(
        "SELECT * FROM data WHERE column_name > 100"
    )

    query = st.text_area(
        "SQL Query",
        "SELECT * FROM data LIMIT 5"
    )

    if st.button("Run Query"):

        try:

            result = pd.read_sql(
                query,
                conn
            )

            st.dataframe(result)

        except Exception as e:
            st.error(e)

# ---------------- MACHINE LEARNING ----------------

elif page == "Machine Learning":

    st.header("🤖 Machine Learning")

    df = st.session_state.df

    if df is None:
        st.warning(
            "Please upload dataset first."
        )
        st.stop()

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) < 2:

        st.warning(
            "Need at least 2 numeric columns."
        )
        st.stop()

    target = st.selectbox(
        "Target Column",
        numeric_cols
    )

    features = st.multiselect(
        "Feature Columns",
        [
            c
            for c in numeric_cols
            if c != target
        ]
    )

    if st.button(
        "Train Model"
    ):

        if len(features) == 0:

            st.error(
                "Please select feature columns."
            )

        else:

            X = df[
                features
            ].fillna(0)

            y = df[
                target
            ].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            score = model.score(
                X_test,
                y_test
            )

            st.success(
                f"Model Accuracy (R²): {score:.2f}"
            )

            prediction = model.predict(
                X_test
            )

            result = pd.DataFrame(
                {
                    "Actual":
                    y_test.values,
                    "Predicted":
                    prediction
                }
            )

            st.subheader(
                "Predictions"
            )

            st.dataframe(
                result.head(10)
            )

# ---------------- REPORT ----------------

elif page == "Report":

    st.header("📄 Reports")

    df = st.session_state.df

    if df is None:
        st.warning(
            "Please upload dataset first."
        )
        st.stop()

    if st.button(
        "Generate PDF Report"
    ):

        buffer = io.BytesIO()

        c = canvas.Canvas(
            buffer
        )

        c.drawString(
            100,
            800,
            "Data Analysis Report"
        )

        c.drawString(
            100,
            780,
            f"Rows : {df.shape[0]}"
        )

        c.drawString(
            100,
            760,
            f"Columns : {df.shape[1]}"
        )

        c.drawString(
            100,
            740,
            f"Missing Values : {df.isnull().sum().sum()}"
        )

        c.drawString(
            100,
            720,
            f"Duplicate Rows : {df.duplicated().sum()}"
        )

        c.save()

        pdf = buffer.getvalue()

        st.download_button(
            "Download PDF",
            pdf,
            "report.pdf",
            "application/pdf"
        )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "Download CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )

    excel = io.BytesIO()

    with pd.ExcelWriter(
        excel,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False
        )

    st.download_button(
        "Download Excel",
        excel.getvalue(),
        "cleaned_data.xlsx"
    )