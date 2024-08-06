import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import altair as alt

# Disable the warning
#st.set_option('deprecation.showPyplotGlobalUse', False)

categorical_cols = ['car_model', 'transmission', 'fuelType', 'brand', 'customer_gender', 'store_location', 'paymentmethod', 'customer_marital_status', 'marketingcampaign']
label_encoders = {}

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_excel("cars_dataset_and_dummy_wo_year_v2.xlsx")

# Data preprocessing
def preprocess_data(df):
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Separate features and target variable
    X = df.drop(columns=['price'])
    y = df['price']
    # Perform label encoding for categorical columns
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col + '_encoded'] = label_encoders[col].fit_transform(df[col])
    # Drop the original categorical columns
    X.drop(categorical_cols, axis=1, inplace=True)
    return X, y

# Train the model
def train_model(X_train, y_train):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor

# Main function
def main():
    # Page title with car symbol
    st.title('🚗 Car Price Prediction App')

    # Load data
    cars_df = load_data()

    try:
        # Preprocess data
        X, y = preprocess_data(cars_df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # User input for prediction
        st.sidebar.subheader('Enter Car Details')
        features = {}
        for col in cars_df.columns:
            if col != 'price':
                if col in categorical_cols:
                    features[col] = st.sidebar.selectbox(f'Select {col}', cars_df[col].unique())
                else:
                    features[col] = st.sidebar.text_input(f'Enter {col}')

        # Prediction
        if st.sidebar.button('Predict Price'):
            input_data = pd.DataFrame([features])
            # Perform label encoding for categorical columns
            for col in categorical_cols:
                input_data[col + '_encoded'] = label_encoders[col].transform([features[col]])
            input_data.drop(categorical_cols, axis=1, inplace=True)
            prediction = model.predict(input_data)
            st.subheader('Predicted Price:')
            st.write(f'The predicted price for the selected car is <span style="color:green; font-size:20px;"><b>${prediction[0]:,.2f}</b></span>', unsafe_allow_html=True)
    except ValueError as ve:
        st.error(str(ve))

    # EDA section
    if cars_df is not None and not cars_df.empty:
        st.sidebar.subheader('Exploratory Data Analysis')
        st.sidebar.write('This section provides insights into the dataset.')

        if st.sidebar.checkbox('Show Pairplot'):
            st.subheader('Pairplot of Numerical Columns')
            numerical_cols = cars_df.select_dtypes(include=['float64', 'int64']).columns
            selected_cols = st.sidebar.multiselect('Select columns for pairplot:', options=numerical_cols)

            if len(selected_cols) > 1:
                hue_col = st.sidebar.selectbox('Select a categorical column for hue:', options=categorical_cols)
                sns.pairplot(cars_df, vars=selected_cols, hue=hue_col)
                st.pyplot()
            else:
                st.warning('Please select at least two numerical columns for pairplot.')

        # Clustering Plot
        if st.sidebar.checkbox('Show Clustering Plot'):
            st.subheader('Clustering Plot')

            # Aggregate data at the level of unique car models
            cars_agg_df = cars_df.groupby('car_model').agg({
                'mileage': 'mean',
                'tax': 'mean',
                'miles_per_gallon': 'mean',
                'engineSize': 'mean',
                'transmission': 'first',
                'fuelType': 'first'
            }).reset_index()

            # Apply StandardScaler to numerical features
            scaler = StandardScaler()
            numerical_cols = ['mileage', 'tax', 'miles_per_gallon', 'engineSize']
            X_scaled = scaler.fit_transform(cars_agg_df[numerical_cols])

            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Fit K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(X_pca)

            # Add cluster labels to the dataset
            cars_agg_df['cluster'] = kmeans.labels_

            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'car_model': cars_agg_df['car_model'],
                'cluster': cars_agg_df['cluster'],
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1]
            })

            # Create interactive scatter plot with labels
            scatter = alt.Chart(plot_df).mark_circle().encode(
                x=alt.X('PC1', axis=alt.Axis(title='Principal Component 1')),
                y=alt.Y('PC2', axis=alt.Axis(title='Principal Component 2')),
                color=alt.Color('cluster:N', title='Cluster'),
                tooltip='car_model'
            ).properties(
                width=800,
                height=500
            ).interactive()

            st.altair_chart(scatter)

        # Display box plot
        if st.sidebar.checkbox('Show Box Plot'):
            st.subheader('Box Plot')
            numeric_cols = cars_df.select_dtypes(include=['float64', 'int64']).columns
            x_col = st.selectbox('Select X-axis:', options=categorical_cols)
            y_col = st.selectbox('Select Y-axis:', options=numeric_cols)
            fig = plt.figure()
            sns.boxplot(data=cars_df, x=x_col, y=y_col)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            st.pyplot(fig)

        if st.sidebar.checkbox('Show Correlation Heatmap'):
            st.subheader('Correlation Heatmap with Relevant Columns')
            # Filter numeric columns
            numeric_cols = cars_df.select_dtypes(include=['float64', 'int64']).columns
            corr_df = cars_df[numeric_cols].corr()
            fig = plt.figure()
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap')
            st.pyplot(fig)

        if st.sidebar.checkbox('Show Histograms of Numerical Columns'):
            st.subheader('Histograms of Numerical Columns')
            numeric_cols = cars_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                st.write(f"### {col}")
                fig = plt.figure()
                sns.histplot(cars_df[col], bins=20, kde=True)
                plt.xlabel(col)
                plt.ylabel('Frequency')
                st.pyplot(fig)

        if st.sidebar.checkbox('Show Scatter Plot'):
            st.subheader('Scatter Plot')
            numeric_cols = cars_df.select_dtypes(include=['float64', 'int64']).columns
            x_col = st.selectbox('Select X-axis:', options=numeric_cols)
            y_col = st.selectbox('Select Y-axis:', options=numeric_cols)
            fig = plt.figure()
            sns.scatterplot(data=cars_df, x=x_col, y=y_col)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            st.pyplot(fig)

        if st.sidebar.checkbox('Show Bar Chart'):
            st.subheader('Bar Chart')
            bar_col = st.selectbox('Select column for bar chart:', options=categorical_cols)
            fig = plt.figure()
            sns.countplot(data=cars_df, x=bar_col)
            plt.xticks(rotation=45)
            plt.xlabel(bar_col)
            plt.ylabel('Count')
            st.pyplot(fig)

        # Plot feature importance
        if st.sidebar.checkbox('Show Feature Importance'):
            st.subheader('Feature Importance')
            feature_importance = model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            st.pyplot(fig)

    # Display dataset
    st.subheader('Dataset')
    st.write(cars_df)

if __name__ == '__main__':
    main()
