import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

st.set_page_config(page_title="Wine Dashboard", layout="wide")
st.sidebar.title("Wine Input Options")

@st.cache_data
def load_data():
    data = pd.read_csv("wine-quality.csv")
    data["quality_label"] = data["quality"].apply(lambda x: "Good" if x >= 7 else "Bad")
    data["alcohol_level"] = pd.cut(data["alcohol"], bins=[0, 9, 11, 13, 15],
                                   labels=["Low", "Medium", "High", "Very High"])
    return data

df = load_data()

st.title("Wine Quality Prediction")

alcohol_choice = st.sidebar.selectbox("Choose Alcohol Level", 
    ["Low (5-9)", "Medium (9-11)", "High (11-13)", "Very High (13-15)"])

alcohol_values = {
    "Low (5-9)": 7.0,
    "Medium (9-11)": 10.0,
    "High (11-13)": 12.0,
    "Very High (13-15)": 14.0
}

user_input = {"alcohol": alcohol_values[alcohol_choice]}

for col in df.columns[:-3]:  # skip last 3 columns
    if col != "alcohol":
        user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([user_input])

X = df.drop(["quality", "quality_label", "alcohol_level"], axis=1)
y = (df["quality"] >= 7).astype(int)

model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(input_df[X.columns])[0]
probability = model.predict_proba(input_df[X.columns])[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.success(f"Wine is predicted to be GOOD ({probability * 100:.1f}% confidence)")
else:
    st.error(f"Wine is predicted to be BAD/AVERAGE ({(1 - probability) * 100:.1f}% confidence)")

tab1, tab2, tab3 ,tab4,tab5,tab6,tab7,tab8,tab9= st.tabs(["Wine Quality", "Average Alcohol by Quality", "Wine Quality Breakdown","Good vs Bad Wines","Features Correlation","Feature Distributions","Best Wine Samples","Worst Wine Samples","Wine features"])
with tab1:
    st.subheader("Wine Quality Count (Bar Chart)")
    quality_counts = df['quality'].value_counts().sort_index().reset_index()
    quality_counts.columns = ['Quality', 'Count']

    sunset_colors = ["#ff9a8b52", "#010000D5", '#ff99ac', '#ffb347', '#fcd5ce', '#e29578', '#f08080']
    
    fig1 = px.bar(quality_counts, x='Quality', y='Count',
                  title="Wine Quality Distribution",
                  color='Quality',
                  color_discrete_sequence=sunset_colors)
    st.plotly_chart(fig1, use_container_width=True)



with tab2:
    st.subheader("Average Alcohol by Quality")
    line_data = df.groupby("quality")["alcohol"].mean().reset_index()
    
    fig2 = px.line(line_data, x="quality", y="alcohol", markers=True,
                   title="Alcohol Content vs Wine Quality",
                   line_shape='linear', 
                   color_discrete_sequence=['#542094']) 
    st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("Sunburst Chart: Quality → Alcohol Level")
    
    fig3 = px.sunburst(df, path=["quality_label", "alcohol_level"],
                       title="Wine Quality and Alcohol Level Breakdown",
                       color="quality_label",
                       color_discrete_map={"Good": "#FE60A2DE", "Bad": "#BA2968EA"})  
    st.plotly_chart(fig3, use_container_width=True)
with tab4:
    st.subheader("Quality Label Distribution (Pie Chart)")
    pie_colors = ["#1C2606","#9fc24f"]  
    
    pie = px.pie(df, names="quality_label", title="Proportion of Good vs Bad Wines",
                 hole=0.6, color_discrete_sequence=pie_colors)
    st.plotly_chart(pie, use_container_width=True)
with tab5:
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True) 
    fig_heatmap = px.imshow(
        corr, 
        text_auto=True, 
        color_continuous_scale='Purples',
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)    
with tab6:
    st.subheader("Feature Distributions")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    selected_feature = st.selectbox("Select a Feature", num_cols)

    fig_dist = px.histogram(df, x=selected_feature, nbins=30,
                            title=f"Distribution of {selected_feature}",
                            color_discrete_sequence=["#079090"])  
    st.plotly_chart(fig_dist, use_container_width=True)

with tab7:
    st.subheader("Top 10 Best Quality Wines")
    best_wines = df[df["quality"] == df["quality"].max()].head(10)
    st.dataframe(best_wines.style.background_gradient(cmap='Greens'))

with tab8:
    st.subheader("Bottom 10 Lowest Quality Wines")
    worst_wines = df[df["quality"] == df["quality"].min()].head(10)
    st.dataframe(worst_wines.style.background_gradient(cmap='Blues'))
with tab9:
    st.subheader("Wine Feature Pair Plot")

    st.markdown(
        "This pair plot shows relationships between selected features like fixed acidity, "
        "volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, "
        "total sulfur dioxide, density, pH, sulphates, alcohol, and quality."
    )

    pairplot_features = df[[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    ]]

    
    pairplot_features["quality"] = pairplot_features["quality"].astype(str)

    sns_plot = sns.pairplot(pairplot_features, hue="quality", palette="Set2")
    st.pyplot(sns_plot.fig)  


with st.expander("See Raw Data"):
    st.dataframe(df.head(50))
