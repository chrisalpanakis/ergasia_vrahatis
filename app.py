import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO


st.set_page_config(
    page_title="Molecular Biology Data Analysis",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">🧬 Molecular Biology Data Analysis Platform</h1>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🔍 Exploratory Analysis", "🤖 Machine Learning", "📈 Visualizations", "👥 Team Info"])


def generate_sample_data():
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(1, 101)]
    samples = [f"Sample_{i}" for i in range(1, 21)]
    

    data = np.random.lognormal(0, 1, (100, 20))
    df = pd.DataFrame(data, index=genes, columns=samples)
    

    metadata = pd.DataFrame({
        'Sample': samples,
        'Condition': ['Control'] * 10 + ['Treatment'] * 10,
        'Batch': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2,
        'Age': np.random.randint(20, 60, 20),
        'Gender': np.random.choice(['M', 'F'], 20)
    })
    
    return df, metadata


with tab1:
    st.markdown('<h2 class="tab-header">Data Upload & Processing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with gene expression data",
            type=['csv'],
            help="Upload a CSV file where rows are genes and columns are samples"
        )
        
        use_sample = st.button("Use Sample Data", type="secondary")
        
    with col2:
        st.subheader("Data Parameters")
        log_transform = st.checkbox("Apply log2 transformation", value=True)
        normalize = st.checkbox("Normalize data", value=False)
        filter_genes = st.slider("Filter genes with low expression", 0, 10, 1)
    

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, index_col=0)
            metadata = None
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            data, metadata = generate_sample_data()
    elif use_sample or 'data' not in st.session_state:
        data, metadata = generate_sample_data()
        st.info("Using sample molecular biology data")
    else:
        data = st.session_state.get('data')
        metadata = st.session_state.get('metadata')
    

    if data is not None:
        processed_data = data.copy()
        

        if log_transform:
            processed_data = np.log2(processed_data + 1)
        
        if normalize:
            processed_data = processed_data.div(processed_data.sum(axis=0), axis=1) * 1e6
        
        if filter_genes > 0:
            gene_means = processed_data.mean(axis=1)
            processed_data = processed_data[gene_means > filter_genes]
        
        st.session_state['data'] = processed_data
        st.session_state['metadata'] = metadata
        st.session_state['original_data'] = data
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Genes", processed_data.shape[0])
        col2.metric("Samples", processed_data.shape[1])
        col3.metric("Total Data Points", processed_data.shape[0] * processed_data.shape[1])
        col4.metric("Missing Values", processed_data.isnull().sum().sum())
        st.markdown('</div>', unsafe_allow_html=True)
        

        st.subheader("Data Preview")
        st.dataframe(processed_data.head(10), use_container_width=True)

with tab2:
    if 'data' in st.session_state:
        st.markdown('<h2 class="tab-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        data = st.session_state['data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Distribution")
            fig_hist = px.histogram(
                data.values.flatten(), 
                nbins=50,
                title="Expression Value Distribution",
                labels={'value': 'Expression Level', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Sample Correlation Heatmap")
            corr_matrix = data.T.corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Sample-Sample Correlation",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        

        st.subheader("Gene Expression Statistics")
        gene_stats = pd.DataFrame({
            'Mean': data.mean(axis=1),
            'Std': data.std(axis=1),
            'CV': data.std(axis=1) / data.mean(axis=1),
            'Max': data.max(axis=1),
            'Min': data.min(axis=1)
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(gene_stats.head(10), use_container_width=True)
        
        with col2:
            
            top_genes = gene_stats.nlargest(10, 'CV')
            fig_var = px.bar(
                x=top_genes.index,
                y=top_genes['CV'],
                title="Top 10 Most Variable Genes",
                labels={'x': 'Genes', 'y': 'Coefficient of Variation'}
            )
            fig_var.update_xaxes(tickangle=45)
            st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.warning("Please upload data in the Data Upload tab first.")

with tab3:
    if 'data' in st.session_state:
        st.markdown('<h2 class="tab-header">Machine Learning Analysis</h2>', unsafe_allow_html=True)
        
        data = st.session_state['data']
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("ML Parameters")
            

            st.write("**PCA Analysis**")
            n_components = st.slider("Number of PCA components", 2, min(10, data.shape[1]-1), 3)
            

            st.write("**K-Means Clustering**")
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            cluster_samples = st.checkbox("Cluster samples", value=True)
            cluster_genes = st.checkbox("Cluster genes", value=False)
            
            run_analysis = st.button("Run Analysis", type="primary")
        
        with col2:
            if run_analysis or 'pca_results' in st.session_state:
   
                X_samples = data.T 
                X_genes = data    

                scaler_samples = StandardScaler()
                X_samples_scaled = scaler_samples.fit_transform(X_samples)
                
                scaler_genes = StandardScaler()
                X_genes_scaled = scaler_genes.fit_transform(X_genes.T).T
                
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_samples_scaled)
                
                st.session_state['pca_results'] = pca_result
                st.session_state['pca_explained_var'] = pca.explained_variance_ratio_
                
                if n_components >= 2:
                    fig_pca = px.scatter(
                        x=pca_result[:, 0], 
                        y=pca_result[:, 1],
                        title=f"PCA Analysis (PC1 vs PC2)",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                        hover_name=data.columns
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                fig_var = px.bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=pca.explained_variance_ratio_,
                    title="PCA Explained Variance Ratio"
                )
                st.plotly_chart(fig_var, use_container_width=True)
                
                if cluster_samples:
                    st.subheader("Sample Clustering Results")
                    kmeans_samples = KMeans(n_clusters=n_clusters, random_state=42)
                    sample_clusters = kmeans_samples.fit_predict(X_samples_scaled)
                    
                    fig_pca_cluster = px.scatter(
                        x=pca_result[:, 0], 
                        y=pca_result[:, 1],
                        color=sample_clusters.astype(str),
                        title=f"PCA with K-Means Clusters (k={n_clusters})",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                        hover_name=data.columns
                    )
                    st.plotly_chart(fig_pca_cluster, use_container_width=True)
                
                if cluster_genes:
                    st.subheader("Gene Clustering Results")
                    kmeans_genes = KMeans(n_clusters=n_clusters, random_state=42)
                    gene_clusters = kmeans_genes.fit_predict(X_genes_scaled)
                    
                    gene_cluster_df = pd.DataFrame({
                        'Gene': data.index,
                        'Cluster': gene_clusters,
                        'Mean_Expression': data.mean(axis=1)
                    })
                    
                    for cluster in range(n_clusters):
                        cluster_genes_list = gene_cluster_df[gene_cluster_df['Cluster'] == cluster].nlargest(5, 'Mean_Expression')
                        st.write(f"**Cluster {cluster} - Top Genes:**")
                        st.write(", ".join(cluster_genes_list['Gene'].tolist()))
    else:
        st.warning("Please upload data in the Data Upload tab first.")

with tab4:
    if 'data' in st.session_state:
        st.markdown('<h2 class="tab-header">Advanced Visualizations</h2>', unsafe_allow_html=True)
        
        data = st.session_state['data']
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Visualization Options")
            selected_genes = st.multiselect(
                "Select genes for analysis",
                options=data.index.tolist(),
                default=data.index[:5].tolist(),
                max_selections=10
            )
            
            plot_type = st.selectbox(
                "Select plot type",
                ["Box Plot", "Violin Plot", "Expression Heatmap", "Gene Expression Profile"]
            )
        
        with col2:
            if selected_genes:
                subset_data = data.loc[selected_genes]
                
                if plot_type == "Box Plot":
                    melted_data = subset_data.T.melt(var_name='Gene', value_name='Expression')
                    fig = px.box(melted_data, x='Gene', y='Expression', title="Gene Expression Box Plot")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Violin Plot":
                    melted_data = subset_data.T.melt(var_name='Gene', value_name='Expression')
                    fig = px.violin(melted_data, x='Gene', y='Expression', title="Gene Expression Violin Plot")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Expression Heatmap":
                    fig = px.imshow(
                        subset_data,
                        title="Gene Expression Heatmap",
                        labels={'x': 'Samples', 'y': 'Genes', 'color': 'Expression'},
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Gene Expression Profile":
                    fig = go.Figure()
                    for gene in selected_genes:
                        fig.add_trace(go.Scatter(
                            x=data.columns,
                            y=data.loc[gene],
                            mode='lines+markers',
                            name=gene,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Gene Expression Profiles Across Samples",
                        xaxis_title="Samples",
                        yaxis_title="Expression Level",
                        hovermode='x unified'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select genes to visualize")
        
        st.subheader("Summary Statistics")
        if selected_genes:
            summary_stats = data.loc[selected_genes].describe().T
            st.dataframe(summary_stats, use_container_width=True)
    else:
        st.warning("Please upload data in the Data Upload tab first.")

with tab5:
    st.markdown('<h2 class="tab-header">Team Information</h2>', unsafe_allow_html=True)
    
    
    st.markdown("""
    ### 🧬 Molecular Biology Analysis Platform Development Team
    
    **Project:** Interactive Molecular Biology Data Analysis Application
    
    **Semester:** Spring 2024-2025
    
    ---
  """)
    
    st.subheader("Application Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Lines of Code", "~300")
    col2.metric("Features", "15+")
    col3.metric("Visualization Types", "8")
    col4.metric("ML Algorithms", "2")

st.sidebar.markdown("## 🧬 About This App")
st.sidebar.info("""
This interactive application provides comprehensive analysis tools for molecular biology data including:

- Data upload and preprocessing
- Exploratory data analysis  
- PCA and clustering analysis
- Interactive visualizations
- Statistical summaries

Upload your gene expression data or use the sample dataset to get started!
""")

st.sidebar.markdown("## 📊 Quick Stats")
if 'data' in st.session_state:
    data = st.session_state['data']
    st.sidebar.metric("Genes Loaded", data.shape[0])
    st.sidebar.metric("Samples", data.shape[1])
    st.sidebar.metric("Data Points", f"{data.shape[0] * data.shape[1]:,}")
else:
    st.sidebar.info("No data loaded yet")