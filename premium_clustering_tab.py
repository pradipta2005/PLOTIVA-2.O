import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler



def render_clustering_tab(df: pd.DataFrame):
    """
    Render the Clustering & Segmentation Module
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Clustering & Segmentation</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Uncover hidden patterns, group similar entities, and reduce dimensionality.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">Unsupervised Learning</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    tab_cluster, tab_dim = st.tabs(["🧩 Clustering Studio", "📉 Dimensionality Reduction"])

    # -------------------------------------------------------------------------
    # CLUSTERING STUDIO
    # -------------------------------------------------------------------------
    with tab_cluster:
        c1, c2 = st.columns([1, 2], gap="large")
        
        with c1:
            st.markdown("### ⚙️ Configuration")
            algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = st.multiselect("Select Features", numeric_cols, default=numeric_cols[:3] if len(numeric_cols)>3 else numeric_cols)
            
            params = {}
            if algo == "K-Means":
                params['n_clusters'] = st.slider("Number of Clusters (K)", 2, 10, 3)
                st.caption("Standard clustering algorithm reducing variance within clusters.")
                
                # OPTIMAL K FINDER
                with st.expander("🔎 Determine Optimal K (Elbow Method)", expanded=False):
                    st.markdown("Run a scan to find the best number of clusters.")
                    max_k = st.slider("Max K to test", 5, 15, 10)
                    
                    if st.button("Run K-Scan", key="run_k_scan"):
                         if not features:
                             st.error("Select features first.")
                         else:
                             with st.spinner("Analyzing cluster quality across K values..."):
                                 X_scan = df[features].dropna()
                                 scaler_scan = StandardScaler()
                                 X_scan_scaled = scaler_scan.fit_transform(X_scan)
                                 
                                 inertias = []
                                 sil_scores = []
                                 k_range = range(2, max_k + 1)
                                 
                                 for k in k_range:
                                     km = KMeans(n_clusters=k, random_state=42, n_init=3) # Faster init for scan
                                     km.fit(X_scan_scaled)
                                     inertias.append(km.inertia_)
                                     sil_scores.append(silhouette_score(X_scan_scaled, km.labels_))
                                 
                                 # Dual Axis Plot
                                 fig_k = go.Figure()
                                 fig_k.add_trace(go.Scatter(x=list(k_range), y=inertias, name="Inertia (Elbow)", line=dict(color="#3B82F6", width=3)))
                                 fig_k.add_trace(go.Scatter(x=list(k_range), y=sil_scores, name="Silhouette Score", yaxis="y2", line=dict(color="#10B981", width=3, dash='dot')))
                                 
                                 fig_k.update_layout(
                                     title="Optimal K Analysis",
                                     xaxis=dict(title="Number of Clusters (K)"),
                                     yaxis=dict(title="Inertia (Lower is Better)", showgrid=False),
                                     yaxis2=dict(title="Silhouette (Higher is Better)", overlaying="y", side="right", showgrid=False),
                                     template="plotly_dark",
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     legend=dict(x=0.5, y=1.1, orientation="h")
                                 )
                                 st.plotly_chart(fig_k, use_container_width=True)
                                 
                                 # Recommendation
                                 best_sil_k = k_range[np.argmax(sil_scores)]
                                 st.success(f"💡 Suggestion: K={best_sil_k} (highest silhouette score)")

            elif algo == "Hierarchical":
                params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
                params['linkage'] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
                st.caption("Builds nested clusters by merging or splitting them successively.")
            elif algo == "DBSCAN":
                params['eps'] = st.slider("Epsilon (Distance)", 0.1, 5.0, 0.5)
                params['min_samples'] = st.slider("Min Samples", 2, 20, 5)
                st.caption("Density-based clustering. Finds core samples and expands clusters.")
            elif algo == "Gaussian Mixture":
                params['n_components'] = st.slider("Number of Components", 2, 10, 3)
                st.caption("Probabilistic model assuming all data points are generated from a mixture of Gaussians.")
                
            run_btn = st.button("✨ Run Clustering", type="primary", use_container_width=True)

        with c2:
            if run_btn and features:
                with st.spinner(f"Running {algo}..."):
                    try:
                        X = df[features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        model = None
                        labels = None
                        
                        if algo == "K-Means":
                            model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
                            labels = model.fit_predict(X_scaled)
                        elif algo == "Hierarchical":
                            model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
                            labels = model.fit_predict(X_scaled)
                        elif algo == "DBSCAN":
                            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
                            labels = model.fit_predict(X_scaled)
                        elif algo == "Gaussian Mixture":
                            model = GaussianMixture(n_components=params['n_components'], random_state=42)
                            labels = model.fit_predict(X_scaled)
                            
                        # Results
                        df_res = X.copy()
                        df_res['Cluster'] = labels.astype(str)
                        
                        # Metrics
                        unique_labels = set(labels)
                        if -1 in unique_labels: unique_labels.remove(-1)
                        
                        if len(unique_labels) > 1:
                            sil_score = silhouette_score(X_scaled, labels)
                            ch_score = calinski_harabasz_score(X_scaled, labels)
                            
                            m1, m2 = st.columns(2)
                            m1.metric("Silhouette Score", f"{sil_score:.3f}", help="Range -1 to 1. Higher is better.")
                            m2.metric("Calinski-Harabasz", f"{ch_score:.1f}", help="Higher is better (dense and well separated)")
                        else:
                            st.warning("Only one cluster found (or only noise). Adjust parameters.")

                        # Visualization (PCA 2D projection if dims > 2)
                        st.subheader("Cluster Visualization")
                        
                        if X.shape[1] > 2:
                            pca = PCA(n_components=2)
                            coords = pca.fit_transform(X_scaled)
                            df_res['PC1'] = coords[:, 0]
                            df_res['PC2'] = coords[:, 1]
                            x_col, y_col = 'PC1', 'PC2'
                            title = "Clusters (PCA Projected)"
                        else:
                            x_col, y_col = features[0], features[1] if len(features) > 1 else features[0]
                            title = "Clusters"
                            
                        fig = px.scatter(
                            df_res, x=x_col, y=y_col, color='Cluster',
                            title=title, template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            hover_data=features
                        )
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cluster Profiling
                        st.subheader("Cluster Profiling")
                        profile = df_res.groupby('Cluster')[features].mean().reset_index()
                        st.dataframe(profile, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Clustering failed: {str(e)}")
            elif run_btn and not features:
                st.error("Please select features.")
            else:
                st.info("Select features and click Run to start clustering.")

    # -------------------------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------
    with tab_dim:
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.markdown("### ⚙️ Configuration")
            dim_method = st.selectbox("Method", ["PCA", "t-SNE", "UMAP"])
            
            dim_features = st.multiselect("Input Features", numeric_cols, default=numeric_cols, key="dim_feats")
            n_comps = st.slider("Components", 2, 3, 2)
            
            extra_params = {}
            if dim_method == "t-SNE":
                extra_params['perplexity'] = st.slider("Perplexity", 5, 50, 30)
            elif dim_method == "UMAP":
                extra_params['neighbors'] = st.slider("n_neighbors", 5, 50, 15)
                extra_params['min_dist'] = st.slider("min_dist", 0.0, 1.0, 0.1)
                
            color_by = st.selectbox("Color By (Optional)", ["None"] + df.columns.tolist())
            
            run_dim = st.button("📉 Project Data", type="primary", use_container_width=True)

        with c2:
            if run_dim and dim_features:
                with st.spinner(f"Running {dim_method}..."):
                    try:
                        X = df[dim_features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        proj = None
                        cols = []
                        
                        if dim_method == "PCA":
                            model = PCA(n_components=n_comps)
                            proj = model.fit_transform(X_scaled)
                            cols = [f"PC{i+1}" for i in range(n_comps)]
                            
                            # Scree Plot
                            expl_var = model.explained_variance_ratio_
                            st.caption(f"Explained Variance: {sum(expl_var):.1%}")
                            
                        elif dim_method == "t-SNE":
                            model = TSNE(n_components=n_comps, perplexity=extra_params['perplexity'], random_state=42)
                            proj = model.fit_transform(X_scaled)
                            cols = [f"tSNE{i+1}" for i in range(n_comps)]
                            
                        elif dim_method == "UMAP":
                            model = umap.UMAP(n_components=n_comps, n_neighbors=extra_params['neighbors'], min_dist=extra_params['min_dist'], random_state=42)
                            proj = model.fit_transform(X_scaled)
                            cols = [f"UMAP{i+1}" for i in range(n_comps)]
                            
                        df_proj = pd.DataFrame(proj, columns=cols)
                        
                        # Add color column if selected
                        color_col = None
                        if color_by != "None":
                            df_proj[color_by] = df.loc[X.index, color_by].values
                            color_col = color_by
                            
                        if n_comps == 3:
                            fig = px.scatter_3d(df_proj, x=cols[0], y=cols[1], z=cols[2], color=color_col, title=f"{dim_method} Projection (3D)", template="plotly_dark")
                        else:
                            fig = px.scatter(df_proj, x=cols[0], y=cols[1], color=color_col, title=f"{dim_method} Projection (2D)", template="plotly_dark")
                            
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Projection failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
