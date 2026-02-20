"""
Premium Analytics Module
Advanced Statistical Analysis and Machine Learning
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, calinski_harabasz_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score,
                             r2_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import warnings

warnings.filterwarnings('ignore')

class PremiumAnalytics:
    """Advanced analytics and statistical testing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'numeric_summary': {},
            'categorical_summary': {},
            'distribution_tests': {},
            'outlier_analysis': {}
        }
        
        # Numeric summary with advanced statistics
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                summary['numeric_summary'][col] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'iqr': data.quantile(0.75) - data.quantile(0.25),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'cv': data.std() / data.mean() if data.mean() != 0 else 0
                }
                
                # Normality test
                if len(data) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for performance
                    summary['distribution_tests'][col] = {
                        'shapiro_stat': shapiro_stat,
                        'shapiro_p': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                
                # Outlier detection
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                summary['outlier_analysis'][col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(data) * 100,
                    'outlier_values': outliers.tolist()[:10]  # Limit to first 10
                }
        
        # Categorical summary
        for col in categorical_cols:
            data = df[col].dropna()
            if len(data) > 0:
                value_counts = data.value_counts()
                summary['categorical_summary'][col] = {
                    'unique_count': data.nunique(),
                    'most_frequent': value_counts.index[0],
                    'most_frequent_count': value_counts.iloc[0],
                    'least_frequent': value_counts.index[-1],
                    'least_frequent_count': value_counts.iloc[-1],
                    'entropy': stats.entropy(value_counts.values),
                    'value_counts': value_counts.head(10).to_dict()
                }
        
        return summary
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation (rank-based)
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Kendall correlation
        kendall_corr = numeric_df.corr(method='kendall')
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'var1': pearson_corr.columns[i],
                        'var2': pearson_corr.columns[j],
                        'correlation': corr_val,
                        'strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                    })
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'kendall_correlation': kendall_corr,
            'strong_correlations': strong_correlations
        }
    
    def hypothesis_testing(self, df: pd.DataFrame, group_col: str, 
                          value_col: str, test_type: str = 'auto') -> Dict[str, Any]:
        """Perform hypothesis testing"""
        
        groups = df[group_col].unique()
        group_data = [df[df[group_col] == group][value_col].dropna() for group in groups]
        
        # Remove empty groups
        group_data = [data for data in group_data if len(data) > 0]
        groups = [group for i, group in enumerate(groups) if len(group_data[i]) > 0]
        
        if len(group_data) < 2:
            return {'error': 'Need at least 2 groups with data for hypothesis testing'}
        
        results = {}
        
        if test_type == 'auto':
            # Determine appropriate test based on data
            if len(group_data) == 2:
                # Two groups - t-test or Mann-Whitney U
                # Check normality
                normal_p_values = []
                for data in group_data:
                    if len(data) >= 8:
                        _, p_val = stats.shapiro(data[:5000])
                        normal_p_values.append(p_val)
                
                if all(p > 0.05 for p in normal_p_values):
                    # Both groups are normal - use t-test
                    stat, p_val = stats.ttest_ind(group_data[0], group_data[1])
                    results['test_used'] = 'Independent t-test'
                    results['assumption'] = 'Normal distribution'
                else:
                    # Non-normal - use Mann-Whitney U
                    stat, p_val = stats.mannwhitneyu(group_data[0], group_data[1])
                    results['test_used'] = 'Mann-Whitney U test'
                    results['assumption'] = 'Non-normal distribution'
            
            else:
                # Multiple groups - ANOVA or Kruskal-Wallis
                # Check normality for all groups
                normal_p_values = []
                for data in group_data:
                    if len(data) >= 8:
                        _, p_val = stats.shapiro(data[:5000])
                        normal_p_values.append(p_val)
                
                if all(p > 0.05 for p in normal_p_values):
                    # All groups are normal - use ANOVA
                    stat, p_val = stats.f_oneway(*group_data)
                    results['test_used'] = 'One-way ANOVA'
                    results['assumption'] = 'Normal distribution'
                else:
                    # Non-normal - use Kruskal-Wallis
                    stat, p_val = stats.kruskal(*group_data)
                    results['test_used'] = 'Kruskal-Wallis test'
                    results['assumption'] = 'Non-normal distribution'
        
        results.update({
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'groups': groups,
            'group_means': [data.mean() for data in group_data],
            'group_stds': [data.std() for data in group_data],
            'effect_size': self._calculate_effect_size(group_data)
        })
        
        return results
    
    def _calculate_effect_size(self, group_data: List[np.ndarray]) -> float:
        """Calculate Cohen's d for effect size"""
        if len(group_data) == 2:
            # Cohen's d for two groups
            mean1, mean2 = group_data[0].mean(), group_data[1].mean()
            std1, std2 = group_data[0].std(), group_data[1].std()
            n1, n2 = len(group_data[0]), len(group_data[1])
            
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            # Prevent division by zero
            if pooled_std == 0:
                return 0.0
            cohens_d = (mean1 - mean2) / pooled_std
            return abs(cohens_d)
        else:
            # Eta-squared for multiple groups
            all_data = np.concatenate(group_data)
            grand_mean = all_data.mean()
            
            ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in group_data)
            ss_total = sum((x - grand_mean)**2 for x in all_data)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            return eta_squared
    
    def advanced_clustering(self, df: pd.DataFrame, features: List[str],
                          algorithm: str = 'kmeans', n_clusters: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """Perform advanced clustering analysis"""
        
        # Prepare data
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for clustering'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return {'error': f'Unknown clustering algorithm: {algorithm}'}
        
        clusters = model.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        metrics = {}
        if n_clusters_found > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X_scaled, clusters)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, clusters)
            except Exception:
                # Clustering metrics calculation failed, continue without them
                pass
        
        # Cluster analysis
        cluster_analysis = {}
        for cluster_id in set(clusters):
            if cluster_id != -1:  # Exclude noise points in DBSCAN
                cluster_mask = clusters == cluster_id
                cluster_data = X[cluster_mask]
                
                cluster_analysis[f'Cluster_{cluster_id}'] = {
                    'size': cluster_mask.sum(),
                    'percentage': cluster_mask.sum() / len(clusters) * 100,
                    'means': cluster_data.mean().to_dict(),
                    'stds': cluster_data.std().to_dict()
                }
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters_found,
            'algorithm': algorithm,
            'features': features,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis,
            'scaled_data': X_scaled,
            'original_data': X
        }
    
    def dimensionality_reduction(self, df: pd.DataFrame, features: List[str],
                               method: str = 'pca', n_components: int = 2) -> Dict[str, Any]:
        """Perform dimensionality reduction"""
        
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for dimensionality reduction'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            
            # Calculate explained variance
            explained_variance = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Get component loadings
            components = pd.DataFrame(
                reducer.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=features
            )
            
            results = {
                'method': 'PCA',
                'reduced_data': X_reduced,
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'components': components,
                'feature_names': [f'PC{i+1}' for i in range(n_components)]
            }
        
        elif method == 'tsne':
            perplexity = min(30, len(X_scaled) - 1)
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
            X_reduced = reducer.fit_transform(X_scaled)
            
            results = {
                'method': 't-SNE',
                'reduced_data': X_reduced,
                'feature_names': [f'tSNE{i+1}' for i in range(n_components)]
            }
        
        else:
            return {'error': f'Unknown dimensionality reduction method: {method}'}
        
        return results
    
    def anomaly_detection(self, df: pd.DataFrame, features: List[str],
                         method: str = 'isolation_forest', **kwargs) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for anomaly detection'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            model = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
        
        elif method == 'statistical':
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(X_scaled, axis=0))
            threshold = kwargs.get('threshold', 3)
            anomaly_labels = np.where((z_scores > threshold).any(axis=1), -1, 1)
            anomaly_scores = np.max(z_scores, axis=1)
        
        else:
            return {'error': f'Unknown anomaly detection method: {method}'}
        
        # Analyze anomalies
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        normal_indices = np.where(anomaly_labels == 1)[0]
        
        anomaly_analysis = {
            'total_anomalies': len(anomaly_indices),
            'anomaly_percentage': len(anomaly_indices) / len(X) * 100,
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores[anomaly_indices].tolist(),
            'top_anomalies': X.iloc[anomaly_indices].head(10).to_dict('records')
        }
        
        return {
            'method': method,
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'analysis': anomaly_analysis,
            'features': features
        }
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Perform time series analysis"""
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        ts_data = df.set_index(date_col)[value_col].sort_index()
        
        # Remove missing values
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 10:
            return {'error': 'Need at least 10 data points for time series analysis'}
        
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'count': len(ts_data),
            'mean': ts_data.mean(),
            'std': ts_data.std(),
            'min': ts_data.min(),
            'max': ts_data.max(),
            'trend': 'increasing' if ts_data.iloc[-1] > ts_data.iloc[0] else 'decreasing'
        }
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(ts_data)
            results['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        except ImportError:
            results['stationarity'] = {'error': 'statsmodels not available'}
        
        # Seasonality detection
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            if len(ts_data) >= 12:  # Need enough data for decomposition
                # Try to infer period if not set
                period = 12 if len(ts_data) > 24 else max(2, len(ts_data)//4)
                
                try:
                    decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                    results['seasonality'] = {
                        'trend': decomposition.trend.dropna().tolist(),
                        'seasonal': decomposition.seasonal.dropna().tolist(),
                        'residual': decomposition.resid.dropna().tolist()
                    }
                except ValueError as e:
                     results['seasonality'] = {'error': f'Decomposition failed: {str(e)}'}
                except Exception as e:
                     results['seasonality'] = {'error': f'Decomposition error: {str(e)}'}
            else:
                 results['seasonality'] = {'error': 'Not enough data for seasonality analysis (need 12+ points)'}
        except ImportError:
            results['seasonality'] = {'error': 'statsmodels not available'}
        
        # Autocorrelation
        autocorr = [ts_data.autocorr(lag=i) for i in range(1, min(21, len(ts_data)//2))]
        results['autocorrelation'] = {
            'lags': list(range(1, len(autocorr) + 1)),
            'values': autocorr
        }
        
        return results
    
    def feature_importance_analysis(self, df: pd.DataFrame, target_col: str,
                                  feature_cols: List[str], method: str = 'random_forest') -> Dict[str, Any]:
        """Analyze feature importance"""
        
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        from sklearn.inspection import permutation_importance
        
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:
            return {'error': 'Need at least 10 samples for feature importance analysis'}
        
        # Determine if regression or classification
        is_classification = y.dtype == 'object' or y.nunique() < 10
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            perm_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
        except Exception:
            # Permutation importance calculation failed
            perm_df = None
        
        return {
            'method': method,
            'task_type': 'classification' if is_classification else 'regression',
            'feature_importance': feature_importance,
            'permutation_importance': perm_df,
            'model_score': model.score(X, y)
        }

    
    def train_model(self, df: pd.DataFrame, target_col: str, feature_cols: List[str],
                   model_type: str, task_type: str = 'regression',
                   test_size: float = 0.2, tune_hyperparams: bool = False) -> Dict[str, Any]:
        """Train and evaluate a machine learning model with scaling and advanced metrics"""
        
        # Prepare data
        try:
            # Handle missing values
            df = df.copy()
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns

            # Impute numeric
            if len(numeric_cols) > 0:
                numeric_imputer = df[numeric_cols].mean()
                df[numeric_cols] = df[numeric_cols].fillna(numeric_imputer)
            
            # Impute categorical
            if len(categorical_cols) > 0:
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

            # Process Features 
            # 1. Numeric
            X_numeric = df[numeric_cols]
            
            # 2. Categorical (One-Hot)
            X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
            
            # 3. Datetime (Convert to timestamp/ordinal)
            X_datetime = pd.DataFrame()
            datetime_cols = df[feature_cols].select_dtypes(include=['datetime', 'datetimetz']).columns
            for col in datetime_cols:
                # Convert to ordinal (numeric)
                X_datetime[col] = pd.to_numeric(df[col].map(pd.Timestamp.toordinal), errors='coerce')
                X_datetime[col] = X_datetime[col].fillna(X_datetime[col].mean())

            # Combine all features
            X = pd.concat([X_numeric, X_categorical, X_datetime], axis=1)

            if X.empty:
                 return {'error': 'No valid features found for training. Please check your input scope.'}

            y = df[target_col]
            
            # Detect mismatch between Task Type and Target Variable
            if task_type == 'regression':
                if not pd.api.types.is_numeric_dtype(y):
                    # Try to see if it can be converted (e.g. numeric strings)
                    try:
                        pd.to_numeric(y, errors='raise')
                    except:
                        return {'error': f"Target '{target_col}' is categorical (text). Please select 'Classification' as the task type."}
                        
                y = pd.to_numeric(y, errors='coerce')
                if y.isna().all():
                     return {'error': f"Target '{target_col}' contains no valid numeric data for Regression."}
                y = y.fillna(y.mean())
                
            elif task_type == 'classification':
                # Check if target is continuous
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                     return {'error': f"Target '{target_col}' appears to be continuous (numeric with high cardinality). Please select 'Regression' as the task type."}

                if y.dtype == 'object' or pd.api.types.is_numeric_dtype(y):
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                
            # DATA LEAKAGE PREVENTION
            # Check for features that are identical to target
            for col in X.columns:
                if X[col].equals(y):
                    return {'error': f"Data Leakage Detected: Feature '{col}' is identical to target '{target_col}'. Please remove it."}
            
            # Check for extremely high correlation (suspicious)
            corrs = X.corrwith(y).abs()
            high_corr_feats = corrs[corrs > 0.99].index.tolist()
            if high_corr_feats:
                 # We won't block it, but we should probably warn or at least be aware.
                 # For now, strict 1.0 equality is the blocker.
                 pass

            if len(X) < 10:
                return {'error': 'Need at least 10 samples for training'}
                
            # Split data (ACCURACY GUARANTEE: Stratified Split)
            stratify_indices = None
            if task_type == 'classification':
                # Check for Class Imbalance
                class_counts = y.value_counts(normalize=True)
                if class_counts.min() < 0.05:
                    # Minor class < 5%
                    # Warning handled in UI usually, but good to know
                    pass
                
                # Only stratify if we have enough samples per class
                if y.value_counts().min() > 1:
                    stratify_indices = y
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_indices)
            
            # Scale data (Important for accuracy in many models)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select model
            if task_type == 'regression':
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                    'Extra Trees': ExtraTreesRegressor(random_state=42),
                    'SVR': SVR(),
                    'Ridge': Ridge(),
                    'Lasso': Lasso(),
                    'Decision Tree': DecisionTreeRegressor(random_state=42)
                }
                if XGBOOST_AVAILABLE:
                    models['XGBoost'] = XGBRegressor(random_state=42)
            else:
                models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                    'Extra Trees': ExtraTreesClassifier(random_state=42),
                    'SVC': SVC(probability=True, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'KNN': KNeighborsClassifier()
                }
                if XGBOOST_AVAILABLE:
                    models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
            
            if model_type not in models:
                return {'error': f'Unknown model type: {model_type}'}
                
            model = models[model_type]
            
            # Hyperparameter tuning
            if tune_hyperparams:
                if 'Random Forest' in model_type:
                    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
                elif 'Gradient Boosting' in model_type or 'XGBoost' in model_type:
                    param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1]}
                elif 'Logistic' in model_type:
                    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
                elif 'Ridge' in model_type or 'Lasso' in model_type:
                    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
                elif 'SVC' in model_type or 'SVR' in model_type:
                    param_grid = {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
                elif 'KNN' in model_type:
                    param_grid = {'n_neighbors': [3, 5, 7, 9]}
                else:
                    param_grid = {}
                    
                if param_grid:
                    try:
                        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2' if task_type == 'regression' else 'accuracy')
                        grid.fit(X_train_scaled, y_train)
                        model = grid.best_estimator_
                    except Exception:
                        pass # Fallback to default if grid search fails
            
            # Train model
            if not tune_hyperparams:
                model.fit(X_train_scaled, y_train)
                
            y_pred = model.predict(X_test_scaled)
            
            # Compare train/test score to check for overfitting
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Calculate metrics
            results = {
                'model': model,
                'scaler': scaler,
                'feature_names': list(X.columns), # Use actual encoded columns
                'test_size': test_size,
                'predictions': y_pred,
                'actual': y_test,
                'indices': X_test.index.tolist(),
                'train_score': train_score,
                'test_score': test_score
            }
            
            if task_type == 'regression':
                results.update({
                    'metrics': {
                        'R2 Score': r2_score(y_test, y_pred),
                        'MSE': mean_squared_error(y_test, y_pred),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                    },
                    'residuals': y_test - y_pred
                })
            else:
                # Calculate Probabilities for ROC/AUC if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_test_scaled)
                        # Handle binary case specifically for ROC
                        if y_prob.shape[1] == 2:
                            y_prob = y_prob[:, 1] 
                    except:
                        pass
                
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                results.update({
                    'metrics': metrics,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'probabilities': y_prob,
                    'classes': getattr(model, 'classes_', None)
                })
                
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Ensure importances align with X columns (which might be more than feature_cols due to encoding)
                if len(importances) == len(X.columns):
                    results['feature_importance'] = dict(zip(X.columns, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
                if len(importances) == len(X.columns):
                    results['feature_importance'] = dict(zip(X.columns, importances))
                
            return results
            
        except Exception as e:
            return {'error': f"Training failed: {str(e)}"}

    def automl(self, df: pd.DataFrame, target_col: str, feature_cols: List[str],
               task_type: str = 'regression') -> Dict[str, Any]:
        """Run AutoML to find best model"""
        
        models_to_try = ['Random Forest', 'Gradient Boosting', 'Extra Trees']
        if XGBOOST_AVAILABLE:
            models_to_try.append('XGBoost')
        if task_type == 'regression':
            models_to_try.extend(['Linear Regression', 'Ridge'])
        else:
            models_to_try.extend(['Logistic Regression', 'Decision Tree'])
            
        best_score = -float('inf')
        best_model_result = None
        leaderboard = []
        errors = []
        
        for model_name in models_to_try:
            result = self.train_model(df, target_col, feature_cols, model_name, task_type)
            if 'error' not in result:
                score = result['metrics']['R2 Score'] if task_type == 'regression' else result['metrics']['Accuracy']
                leaderboard.append({
                    'Model': model_name,
                    'Score': score,
                    'Metrics': result['metrics']
                })
                
                if score > best_score:
                    best_score = score
                    best_model_result = result
                    best_model_result['model_name'] = model_name
            else:
                errors.append(f"{model_name}: {result['error']}")
        
        if not leaderboard:
            # Consolidate errors to give user feedback
            unique_errors = list(set(errors))
            error_msg = unique_errors[0] if len(unique_errors) == 1 else " | ".join(unique_errors[:3])
            return {'error': f"AutoML Failed. Reasons: {error_msg}"}
            
        leaderboard_df = pd.DataFrame(leaderboard)
        if 'Score' in leaderboard_df.columns:
            leaderboard_df = leaderboard_df.sort_values('Score', ascending=False)
            
        return {
            'best_model': best_model_result,
            'leaderboard': leaderboard_df
        }

def create_analytics_dashboard(df: pd.DataFrame, analytics_type: str, **kwargs) -> Dict[str, Any]:
    """Create analytics dashboard with results and visualizations"""
    
    analytics = PremiumAnalytics()
    
    if analytics_type == 'statistical_summary':
        return analytics.statistical_summary(df)
    elif analytics_type == 'correlation_analysis':
        return analytics.correlation_analysis(df)
    elif analytics_type == 'hypothesis_testing':
        return analytics.hypothesis_testing(df, **kwargs)
    elif analytics_type == 'clustering':
        return analytics.advanced_clustering(df, **kwargs)
    elif analytics_type == 'dimensionality_reduction':
        return analytics.dimensionality_reduction(df, **kwargs)
    elif analytics_type == 'anomaly_detection':
        return analytics.anomaly_detection(df, **kwargs)
    elif analytics_type == 'time_series':
        return analytics.time_series_analysis(df, **kwargs)
    elif analytics_type == 'feature_importance':
        return analytics.feature_importance_analysis(df, **kwargs)
    elif analytics_type == 'train_model':
        return analytics.train_model(df, **kwargs)
    elif analytics_type == 'automl':
        return analytics.automl(df, **kwargs)
    else:
        return {'error': f'Unknown analytics type: {analytics_type}'}