import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta  # Technical Analysis library for Python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

npNaN = np.nan

class StockPriceMovementPredictor:
    def __init__(self):
        self.model = LinearRegression()
        
        # Feature groups based on project specification
        self.feature_groups = {
            'price_volume': [
                'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'PREV_CLOSE', 
                'AVG_PRICE', 'TTL_TRD_QNTY', 'DELIV_QTY', 'DELIV_PER'
            ],
            'derived_metrics': [
                'CLO', 'HiLo', 'ClOP_per', 'Wick_Percentage', 'HiLo_Percentage', 
                'vol_dis', 'closing_strength'
            ],
            'delivery_volatility': [
                'delivery_vol_ratio', 'delivery_Factor', 'delivery_density', 
                'volatility_score', 'NR7'
            ],
            'breakout_signals': [
                'ATR_ratio', 'BB_width_pct', 'volume_zscore', 'volume_ratio', 
                'Range20', 'Range20_percentile', 'is_narrow_range', 
                'is_volume_spike', 'pre_breakout_candidate'
            ]
        }
        
        self.results = {}
    
    def load_data(self, file_path):
        """Load the stock data CSV file"""
        print("Loading stock data...")
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Clean the data - handle Excel errors and non-numeric values
            print("Cleaning data...")
            
            # Replace Excel error values with NaN
            df = df.replace(['#NAME?', '#VALUE!', '#REF!', '#DIV/0!', '#N/A', '#NULL!'], np.nan)
            

            # Get numeric columns (excluding SYMBOL, DATE1, and other text columns)
            text_columns = ['SYMBOL', 'DATE1', 'Signal']  # Add other text columns if any
            
            numeric_columns = [col for col in df.columns if col not in text_columns]
            
            # Convert numeric columns to float, replacing non-numeric values with NaN
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['ATR_ratio'] = ta.atr(df['HIGH_PRICE'], df['LOW_PRICE'], df['CLOSE_PRICE'], length=14) / df['CLOSE_PRICE']
            df['BB_width_pct'] = ta.bbands(df['CLOSE_PRICE'], length=20).iloc[:, -1] / df['CLOSE_PRICE']  # % BB width
            df['volume_zscore'] = (df['TTL_TRD_QNTY'] - df['TTL_TRD_QNTY'].rolling(20).mean()) / df['TTL_TRD_QNTY'].rolling(20).std()
            
            
            # Report data quality
            total_values = df.shape[0] * df.shape[1]
            nan_count = df.isnull().sum().sum()
            print(f"Data cleaning completed:")
            print(f"  - Total values: {total_values:,}")
            print(f"  - NaN values: {nan_count:,} ({nan_count/total_values*100:.2f}%)")
            
            # Show columns with high NaN percentage
            high_nan_cols = df.isnull().sum() / len(df)
            problematic_cols = high_nan_cols[high_nan_cols > 0.5]
            if len(problematic_cols) > 0:
                print(f"  - Columns with >50% missing data: {list(problematic_cols.index)}")
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_target_variable(self, df):
        """
        Task 1: Target Generation
        Compute average of next 5 closing prices and subtract today's close
        """
        print("\n=== TASK 1: TARGET GENERATION ===")
        
        # Ensure data is sorted by symbol and date
        # Handle DD-MM-YYYY date format (dayfirst=True)
        df['DATE1'] = pd.to_datetime(df['DATE1'], dayfirst=True, errors='coerce')
        
        # Check if any dates failed to parse
        if df['DATE1'].isnull().any():
            print(f"Warning: {df['DATE1'].isnull().sum()} dates could not be parsed")
            # Remove rows with invalid dates
            df = df.dropna(subset=['DATE1'])
        
        df = df.sort_values(['SYMBOL', 'DATE1']).reset_index(drop=True)
        
        # Calculate 5-day forward return for each stock
        def calculate_5day_return(group):
            group = group.sort_values('DATE1')
            # Get next 5 closing prices
            group['next_5_avg'] = group['CLOSE_PRICE'].shift(-1).rolling(window=5, min_periods=5).mean()
            # Calculate 5-day forward return
            group['target_5day_return'] = group['next_5_avg'] - group['CLOSE_PRICE']
            return group
        
        df = df.groupby('SYMBOL').apply(calculate_5day_return).reset_index(drop=True)
        
        # Remove rows where target cannot be calculated
        initial_rows = len(df)
        df = df.dropna(subset=['target_5day_return'])
        final_rows = len(df)
        
        print(f"Target variable created successfully")
        print(f"Rows with valid targets: {final_rows} (removed {initial_rows - final_rows} rows)")
        print(f"Target statistics:")
        print(df['target_5day_return'].describe())
        
        return df
    
    def exploratory_data_analysis(self, df):
        """
        Task 2: EDA + Correlation
        Study relationships between features and target, grouped by categories
        """
        print("\n=== TASK 2: EXPLORATORY DATA ANALYSIS ===")
        
        # Get all available features from the dataset
        all_features = []
        available_features = {}
        
        for group_name, features in self.feature_groups.items():
            available_features[group_name] = [f for f in features if f in df.columns]
            all_features.extend(available_features[group_name])
        
        print(f"Available features by group:")
        for group, features in available_features.items():
            print(f"  {group}: {len(features)} features")
        
        # Calculate correlations with target - handle NaN values properly
        correlations = {}
        for group_name, features in available_features.items():
            if features:  # Only if features exist in dataset
                # Select only numeric columns and handle NaN
                group_data = df[features + ['target_5day_return']].select_dtypes(include=[np.number])
                
                # Only calculate correlation if we have valid data
                if len(group_data.columns) > 1 and group_data.notnull().sum().sum() > 0:
                    try:
                        group_corr = group_data.corr()['target_5day_return'].drop('target_5day_return', errors='ignore')
                        # Remove NaN correlations
                        group_corr = group_corr.dropna()
                        if len(group_corr) > 0:
                            correlations[group_name] = group_corr.sort_values(key=abs, ascending=False)
                    except Exception as e:
                        print(f"Warning: Could not calculate correlations for {group_name}: {e}")
        
        # Visualization - only plot groups that have valid correlations
        valid_groups = [group for group in correlations.keys() if len(correlations[group]) > 0]
        
        if len(valid_groups) > 0:
            n_groups = min(len(valid_groups), 4)  # Maximum 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            axes = axes.flatten()
            
            for i, group_name in enumerate(valid_groups[:4]):
                corr_data = correlations[group_name]
                ax = axes[i]
                
                # Create horizontal bar plot
                colors = ['red' if x < 0 else 'blue' for x in corr_data.values]
                bars = ax.barh(range(len(corr_data)), corr_data.values, color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(corr_data)))
                ax.set_yticklabels(corr_data.index, fontsize=8)
                ax.set_xlabel('Correlation with 5-Day Return')
                ax.set_title(f'{group_name.replace("_", " ").title()} Features')
                ax.set_xlim(-1, 1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(valid_groups), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        else:
            print("Warning: No valid correlations found to plot")
        
        # Overall top correlations
        if correlations:
            all_correlations = pd.concat(correlations.values()).sort_values(key=abs, ascending=False)
            print(f"\nTop 15 Features by Absolute Correlation:")
            print(all_correlations.head(15))
        else:
            print("Warning: No correlations could be calculated")
            all_correlations = pd.Series(dtype=float)
        
        self.results['correlations'] = correlations
        self.results['available_features'] = available_features
        
        return df, all_correlations
    
    def build_regression_model(self, df):
        """
        Task 3: Modeling
        Build Linear Regression model and analyze feature weights
        """
        print("\n=== TASK 3: REGRESSION MODELING ===")
        
        # Prepare features
        all_features = []
        for features in self.results['available_features'].values():
            all_features.extend(features)
        
        # Remove any features that might have too many missing values
        feature_completeness = df[all_features].isnull().sum() / len(df)
        valid_features = feature_completeness[feature_completeness < 0.1].index.tolist()
        
        print(f"Using {len(valid_features)} features with <10% missing values")
        
        # Prepare data with proper cleaning
        X = df[valid_features].copy()
        y = df['target_5day_return'].copy()
        
        # Clean infinite and extreme values
        print("Cleaning infinite and extreme values...")
        
        # Replace infinity with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values at 99.9th percentile to handle outliers
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                q99 = X[col].quantile(0.999)
                q01 = X[col].quantile(0.001)
                X[col] = X[col].clip(lower=q01, upper=q99)
        
        # Cap target variable extreme values
        y_q99 = y.quantile(0.999)
        y_q01 = y.quantile(0.001)
        y = y.clip(lower=y_q01, upper=y_q99)
        
        # Fill remaining NaN with median (more robust than 0)
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Final check for any remaining invalid values
        invalid_rows = ~(np.isfinite(X).all(axis=1) & np.isfinite(y))
        if invalid_rows.sum() > 0:
            print(f"Removing {invalid_rows.sum()} rows with invalid values")
            X = X[~invalid_rows]
            y = y[~invalid_rows]
        
        print(f"Final dataset: {len(X)} rows, {len(X.columns)} features")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Train model
        print("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Model performance
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\nModel Performance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        
        # Feature weights analysis
        feature_weights = pd.DataFrame({
            'feature': valid_features,
            'weight': self.model.coef_,
            'abs_weight': np.abs(self.model.coef_)
        }).sort_values('abs_weight', ascending=False)
        
        print(f"\nTop 15 Most Important Features by Weight:")
        print(feature_weights.head(15)[['feature', 'weight', 'abs_weight']])
        
        # Group importance analysis
        group_importance = {}
        for group_name, group_features in self.results['available_features'].items():
            group_features_in_model = [f for f in group_features if f in valid_features]
            if group_features_in_model:
                group_weights = feature_weights[feature_weights['feature'].isin(group_features_in_model)]
                group_importance[group_name] = group_weights['abs_weight'].sum()
        
        # Sort group importance
        sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nFeature Group Importance (Sum of Absolute Weights):")
        for group, importance in sorted_groups:
            print(f"{group.replace('_', ' ').title()}: {importance:.4f}")
        
        # Visualization of top features
        plt.figure(figsize=(12, 8))
        top_features = feature_weights.head(20)
        colors = ['red' if w < 0 else 'blue' for w in top_features['weight']]
        
        plt.barh(range(len(top_features)), top_features['weight'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Model Weight')
        plt.title('Top 20 Feature Weights in Linear Regression Model')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Group importance visualization
        plt.figure(figsize=(10, 6))
        groups, importances = zip(*sorted_groups)
        plt.bar(groups, importances, alpha=0.7, color='steelblue')
        plt.xlabel('Feature Groups')
        plt.ylabel('Total Absolute Weight')
        plt.title('Feature Group Importance in Predicting 5-Day Price Movement')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Store results
        self.results.update({
            'model_performance': {
                'train_r2': train_r2, 'test_r2': test_r2,
                'train_rmse': train_rmse, 'test_rmse': test_rmse
            },
            'feature_weights': feature_weights,
            'group_importance': dict(sorted_groups),
            'features_used': valid_features
        })
        
        return feature_weights, dict(sorted_groups)
    
    def generate_insights(self):
        """Generate final insights about what drives stock price movement"""
        print("\n=== FINAL INSIGHTS: WHAT DRIVES STOCK PRICE MOVEMENT ===")
        
        print("Key Findings:")
        print("-" * 50)
        
        # Top driving factors
        top_groups = list(self.results['group_importance'].items())[:2]
        print(f"1. PRIMARY DRIVERS:")
        for group, importance in top_groups:
            print(f"   • {group.replace('_', ' ').title()}: {importance:.4f}")
        
        # Most important individual features
        top_features = self.results['feature_weights'].head(5)
        print(f"\n2. TOP INDIVIDUAL PREDICTORS:")
        for _, row in top_features.iterrows():
            direction = "↑" if row['weight'] > 0 else "↓"
            print(f"   • {row['feature']}: {row['weight']:.4f} {direction}")
        
        # Model effectiveness
        test_r2 = self.results['model_performance']['test_r2']
        print(f"\n3. MODEL EFFECTIVENESS:")
        print(f"   • Explains {test_r2*100:.1f}% of price movement variance")
        print(f"   • {'Strong' if test_r2 > 0.1 else 'Moderate' if test_r2 > 0.05 else 'Weak'} predictive power")
        
        print(f"\n4. INVESTMENT IMPLICATIONS:")
        if 'breakout_signals' in [g[0] for g in top_groups]:
            print("   • Technical breakout patterns are key predictors")
        if 'delivery_volatility' in [g[0] for g in top_groups]:
            print("   • Delivery and volatility metrics drive short-term moves")
        if 'price_volume' in [g[0] for g in top_groups]:
            print("   • Traditional price/volume signals remain important")
    
    def run_complete_analysis(self, file_path):
        """Run the complete analysis pipeline"""
        print("=== STOCK PRICE MOVEMENT PREDICTION ANALYSIS ===")
        print("Multi-Factor Modeling Using Technical & Volume Signals")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Step 2: Create target variable
        df = self.create_target_variable(df)
        
        # Step 3: EDA and correlation analysis
        df, correlations = self.exploratory_data_analysis(df)
        
        # Step 4: Build regression model
        feature_weights, group_importance = self.build_regression_model(df)
        
        # Step 5: Generate insights
        self.generate_insights()
        
        return self.results

# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockPriceMovementPredictor()
    
    # Run complete analysis
    # Note: Replace with actual file path when running
    results = predictor.run_complete_analysis('merged_cash_stocks.csv')
    
    # Additional analysis can be performed using the results dictionary
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Results stored in predictor.results dictionary")
