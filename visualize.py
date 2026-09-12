import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
from sklearn.inspection import permutation_importance

def plot_analytics():
    df = pd.read_csv('data/lrt_data.csv')
    saved_objects = joblib.load('models/lrt_model.joblib')
    with open('config.yaml', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    model = saved_objects['model']
    preprocessor = saved_objects['preprocessor']
    features = config['model']['features']['numeric'] + config['model']['features']['categorical']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    feature_names = preprocessor.get_feature_names_out()
    transformed_data = preprocessor.transform(df[features])
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        importances = permutation_importance(
            model,
            transformed_data,
            df['delay_minutes'],
            n_repeats=5,
            random_state=42,
            scoring='neg_mean_absolute_error',
        ).importances_mean
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', legend=False, ax=axes[0], palette='viridis')
    axes[0].set_title('Feature Importance')
    
    sns.scatterplot(data=df, x='lane_isolation_score', y='delay_minutes', hue='is_peak_hour', alpha=0.6, ax=axes[1])
    axes[1].set_title('Delay vs Lane Isolation')
    
    plt.tight_layout()
    plt.savefig('lrt_analytics.png')
    plt.close(fig)

if __name__ == "__main__":
    plot_analytics()