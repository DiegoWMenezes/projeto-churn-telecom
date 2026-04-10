"""Módulo para visualização de dados."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer:
    """Classe para criar visualizações."""

    def __init__(self, config: dict):
        self.config = config
        self.style = config['visualization']['style']
        self.palette = config['visualization']['palette']
        self.fig_size = tuple(config['visualization']['figure_size'])
        self.dpi = config['visualization']['dpi']
        self.output_path = Path(config['reports']['figures_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        plt.style.use(self.style)
        sns.set_palette(self.palette)

    def plot_churn_distribution(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """Gráfico de distribuição de churn."""
        churn_counts = df['churn'].value_counts()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribuição de Churn', 'Percentual'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )

        fig.add_trace(
            go.Bar(
                x=['Não Churn', 'Churn'],
                y=churn_counts.values,
                marker_color=['#2ecc71', '#e74c3c']
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(
                labels=['Não Churn', 'Churn'],
                values=churn_counts.values,
                marker_colors=['#2ecc71', '#e74c3c'],
                textinfo='percent+label'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title='Análise de Churn',
            showlegend=False,
            height=500
        )

        if save:
            fig.write_html(self.output_path / 'churn_distribution.html')

        return fig

    def plot_tenure_churn(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """Gráfico de churn por tenure."""
        df_copy = df.copy()
        df_copy['tenure_group'] = pd.cut(
            df_copy['tenure'],
            bins=[0, 6, 12, 24, 72],
            labels=['0-6', '6-12', '12-24', '24+']
        )

        churn_by_tenure = df_copy.groupby('tenure_group', observed=True).agg({
            'churn': ['sum', 'count']
        }).reset_index()
        churn_by_tenure.columns = ['tenure_group', 'churn_count', 'total']
        churn_by_tenure['churn_rate'] = churn_by_tenure['churn_count'] / churn_by_tenure['total']

        fig = px.bar(
            churn_by_tenure,
            x='tenure_group',
            y='churn_rate',
            text_auto='.1%',
            color='churn_rate',
            color_continuous_scale='RdYlGn_r',
            title='Taxa de Churn por Tenure'
        )

        fig.update_layout(
            xaxis_title='Tenure (meses)',
            yaxis_title='Taxa de Churn',
            showlegend=False
        )

        if save:
            fig.write_html(self.output_path / 'churn_by_tenure.html')

        return fig

    def plot_reclamation_impact(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """Gráfico do impacto de reclamações no churn."""
        churn_by_recl = df.groupby('num_reclamations_30d').agg({
            'churn': ['sum', 'count']
        }).reset_index()
        churn_by_recl.columns = ['reclamations', 'churn_count', 'total']
        churn_by_recl['churn_rate'] = churn_by_recl['churn_count'] / churn_by_recl['total']

        churn_by_recl = churn_by_recl[churn_by_recl['reclamations'] <= 7]

        fig = px.bar(
            churn_by_recl,
            x='reclamations',
            y='churn_rate',
            text_auto='.1%',
            color='churn_rate',
            color_continuous_scale='Reds',
            title='Impacto de Reclamações (30 dias) na Taxa de Churn'
        )

        fig.update_layout(
            xaxis_title='Número de Reclamações',
            yaxis_title='Taxa de Churn',
            showlegend=False
        )

        if save:
            fig.write_html(self.output_path / 'churn_by_reclamations.html')

        return fig

    def plot_contract_churn(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """Gráfico de churn por tipo de contrato."""
        churn_by_contract = df.groupby('contract_type').agg({
            'churn': ['sum', 'count']
        }).reset_index()
        churn_by_contract.columns = ['contract', 'churn_count', 'total']
        churn_by_contract['churn_rate'] = churn_by_contract['churn_count'] / churn_by_contract['total']
        churn_by_contract = churn_by_contract.sort_values('churn_rate', ascending=True)

        fig = px.bar(
            churn_by_contract,
            x='contract',
            y='churn_rate',
            text_auto='.1%',
            color='churn_rate',
            color_continuous_scale='RdYlGn_r',
            title='Taxa de Churn por Tipo de Contrato'
        )

        fig.update_layout(
            xaxis_title='Tipo de Contrato',
            yaxis_title='Taxa de Churn',
            showlegend=False
        )

        if save:
            fig.write_html(self.output_path / 'churn_by_contract.html')

        return fig

    def plot_feature_correlation(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """Mapa de calor de correlação entre features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if 'customer_id' in numeric_cols:
            numeric_cols.remove('customer_id')

        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Correlações entre Features'
        )

        fig.update_layout(
            width=800,
            height=800
        )

        if save:
            fig.write_html(self.output_path / 'feature_correlation.html')

        return fig

    def plot_model_comparison(self, results: dict, save: bool = True) -> go.Figure:
        """Gráfico comparativo de modelos."""
        models = list(results.keys())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Métricas de Classificação', 'ROC AUC Comparison')
        )

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(
                go.Bar(name=metric, x=models, y=values, texttemplate='%{y:.3f}'),
                row=1, col=1
            )

        auc_values = [results[m].get('roc_auc', 0) for m in models]
        fig.add_trace(
            go.Bar(name='ROC AUC', x=models, y=auc_values,
                   texttemplate='%{y:.3f}', marker_color='#3498db'),
            row=1, col=2
        )

        fig.update_layout(
            title='Comparação de Modelos',
            barmode='group',
            height=500
        )

        if save:
            fig.write_html(self.output_path / 'model_comparison.html')

        return fig

    def plot_confusion_matrix(self, cm: list, model_name: str,
                              save: bool = True) -> go.Figure:
        """Matriz de confusão interativa."""
        fig = px.imshow(
            np.array(cm),
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x='Predito', y='Real', color='Quantidade'),
            title=f'Matriz de Confusão - {model_name}'
        )

        fig.update_layout(
            xaxis=['Não Churn', 'Churn'],
            yaxis=['Não Churn', 'Churn']
        )

        if save:
            fig.write_html(self.output_path / f'confusion_matrix_{model_name}.html')

        return fig

    def plot_feature_importance(self, df_importance: pd.DataFrame,
                                top_n: int = 15,
                                save: bool = True) -> go.Figure:
        """Gráfico de importância de features."""
        df_top = df_importance.head(top_n).sort_values('importance', ascending=True)

        fig = px.bar(
            df_top,
            y='feature',
            x='importance',
            orientation='h',
            text_auto='.4f',
            color='importance',
            color_continuous_scale='Viridis',
            title=f'Top {top_n} Features Mais Importantes'
        )

        fig.update_layout(
            xaxis_title='Importância',
            yaxis_title='Feature'
        )

        if save:
            fig.write_html(self.output_path / 'feature_importance.html')

        return fig

    def plot_risk_segmentation(self, df: pd.DataFrame, risk_col: str = 'risk_score',
                               save: bool = True) -> go.Figure:
        """Gráfico de segmentação de risco."""
        df_copy = df.copy()
        df_copy['risk_segment'] = pd.cut(
            df_copy[risk_col],
            bins=[-1, 2, 5, 10, 20],
            labels=['Baixo', 'Médio', 'Alto', 'Crítico']
        )

        risk_stats = df_copy.groupby('risk_segment', observed=True).agg({
            'churn': ['sum', 'count']
        }).reset_index()
        risk_stats.columns = ['segment', 'churn_count', 'total']
        risk_stats['churn_rate'] = risk_stats['churn_count'] / risk_stats['total']
        risk_stats = risk_stats.sort_values('churn_rate', ascending=False)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Volume por Segmento', 'Taxa de Churn por Segmento'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        colors = {'Baixo': '#2ecc71', 'Médio': '#f39c12', 'Alto': '#e67e22', 'Crítico': '#e74c3c'}

        fig.add_trace(
            go.Bar(
                x=risk_stats['segment'],
                y=risk_stats['total'],
                marker_color=[colors.get(s, '#95a5a6') for s in risk_stats['segment']],
                text=risk_stats['total']
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=risk_stats['segment'],
                y=risk_stats['churn_rate'],
                marker_color=[colors.get(s, '#95a5a6') for s in risk_stats['segment']],
                text=[f'{v:.1%}' for v in risk_stats['churn_rate']]
            ),
            row=1, col=2
        )

        fig.update_layout(
            title='Segmentação de Risco de Churn',
            showlegend=False,
            height=500
        )

        if save:
            fig.write_html(self.output_path / 'risk_segmentation.html')

        return fig

    def create_dashboard_summary(self, df: pd.DataFrame, results: dict,
                                 df_importance: pd.DataFrame) -> go.Figure:
        """Cria dashboard resumido."""
        churn_rate = df['churn'].mean()
        avg_tenure = df['tenure'].mean()
        avg_monthly = df['monthly_charges'].mean()

        best_model = max(results, key=lambda x: results[x]['roc_auc'])
        best_auc = results[best_model]['roc_auc']

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Churn Rate', 'Tempo Médio de Cliente', 'Receita Média Mensal',
                'Contratos por Tipo', 'Top 5 Features', 'Performance do Modelo'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'pie'}, {'type': 'bar', 'orientation': 'v'}, {'type': 'bar', 'orientation': 'v'}]
            ]
        )

        fig.add_trace(
            go.Indicator(
                mode='number',
                value=churn_rate,
                suffix='%',
                number={'font': {'size': 40}},
                title={'text': 'Churn Rate'}
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Indicator(
                mode='number',
                value=avg_tenure,
                suffix=' meses',
                number={'font': {'size': 40}},
                title={'text': 'Tenure Médio'}
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Indicator(
                mode='currency',
                value=avg_monthly,
                number={'font': {'size': 40}},
                title={'text': 'Receita Média Mensal'}
            ),
            row=1, col=3
        )

        contract_counts = df['contract_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=contract_counts.index,
                values=contract_counts.values,
                hole=0.4
            ),
            row=2, col=1
        )

        top5 = df_importance.head(5).sort_values('importance', ascending=True)
        fig.add_trace(
            go.Bar(
                x=top5['importance'],
                y=top5['feature'],
                orientation='h',
                marker_color='#3498db'
            ),
            row=2, col=2
        )

        model_names = list(results.keys())
        aucs = [results[m]['roc_auc'] for m in model_names]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=aucs,
                marker_color=['#e74c3c' if m == best_model else '#3498db' for m in model_names],
                text=[f'{a:.3f}' for a in aucs]
            ),
            row=2, col=3
        )

        fig.update_layout(
            title='Dashboard de Churn - Resumo Executivo',
            height=800,
            showlegend=False
        )

        fig.write_html(self.output_path / 'dashboard_summary.html')

        return fig


def create_all_visualizations(df: pd.DataFrame, results: dict,
                              df_importance: pd.DataFrame,
                              config: dict) -> dict:
    """Função principal para criar todas as visualizações."""
    viz = DataVisualizer(config)

    figures = {
        'churn_distribution': viz.plot_churn_distribution(df),
        'tenure_churn': viz.plot_tenure_churn(df),
        'reclamation_impact': viz.plot_reclamation_impact(df),
        'contract_churn': viz.plot_contract_churn(df),
        'model_comparison': viz.plot_model_comparison(results),
        'feature_importance': viz.plot_feature_importance(df_importance),
        'risk_segmentation': viz.plot_risk_segmentation(df),
        'dashboard_summary': viz.create_dashboard_summary(df, results, df_importance)
    }

    best_model = max(results, key=lambda x: results[x]['roc_auc'])
    figures['confusion_matrix'] = viz.plot_confusion_matrix(
        results[best_model]['confusion_matrix'],
        best_model
    )

    return figures


if __name__ == "__main__":
    from src.data.load_data import generate_synthetic_data
    from src.features.build_features import create_features
    from src.models.train_model import train_models
    import yaml

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = generate_synthetic_data(n_samples=5000)
    df_processed = create_features(df, config, fit=True)

    target_col = config['model']['target_column']
    exclude_cols = [config['features']['id_column'], target_col, 'tenure_group']
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]

    X = df_processed[feature_cols]
    y = df_processed[target_col]

    trainer, results, importance = train_models(X, y, config)

    figures = create_all_visualizations(df_processed, results, importance, config)

    print("\nVisualizações salvas em:", config['reports']['figures_path'])
