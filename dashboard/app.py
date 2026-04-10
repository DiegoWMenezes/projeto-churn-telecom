"""Dashboard Streamlit para análise de churn."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import generate_synthetic_data
from src.features.build_features import create_features
from src.models.train_model import ModelTrainer
from src.visualization.visualize import DataVisualizer
import yaml

st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(n_samples=5000):
    """Carrega e processa os dados."""
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'config' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = generate_synthetic_data(n_samples=n_samples)
    df_processed = create_features(df, config, fit=True)

    return df, df_processed, config


@st.cache_resource
def train_models_cached(df_processed, config):
    """Treina modelos de machine learning."""
    target_col = config['model']['target_column']
    exclude_cols = [config['features']['id_column'], target_col, 'tenure_group']
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]

    X = df_processed[feature_cols]
    y = df_processed[target_col]

    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    results = trainer.train_all(X_train, y_train, X_test, y_test)
    importance = trainer.get_feature_importance()

    return trainer, results, importance


def main():
    st.markdown('<h1 class="main-header">📊 Análise de Churn em Telecomunicações</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df, df_processed, config = load_data()

    st.sidebar.title("Navegação")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Selecione a seção:",
        ["📈 Overview", "👥 Análise de Clientes", "📊 Churn por Segmentos",
         "🤖 Modelos de Predição", "🎯 Feature Importance", "⚠️ Score de Risco"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configurações")
    n_samples = st.sidebar.slider("Número de Amostras", 1000, 10000, 5000, step=1000)

    if page == "📈 Overview":
        overview_page(df, df_processed, config)

    elif page == "👥 Análise de Clientes":
        customer_analysis_page(df, df_processed)

    elif page == "📊 Churn por Segmentos":
        churn_segments_page(df, df_processed)

    elif page == "🤖 Modelos de Predição":
        models_page(df_processed, config)

    elif page == "🎯 Feature Importance":
        importance_page(df_processed, config)

    elif page == "⚠️ Score de Risco":
        risk_page(df, df_processed)


def overview_page(df, df_processed, config):
    """Página de overview."""
    st.title("📈 Visão Geral")

    col1, col2, col3, col4 = st.columns(4)

    churn_rate = df['churn'].mean()
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['monthly_charges'].mean()
    total_customers = len(df)

    col1.metric("Churn Rate", f"{churn_rate:.1%}", delta_color="inverse")
    col2.metric("Clientes Ativos", f"{total_customers:,}", delta_color="normal")
    col3.metric("Tenure Médio", f"{avg_tenure:.1f} meses")
    col4.metric("Receita Média", f"R$ {avg_monthly:.2f}", delta_color="normal")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Churn")
        churn_counts = df['churn'].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['Não Churn', 'Churn'],
                values=churn_counts.values,
                marker_colors=['#2ecc71', '#e74c3c'],
                textinfo='percent+label'
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn por Tipo de Contrato")
        viz = DataVisualizer(config)
        fig = viz.plot_contract_churn(df, save=False)
        st.plotly_chart(fig, use_container_width=True)


def customer_analysis_page(df, df_processed):
    """Página de análise de clientes."""
    st.title("👥 Análise de Clientes")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Clientes Feminino", f"{(df['gender'] == 'Female').mean():.1%}")
    col2.metric("Com Parceiros", f"{(df['partner'] == 'Yes').mean():.1%}")
    col3.metric("Com Dependentes", f"{(df['dependents'] == 'Yes').mean():.1%}")
    col4.metric("Idosos", f"{(df['senior_citizen'] == 1).mean():.1%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Tenure")
        fig = px.histogram(
            df, x='tenure', nbins=30,
            title='Histograma de Tenure',
            labels={'tenure': 'Meses como Cliente', 'count': 'Frequência'}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Receita Mensal por Contrato")
        fig = px.box(
            df, x='contract_type', y='monthly_charges',
            color='churn',
            title='Receita por Tipo de Contrato'
        )
        st.plotly_chart(fig, use_container_width=True)


def churn_segments_page(df, df_processed):
    """Página de churn por segmentos."""
    st.title("📊 Churn por Segmentos")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract_filter = st.multiselect(
            "Tipo de Contrato",
            options=df['contract_type'].unique(),
            default=df['contract_type'].unique()
        )

    with col2:
        service_filter = st.multiselect(
            "Serviço de Internet",
            options=df['internet_service'].unique(),
            default=df['internet_service'].unique()
        )

    with col3:
        tenure_range = st.slider(
            "Tenure (meses)",
            min_value=int(df['tenure'].min()),
            max_value=int(df['tenure'].max()),
            value=(0, int(df['tenure'].max()))
        )

    df_filtered = df[
        (df['contract_type'].isin(contract_filter)) &
        (df['internet_service'].isin(service_filter)) &
        (df['tenure'] >= tenure_range[0]) &
        (df['tenure'] <= tenure_range[1])
    ]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taxa de Churn por Tenure")
        df_copy = df_filtered.copy()
        df_copy['tenure_group'] = pd.cut(
            df_copy['tenure'], bins=[0, 6, 12, 24, 72],
            labels=['0-6', '6-12', '12-24', '24+']
        )
        churn_by_tenure = df_copy.groupby('tenure_group', observed=True).agg({
            'churn': 'mean'
        }).reset_index()

        fig = px.bar(
            churn_by_tenure, x='tenure_group', y='churn',
            text_auto='.1%', color='churn',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Impacto de Reclamações")
        churn_by_recl = df_filtered.groupby('num_reclamations_30d').agg({
            'churn': 'mean'
        }).reset_index()
        churn_by_recl = churn_by_recl[churn_by_recl['num_reclamations_30d'] <= 7]

        fig = px.line(
            churn_by_recl, x='num_reclamations_30d', y='churn',
            markers=True
        )
        fig.update_layout(
            xaxis_title='Número de Reclamações (30 dias)',
            yaxis_title='Taxa de Churn',
            yaxis_tickformat='.1%'
        )
        fig.add_trace(go.Scatter(
            x=churn_by_recl['num_reclamations_30d'],
            y=churn_by_recl['churn'],
            text=[f'{v:.1%}' for v in churn_by_recl['churn']],
            mode='text',
            textposition='top center',
            showlegend=False
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Segmentação de Risco")

    df_filtered['risk_segment'] = pd.cut(
        df_filtered['num_reclamations_30d'],
        bins=[-1, 0, 1, 3, 20],
        labels=['Sem Reclamação', '1 Reclamação', '2-3 Reclamações', '4+ Reclamações']
    )

    segment_table = df_filtered.groupby('risk_segment', observed=True).agg({
        'churn': ['count', 'sum', 'mean']
    }).reset_index()
    segment_table.columns = ['Segmento', 'Total Clientes', 'Churns', 'Taxa Churn']
    segment_table['Taxa Churn'] = segment_table['Taxa Churn'].apply(lambda x: f"{x:.1%}")

    st.dataframe(segment_table, use_container_width=True)


def models_page(df_processed, config):
    """Página de modelos."""
    st.title("🤖 Modelos de Predição")

    with st.spinner('Treinando modelos...'):
        trainer, results, importance = train_models_cached(df_processed, config)

    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_result = results[best_model_name]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Melhor Modelo", best_model_name.upper())
    col2.metric("ROC AUC", f"{best_result['roc_auc']:.3f}")
    col3.metric("Precisão", f"{best_result['precision']:.3f}")
    col4.metric("Recall", f"{best_result['recall']:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comparação de Modelos")

        model_names = list(results.keys())
        metrics_df = pd.DataFrame({
            'Modelo': model_names,
            'Accuracy': [results[m]['accuracy'] for m in model_names],
            'Precision': [results[m]['precision'] for m in model_names],
            'Recall': [results[m]['recall'] for m in model_names],
            'F1': [results[m]['f1'] for m in model_names],
            'ROC AUC': [results[m]['roc_auc'] for m in model_names]
        })

        fig = px.bar(
            metrics_df.melt(id_vars='Modelo', var_name='Métrica', value_name='Score'),
            x='Modelo', y='Score', color='Métrica', barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Matriz de Confusão - {best_model_name.upper()}")

        cm = np.array(best_result['confusion_matrix'])
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x='Predito', y='Real'),
            x=['Não Churn', 'Churn'],
            y=['Não Churn', 'Churn']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    st.code(best_result['classification_report'])


def importance_page(df_processed, config):
    """Página de feature importance."""
    st.title("🎯 Feature Importance")

    trainer, results, importance = train_models_cached(df_processed, config)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top 15 Features Mais Importantes")

        top15 = importance.head(15).sort_values('importance', ascending=True)

        fig = px.bar(
            top15, y='feature', x='importance', orientation='h',
            text_auto='.4f', color='importance', color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title='Importância',
            yaxis_title='Feature',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Principais Insights")

        insights = """
        **Reclamações** - Clientes com 3+ reclamações em 30 dias têm maior chance de churn.

        **Tenure** - Quanto menor o tempo de cliente, maior o risco de cancelamento.

        **Contrato** - Contratos month-to-month têm taxa de churn mais alta.

        **Serviços** - Clientes sem serviços adicionais são mais propensos a cancelar.

        **Engajamento** - Clientes engajados com múltiplos serviços churnam menos.
        """

        st.markdown(insights)


def risk_page(df, df_processed):
    """Página de score de risco."""
    st.title("⚠️ Score de Risco de Churn")

    df['risk_score'] = (
        (df['num_reclamations_30d'] >= 3).astype(int) * 5 +
        (df['num_reclamations_30d'] >= 1).astype(int) * 2 +
        (df['contract_type'] == 'Month-to-month').astype(int) * 3 +
        (df['tenure'] < 6).astype(int) * 2 +
        (df['num_support_tickets'] > 5).astype(int) * 2
    )

    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[-1, 2, 5, 10, 20],
        labels=['Baixo', 'Médio', 'Alto', 'Crítico']
    )

    col1, col2, col3, col4 = st.columns(4)

    counts = df['risk_level'].value_counts()
    for level, col in zip(['Baixo', 'Médio', 'Alto', 'Crítico'], [col1, col2, col3, col4]):
        count = counts.get(level, 0)
        pct = count / len(df)
        delta = "🔴" if level in ['Alto', 'Crítico'] else "🟡" if level == 'Médio' else "🟢"
        col.metric(f"{delta} {level}", f"{count:,}", f"{pct:.1%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição do Score de Risco")
        fig = px.histogram(
            df, x='risk_score', color='risk_level',
            category_orders={'risk_level': ['Baixo', 'Médio', 'Alto', 'Crítico']},
            color_discrete_map={
                'Baixo': '#2ecc71', 'Médio': '#f39c12',
                'Alto': '#e67e22', 'Crítico': '#e74c3c'
            }
        )
        fig.update_layout(
            xaxis_title='Score de Risco',
            yaxis_title='Número de Clientes',
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate por Nível de Risco")
        risk_churn = df.groupby('risk_level', observed=True).agg({
            'churn': ['count', 'sum', 'mean']
        }).reset_index()
        risk_churn.columns = ['Nível', 'Total', 'Churns', 'Taxa']
        risk_churn = risk_churn.sort_values('Taxa', ascending=False)

        fig = px.bar(
            risk_churn, x='Nível', y='Taxa', text_auto='.1%',
            color='Taxa', color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Clientes de Alto Risco (Score >= 5)")

    high_risk = df[df['risk_score'] >= 5][
        ['customer_id', 'tenure', 'contract_type', 'num_reclamations_30d',
         'num_support_tickets', 'risk_score', 'churn']
    ].sort_values('risk_score', ascending=False)

    st.dataframe(high_risk.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
