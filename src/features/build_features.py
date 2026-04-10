"""Módulo para feature engineering."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    """Classe para engenharia de features."""

    def __init__(self, config: dict):
        self.config = config
        self.categorical_cols = config['features']['categorical_columns']
        self.numerical_cols = config['features']['numerical_columns']
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def create_tenure_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria grupos de tenure."""
        def tenure_group(tenure):
            if tenure < 6:
                return '0-6 months'
            elif tenure < 12:
                return '6-12 months'
            elif tenure < 24:
                return '12-24 months'
            else:
                return '24+ months'

        df['tenure_group'] = df['tenure'].apply(tenure_group)
        return df

    def create_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria score de engajamento do cliente."""
        service_cols = ['phone_service', 'multiple_lines', 'internet_service',
                       'online_security', 'online_backup', 'device_protection',
                       'tech_support', 'streaming_tv', 'streaming_movies']

        df['num_services'] = 0
        for col in service_cols:
            if col in df.columns:
                df['num_services'] += (df[col] == 'Yes').astype(int)

        df['engagement_score'] = (
            df['num_services'] * 2 +
            (df['tenure'] / 12) * 3 -
            df['num_support_tickets'] * 2 -
            df['num_reclamations_30d'] * 5
        )

        return df

    def create_charge_per_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria métricas de cobrança por tenure."""
        df['charge_per_month'] = np.where(
            df['tenure'] > 0,
            df['total_charges'] / df['tenure'],
            df['monthly_charges']
        )

        avg_charge = df['monthly_charges'].mean()
        df['charge_vs_avg'] = df['monthly_charges'] - avg_charge

        return df

    def create_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria flags de risco para churn."""
        df['high_reclamation_risk'] = (df['num_reclamations_30d'] >= 3).astype(int)

        df['medium_reclamation_risk'] = (
            (df['num_reclamations_30d'] >= 1) & (df['num_reclamations_30d'] < 3)
        ).astype(int)

        df['mtm_contract_risk'] = (df['contract_type'] == 'Month-to-month').astype(int)

        df['low_tenure_risk'] = (df['tenure'] < 6).astype(int)

        df['risk_score'] = (
            df['high_reclamation_risk'] * 5 +
            df['medium_reclamation_risk'] * 2 +
            df['mtm_contract_risk'] * 3 +
            df['low_tenure_risk'] * 2 +
            (df['num_support_tickets'] > 5).astype(int) * 2
        )

        return df

    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features relacionadas a família."""
        df['has_family'] = ((df['partner'] == 'Yes') | (df['dependents'] == 'Yes')).astype(int)

        df['family_size'] = (
            (df['partner'] == 'Yes').astype(int) +
            (df['dependents'] == 'Yes').astype(int) * 2
        )

        return df

    def create_service_bundles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica pacotes de serviços."""
        df['has_streaming_bundle'] = (
            (df['streaming_tv'] == 'Yes') & (df['streaming_movies'] == 'Yes')
        ).astype(int)

        df['has_security_bundle'] = (
            (df['online_security'] == 'Yes') &
            (df['device_protection'] == 'Yes') &
            (df['tech_support'] == 'Yes')
        ).astype(int)

        df['has_backup_bundle'] = (
            (df['online_backup'] == 'Yes') & (df['device_protection'] == 'Yes')
        ).astype(int)

        return df

    def apply_encoding(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Aplica encoding nas variáveis categóricas."""
        df_encoded = df.copy()

        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))

        return df_encoded

    def create_all_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Aplica todas as transformações de features."""
        df = df.copy()

        df = self.create_tenure_group(df)
        df = self.create_engagement_score(df)
        df = self.create_charge_per_tenure(df)
        df = self.create_risk_flags(df)
        df = self.create_family_features(df)
        df = self.create_service_bundles(df)

        df = self.apply_encoding(df, fit=fit)

        return df


def create_features(df: pd.DataFrame, config: dict, fit: bool = True) -> pd.DataFrame:
    """Função principal para criação de features."""
    engineer = FeatureEngineer(config)
    return engineer.create_all_features(df, fit=fit)


if __name__ == "__main__":
    from src.data.load_data import generate_synthetic_data
    import yaml

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = generate_synthetic_data(n_samples=1000)
    df_features = create_features(df, config, fit=True)

    print(f"Shape original: {df.shape}")
    print(f"Shape com features: {df_features.shape}")
    print(df_features.columns.tolist())
