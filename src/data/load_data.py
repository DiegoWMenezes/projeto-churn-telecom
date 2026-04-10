"""
Módulo para carregamento e geração de dados.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class DataLoader:
    """Classe para carregar dados de diferentes fontes."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config['data']['raw_path'])
        self.processed_path = Path(config['data']['processed_path'])

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Carrega dados de um arquivo CSV."""
        filepath = self.raw_path / filename
        df = pd.read_csv(filepath)
        print(f"Dados carregados: {filepath}")
        print(f"Shape: {df.shape}")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str, processed: bool = True) -> None:
        """Salva dados em um arquivo CSV."""
        if processed:
            filepath = self.processed_path / filename
        else:
            filepath = self.raw_path / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dados salvos em: {filepath}")

    def load_processed(self, filename: str) -> pd.DataFrame:
        """Carrega dados processados."""
        filepath = self.processed_path / filename
        return pd.read_csv(filepath)


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Gera dataset sintético para análise de churn."""
    np.random.seed(random_state)

    customer_ids = [f"CUST_{i:06d}" for i in range(n_samples)]
    genders = np.random.choice(['Male', 'Female'], n_samples)
    ages = np.random.randint(18, 70, n_samples)
    partners = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

    phone_services = np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'],
                                       n_samples, p=[0.4, 0.4, 0.2])
    internet_services = np.random.choice(['Fiber optic', 'DSL', 'No'],
                                         n_samples, p=[0.45, 0.35, 0.2])

    online_security = np.random.choice(['Yes', 'No', 'No internet service'],
                                       n_samples, p=[0.3, 0.5, 0.2])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'],
                                     n_samples, p=[0.35, 0.45, 0.2])
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'],
                                         n_samples, p=[0.3, 0.5, 0.2])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'],
                                    n_samples, p=[0.25, 0.55, 0.2])
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'],
                                     n_samples, p=[0.35, 0.45, 0.2])
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'],
                                        n_samples, p=[0.35, 0.45, 0.2])

    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                     n_samples, p=[0.5, 0.3, 0.2])
    payment_methods = np.random.choice(['Electronic check', 'Mailed check',
                                        'Bank transfer (automatic)',
                                        'Credit card (automatic)'],
                                       n_samples, p=[0.35, 0.15, 0.3, 0.2])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])

    tenures = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenures = np.clip(tenures, 0, 72)

    monthly_charges = np.random.uniform(20, 120, n_samples)
    monthly_charges = np.where(internet_services == 'Fiber optic',
                               monthly_charges + 30, monthly_charges)
    monthly_charges = np.where(
        (streaming_tv == 'Yes') & (streaming_movies == 'Yes'),
        monthly_charges + 15, monthly_charges
    )

    total_charges = []
    for i in range(n_samples):
        total = monthly_charges[i] * tenures[i] + np.random.normal(0, 50)
        total_charges.append(max(0, total))
    total_charges = np.array(total_charges)

    num_support_tickets = np.random.poisson(1.5, n_samples)
    num_reclamations_30d = np.random.poisson(0.5, n_samples)

    churn_prob = np.zeros(n_samples)
    churn_prob[:] = 0.15

    churn_prob = np.where(tenures < 6, churn_prob + 0.25, churn_prob)
    churn_prob = np.where(tenures < 12, churn_prob + 0.10, churn_prob)
    churn_prob = np.where(tenures > 24, churn_prob - 0.08, churn_prob)

    churn_prob = np.where(num_reclamations_30d >= 3, churn_prob + 0.35, churn_prob)
    churn_prob = np.where(num_reclamations_30d >= 1, churn_prob + 0.10, churn_prob)

    churn_prob = np.where(contract_types == 'Month-to-month', churn_prob + 0.15, churn_prob)
    churn_prob = np.where(contract_types == 'Two year', churn_prob - 0.10, churn_prob)

    churn_prob = np.where(num_support_tickets >= 5, churn_prob + 0.20, churn_prob)

    fiber_mtm = (internet_services == 'Fiber optic') & (contract_types == 'Month-to-month')
    churn_prob = np.where(fiber_mtm, churn_prob + 0.10, churn_prob)

    churn_prob = np.clip(churn_prob, 0.01, 0.95)

    actual_churn = []
    for i in range(n_samples):
        actual_churn.append(1 if np.random.random() < churn_prob[i] else 0)
    churns = np.array(actual_churn)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'senior_citizen': (ages > 65).astype(int),
        'partner': partners,
        'dependents': dependents,
        'tenure': tenures,
        'phone_service': phone_services,
        'multiple_lines': multiple_lines,
        'internet_service': internet_services,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract_type': contract_types,
        'paperless_billing': paperless_billing,
        'payment_method': payment_methods,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'num_support_tickets': num_support_tickets,
        'num_reclamations_30d': num_reclamations_30d,
        'churn': churns
    })

    return df


if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=5000)
    print(f"Dados gerados: {df.shape}")
    print(f"Churn Rate: {df['churn'].mean():.2%}")
    print(df.head())
