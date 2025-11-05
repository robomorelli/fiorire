import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Tuple
import sys
import argparse
import os
from omegaconf import OmegaConf
from anomalies_injection.utils import load_data
from anomalies_injection.utils import ANOMALIES_REGISTRY

'''
ANOMALIES_REGISTRY = {
    'GWN': GWN,
    'Constant': Constant,
    'Step': Step,
    'Impulse': Impulse,
    'GNN': GNN
}

Ogni funzione nel registry ha la firma:
def anomaly_function(window: np.ndarray, delta: float) -> np.ndarray
Dove:
    - window ha shape (window_length, n_features)
    - delta è il parametro di intensità dell'anomalia
'''

from config import *


class StandardizationHandler:
    """
    Gestisce la standardizzazione e de-standardizzazione dei dati
    garantendo che i valori originali NON anomali rimangano identici
    """

    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
        """
        Standardizza il dataframe e salva i parametri per la trasformazione inversa
        """
        df_standardized = df.copy()

        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_columns = feature_columns

        for col in feature_columns:
            self.mean_dict[col] = df[col].mean()
            self.std_dict[col] = df[col].std()
            df_standardized[col] = (df[col] - self.mean_dict[col]) / self.std_dict[col]

        print(f"\n✓ Standardizzazione completata su {len(feature_columns)} features")
        print(f"  Parametri salvati:")
        for col in feature_columns[:3]:
            print(f"    - {col}: μ={self.mean_dict[col]:.4f}, σ={self.std_dict[col]:.4f}")
        if len(feature_columns) > 3:
            print(f"    ... e altre {len(feature_columns) - 3} features")

        return df_standardized

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trasforma il dataframe standardizzato ai valori originali
        """
        df_destandardized = df.copy()

        for col in self.feature_columns:
            df_destandardized[col] = (df[col] * self.std_dict[col]) + self.mean_dict[col]

        print(f"\n✓ De-standardizzazione completata")

        return df_destandardized


class TimeSeriesAnomalyInjector:
    """
    Inietta anomalie in serie temporali multivariate compatibile con WOMBAT:
    - Estrae finestre con mean=0 e std=1
    - Applica anomalie con parametro delta da distribuzione normale
    - Reinserisce le finestre modificate nel dataframe
    """

    def __init__(
            self,
            anomaly_percentage: float = 5.0,
            window_mean: int = 50,
            window_skewness: float = 2.0,
            min_channels: int = 1,
            max_channels: int = None,
            channel_prob_decay: float = 0.7,
            delta_mean: float = 0.8,
            delta_std: float = 0.5,
            random_seed: int = None,
            anomaly_registry: dict = None
    ):
        """
        Parameters:
        -----------
        anomaly_percentage : float
            Percentuale di punti anomali rispetto al totale
        window_mean : int
            Lunghezza media delle finestre anomale
        window_skewness : float
            Skewness della distribuzione delle lunghezze
        min_channels : int
            Numero minimo di canali da rendere anomali
        max_channels : int
            Numero massimo di canali da rendere anomali
        channel_prob_decay : float
            Fattore di decadimento della probabilità per canale aggiuntivo
        delta_mean : float
            Media della distribuzione normale per il parametro delta
        delta_std : float
            Deviazione standard della distribuzione normale per delta
        random_seed : int
            Seed per riproducibilità
        anomaly_registry : dict
            Dizionario con le funzioni di iniezione anomalie
        """
        self.anomaly_percentage = anomaly_percentage
        self.window_mean = window_mean
        self.window_skewness = window_skewness
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.channel_prob_decay = channel_prob_decay
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self.random_seed = random_seed

        # Usa il registry fornito o quello di default
        self.anomaly_registry = anomaly_registry if anomaly_registry is not None else ANOMALIES_REGISTRY

        if not self.anomaly_registry:
            raise ValueError("ANOMALIES_REGISTRY è vuoto!")

        print(f"\n✓ Anomalie disponibili: {list(self.anomaly_registry.keys())}")
        print(f"✓ Delta: μ={delta_mean:.2f}, σ={delta_std:.2f}")

        if random_seed is not None:
            np.random.seed(random_seed)

    def _generate_skewed_window_length(self, fixed_mean=False) -> int:
        """Genera lunghezza finestra con distribuzione skew-normal"""
        if not fixed_mean:
            length = stats.skewnorm.rvs(
            a=self.window_skewness,
            loc=self.window_mean,
            scale=self.window_mean * 0.3,
            size=1
            )[0]
        else:
            length = self.window_mean

        return max(5, int(round(length)))

    def _generate_delta(self, fixed_delta=False) -> float:
        """
        Genera il parametro delta da una distribuzione normale

        Returns:
        --------
        float
            Valore di delta campionato da N(delta_mean, delta_std)
        """
        if not fixed_delta:
            delta = np.random.normal(self.delta_mean, self.delta_std)
            # Assicura che delta sia positivo
            delta = max(0.1, delta)
        else:
            delta = self.delta_mean
        return delta

    def _generate_channel_probabilities(self, n_channels: int) -> np.ndarray:
        """Genera probabilità decrescenti per il numero di canali anomali"""
        max_ch = self.max_channels if self.max_channels else n_channels
        max_ch = min(max_ch, n_channels)

        probs = []
        for i in range(self.min_channels, max_ch + 1):
            prob = self.channel_prob_decay ** (i - self.min_channels)
            probs.append(prob)

        probs = np.array(probs)
        probs = probs / probs.sum()
        return probs

    def _select_num_channels(self, n_channels: int) -> int:
        """Seleziona il numero di canali da rendere anomali"""
        max_ch = self.max_channels if self.max_channels else n_channels
        max_ch = min(max_ch, n_channels)

        possible_channels = list(range(self.min_channels, max_ch + 1))
        probs = self._generate_channel_probabilities(n_channels)

        return np.random.choice(possible_channels, p=probs)

    def _extract_window(
            self,
            df: pd.DataFrame,
            start_idx: int,
            window_length: int,
            feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estrae una finestra dal dataframe in formato WOMBAT

        WOMBAT richiede:
        - Shape: (window_length, n_features)
        - Mean = 0 per ogni feature
        - Std = 1 per ogni feature (unit power)

        Returns:
        --------
        window_normalized : np.ndarray
            Finestra normalizzata (mean=0, std=1)
        window_mean : np.ndarray
            Media originale della finestra
        window_std : np.ndarray
            Std originale della finestra
        """
        end_idx = start_idx + window_length

        # Estrai la finestra
        window = df.iloc[start_idx:end_idx][feature_columns].values

        # Normalizza la finestra (WOMBAT requirement)
        window_mean = window.mean(axis=0, keepdims=True)
        window_std = window.std(axis=0, keepdims=True)

        # Evita divisione per zero
        window_std[window_std == 0] = 1.0

        # Normalizza: mean=0, std=1
        window_normalized = (window - window_mean) / window_std

        return window_normalized, window_mean, window_std

    def _inject_window_anomaly(
            self,
            window: np.ndarray,
            channels_to_modify: List[int],
            delta: float,
            anomaly_type: str = 'random'
    ) -> np.ndarray:
        """
        Inietta anomalia in una finestra usando il registro WOMBAT

        Parameters:
        -----------
        window : np.ndarray
            Finestra con shape (window_length, n_features), mean=0, std=1
        channels_to_modify : List[int]
            Indici dei canali da modificare
        delta : float
            Parametro di intensità dell'anomalia
        anomaly_type : str
            Tipo di anomalia da iniettare

        Returns:
        --------
        np.ndarray
            Finestra con anomalie iniettate
        """
        # Seleziona tipo di anomalia
        if anomaly_type == 'random':
            anomaly_type = np.random.choice(list(self.anomaly_registry.keys()))

        # Verifica che il tipo sia nel registry
        if anomaly_type not in self.anomaly_registry:
            available_types = list(self.anomaly_registry.keys())
            print(f"⚠ Warning: Tipo anomalia '{anomaly_type}' non trovato.")
            print(f"   Uso '{available_types[0]}' come fallback.")
            anomaly_type = available_types[0]

        for k in self.anomaly_registry:
            fitted_ans_funcs = self.anomaly_registry[anomaly_type]

        # Ottieni la funzione dal registry
        anomaly_function = self.anomaly_registry[anomaly_type](delta)

        # Copia la finestra
        window_anomalous = window.copy()

        try:
            # Estrai solo i canali da modificare
            window_to_modify = window[:, channels_to_modify]  # (window_length, n_channels_selected)

            # Applica l'anomalia con delta
            # La funzione WOMBAT riceve la finestra e il parametro delta
            window_to_modify_ = np.reshape(window_to_modify[:,0], (1, window_to_modify.shape[0]))
            window_modified = anomaly_function.fit(window_to_modify, delta)

            # Sostituisci solo i canali modificati
            window_anomalous[:, channels_to_modify] = window_modified

            return window_anomalous

        except Exception as e:
            print(f"⚠ Errore nell'applicazione dell'anomalia '{anomaly_type}': {e}")
            print(f"   Ritorno la finestra originale.")
            return window.copy()

    def _denormalize_window(
            self,
            window_normalized: np.ndarray,
            window_mean: np.ndarray,
            window_std: np.ndarray
    ) -> np.ndarray:
        """
        De-normalizza una finestra riportandola alla scala standardizzata originale
        """
        return (window_normalized * window_std) + window_mean

    def inject_anomalies(
            self,
            df: pd.DataFrame,
            feature_columns: List[str] = None,
            output_filename: str = None,
            block_percentage: float = 10.0
    ) -> pd.DataFrame:
        """
        Inietta anomalie nel dataframe usando strategia WOMBAT

        Pipeline:
        1. Estrai finestra dal dataframe standardizzato
        2. Normalizza finestra (mean=0, std=1 per ogni feature) - WOMBAT requirement
        3. Genera delta da distribuzione normale N(delta_mean, delta_std)
        4. Applica anomalia usando funzioni WOMBAT dal registry con delta
        5. De-normalizza finestra
        6. Reinserisci nel dataframe
        """
        df_result = df.copy()

        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        n_channels = len(feature_columns)
        n_points = len(df)
        n_anomalous_points = int(n_points * self.anomaly_percentage / 100)

        print(f"\nDataset info:")
        print(f"  - Lunghezza totale: {n_points}")
        print(f"  - Numero di features: {n_channels}")
        print(f"  - Features: {feature_columns}")
        print(f"  - Punti anomali target: {n_anomalous_points} ({self.anomaly_percentage}%)")
        print(f"  - Lunghezza media finestre: {self.window_mean}")
        print(f"  - Delta: μ={self.delta_mean:.2f}, σ={self.delta_std:.2f}")
        print(f"  - Strategia: Finestre WOMBAT-style + algoritmo a blocchi")

        # Crea colonna per marcare le anomalie
        df_result['is_anomaly'] = 0

        # Calcola dimensione del blocco
        block_size = max(int(n_points * block_percentage / 100), self.window_mean * 3)
        n_blocks = int(np.ceil(n_points / block_size))

        print(f"  - Numero di blocchi: {n_blocks}")
        print(f"  - Dimensione blocco: {block_size}")

        # Track dei punti già anomali
        anomalous_mask = np.zeros(n_points, dtype=bool)
        total_anomalous = 0

        # Statistiche delta
        delta_values = []

        # Itera sui blocchi
        for block_idx in range(n_blocks):
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, n_points)
            block_length = block_end - block_start

            target_anomalous_in_block = int(block_length * self.anomaly_percentage / 100)

            if block_idx == n_blocks - 1:
                remaining = n_anomalous_points - total_anomalous
                target_anomalous_in_block = max(target_anomalous_in_block, remaining)

            target_anomalous_in_block = min(target_anomalous_in_block,
                                            n_anomalous_points - total_anomalous)

            if target_anomalous_in_block <= 0:
                continue

            print(f"\n  Blocco {block_idx + 1}/{n_blocks} [{block_start}:{block_end}]")
            print(f"    Target anomalie: {target_anomalous_in_block}")

            block_anomalous = 0
            max_attempts_per_block = 100
            attempts = 0

            while block_anomalous < target_anomalous_in_block and attempts < max_attempts_per_block:
                attempts += 1

                # Genera lunghezza della finestra
                window_length = self._generate_skewed_window_length(fixed_mean=True)
                window_length = min(window_length, target_anomalous_in_block - block_anomalous)
                window_length = min(window_length, block_length)

                if window_length < 5:
                    continue

                max_start_in_block = block_end - window_length
                if max_start_in_block <= block_start:
                    break

                # Cerca una posizione libera nel blocco
                found_position = False
                for _ in range(50):
                    start_idx = np.random.randint(block_start, max_start_in_block + 1)
                    end_idx = start_idx + window_length

                    if end_idx <= block_end and not anomalous_mask[start_idx:end_idx].any():
                        found_position = True
                        break

                if not found_position:
                    continue

                # Seleziona il numero di canali da rendere anomali
                num_channels_to_modify = self._select_num_channels(n_channels)

                # Seleziona casualmente quali canali modificare
                channel_indices = list(range(n_channels))
                channels_to_modify = np.random.choice(
                    channel_indices,
                    size=num_channels_to_modify,
                    replace=False
                )

                # =============================================================
                # PIPELINE WOMBAT
                # =============================================================

                # 1. Estrai finestra normalizzata (mean=0, std=1)
                window_normalized, window_mean, window_std = self._extract_window(
                    df_result, start_idx, window_length, feature_columns
                )

                # 2. Genera delta da distribuzione normale
                delta = self._generate_delta(fixed_delta=True)
                delta_values.append(delta)

                # 3. Applica anomalia con delta
                window_anomalous = self._inject_window_anomaly(
                    window_normalized,
                    channels_to_modify,
                    delta,
                    anomaly_type='random'
                )

                # 4. De-normalizza finestra
                window_denormalized = self._denormalize_window(
                    window_anomalous,
                    window_mean,
                    window_std
                )

                # 5. Reinserisci nel dataframe
                for i, col in enumerate(feature_columns):
                    col_idx = df_result.columns.get_loc(col)
                    df_result.iloc[start_idx:end_idx, col_idx] = window_denormalized[:, i]

                # Marca come anomalo
                anomalous_mask[start_idx:end_idx] = True
                anomaly_col_idx = df_result.columns.get_loc('is_anomaly')
                df_result.iloc[start_idx:end_idx, anomaly_col_idx] = 1
                block_anomalous += window_length
                total_anomalous += window_length

            print(f"    Iniettate: {block_anomalous} (tentativi: {attempts})")

        actual_percentage = (total_anomalous / n_points) * 100
        print(f"\n{'=' * 70}")
        print(f"Iniezione completata:")
        print(f"  - Punti anomali iniettati: {total_anomalous}/{n_anomalous_points} ({actual_percentage:.2f}%)")
        print(f"  - Efficienza: {(total_anomalous / n_anomalous_points) * 100:.1f}%")

        if delta_values:
            print(f"\nStatistiche Delta:")
            print(f"  - Media: {np.mean(delta_values):.4f}")
            print(f"  - Std: {np.std(delta_values):.4f}")
            print(f"  - Min: {np.min(delta_values):.4f}")
            print(f"  - Max: {np.max(delta_values):.4f}")

        # Salva il dataframe
        if output_filename is not None:
            dir_path = os.path.dirname(output_filename)
            base_name = os.path.basename(output_filename)
            base_name = os.path.splitext(base_name)[0]

            if base_name.endswith('_with_anomalies'):
                base_name = base_name[:-15]

            output_filename = os.path.join(dir_path, f"{base_name}_with_anomalies.csv")
            df_result.to_csv(output_filename, index=False)
            print(f"\nDataframe salvato: {output_filename}")

        print(f"\nStatistiche anomalie:")
        print(f"  - Punti totali: {len(df_result)}")
        print(f"  - Punti normali: {(df_result['is_anomaly'] == 0).sum()}")
        print(f"  - Punti anomali: {(df_result['is_anomaly'] == 1).sum()}")

        return df_result


def main(args):
    """
    Main function con standardizzazione e pipeline WOMBAT
    """
    cfg = OmegaConf.load(args.conf_file)
    data_path = cfg.dataset.data_path

    # STEP 1: CARICAMENTO
    print(f"\n{'=' * 70}")
    print("STEP 1: CARICAMENTO DATI ORIGINALI")
    print('=' * 70)
    print(f"\nCaricamento file: {data_path}")

    try:
        df_original = load_data(cfg)
        print(f"✓ Caricato: {df_original.shape[0]} righe, {df_original.shape[1]} colonne")
    except Exception as e:
        print(f"✗ Errore: {e}")
        sys.exit(1)

    df_backup = df_original.copy()

    # Parametri dalla configurazione
    anomaly_percentage = cfg.dataset.anomaly_percentage
    window_mean = cfg.dataset.window_mean
    window_skewness = cfg.dataset.window_skewness
    min_channels = cfg.dataset.min_channels
    max_channels = cfg.dataset.max_channels
    channel_prob_decay = cfg.dataset.channel_prob_decay
    random_seed = cfg.dataset.random_seed
    features = cfg.dataset.feats
    block_percentage = cfg.dataset.block_percentage

    # Parametri delta
    delta_mean = cfg.dataset.delta_mean
    delta_std = cfg.dataset.delta_std

    print(f"\nConfigurazione:")
    print(f"  - Percentuale anomalie: {anomaly_percentage}%")
    print(f"  - Lunghezza media finestre: {window_mean}")
    print(f"  - Skewness: {window_skewness}")
    print(f"  - Canali: {min_channels} - {max_channels if max_channels else 'tutti'}")
    print(f"  - Decay probabilità: {channel_prob_decay}")
    print(f"  - Blocco: {block_percentage}%")
    print(f"  - Delta: μ={delta_mean}, σ={delta_std}")
    if random_seed:
        print(f"  - Random seed: {random_seed}")

    # STEP 2: STANDARDIZZAZIONE
    print(f"\n{'=' * 70}")
    print("STEP 2: STANDARDIZZAZIONE DATI")
    print('=' * 70)

    handler = StandardizationHandler()
    df_standardized = handler.fit_transform(df_original, feature_columns=features)

    print(f"\nVerifica standardizzazione (prime 3 features):")
    for col in features[:min(3, len(features))]:
        mean_std = df_standardized[col].mean()
        std_std = df_standardized[col].std()
        print(f"  - {col}: μ={mean_std:.6f}, σ={std_std:.6f}")

    # STEP 3: INIEZIONE ANOMALIE (WOMBAT pipeline)
    print(f"\n{'=' * 70}")
    print("STEP 3: INIEZIONE ANOMALIE (pipeline WOMBAT)")
    print('=' * 70)

    injector = TimeSeriesAnomalyInjector(
        anomaly_percentage=anomaly_percentage,
        window_mean=window_mean,
        window_skewness=window_skewness,
        min_channels=min_channels,
        max_channels=max_channels,
        channel_prob_decay=channel_prob_decay,
        delta_mean=delta_mean,
        delta_std=delta_std,
        random_seed=random_seed,
        anomaly_registry=ANOMALIES_REGISTRY
    )

    df_standardized_with_anomalies = injector.inject_anomalies(
        df=df_standardized,
        feature_columns=features,
        output_filename=None,
        block_percentage=block_percentage
    )

    anomaly_mask = df_standardized_with_anomalies['is_anomaly'].values.astype(bool)

    # STEP 4: DE-STANDARDIZZAZIONE
    print(f"\n{'=' * 70}")
    print("STEP 4: DE-STANDARDIZZAZIONE")
    print('=' * 70)

    df_destandardized = handler.inverse_transform(df_standardized_with_anomalies)

    # STEP 5: PRESERVAZIONE VALORI ORIGINALI
    print(f"\n{'=' * 70}")
    print("STEP 5: PRESERVAZIONE VALORI ORIGINALI (punti NON anomali)")
    print('=' * 70)

    for col in features:
        df_destandardized.loc[~anomaly_mask, col] = df_backup.loc[~anomaly_mask, col]

    print(f"\n✓ Valori originali preservati per i punti NON anomali")

    print(f"\nVerifica preservazione (prime 3 features):")
    for col in features[:min(3, len(features))]:
        non_anomalous_identical = np.array_equal(
            df_destandardized.loc[~anomaly_mask, col].values,
            df_backup.loc[~anomaly_mask, col].values
        )
        if non_anomalous_identical:
            print(f"  ✓ {col}: Valori NON anomali IDENTICI all'originale")
        else:
            max_diff = np.max(np.abs(
                df_destandardized.loc[~anomaly_mask, col].values -
                df_backup.loc[~anomaly_mask, col].values
            ))
            print(f"  ⚠ {col}: Differenza massima = {max_diff:.2e}")

    # STEP 6: SALVATAGGIO
    print(f"\n{'=' * 70}")
    print("STEP 6: SALVATAGGIO FILE FINALE")
    print('=' * 70)

    dir_path = os.path.dirname(data_path)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    output_filename = os.path.join(dir_path, f"{base_name}_with_anomalies.csv")

    df_destandardized.to_csv(output_filename, index=False)
    print(f"\n✓ File salvato: {output_filename}")

    # STEP 7: STATISTICHE
    print(f"\n{'=' * 70}")
    print("STATISTICHE FINALI")
    print('=' * 70)

    n_total = len(df_destandardized)
    n_anomalies = anomaly_mask.sum()
    n_preserved = n_total - n_anomalies

    print(f"\nDataset:")
    print(f"  - Punti totali: {n_total}")
    print(f"  - Punti anomali: {n_anomalies} ({n_anomalies / n_total * 100:.2f}%)")
    print(f"  - Punti preservati: {n_preserved} ({n_preserved / n_total * 100:.2f}%)")

    print(f"\nStatistiche per feature (prime 3):")
    for col in features[:min(3, len(features))]:
        print(f"\n  {col}:")
        print(f"    Originale: μ={df_backup[col].mean():.4f}, σ={df_backup[col].std():.4f}")
        print(f"    Con anomalie: μ={df_destandardized[col].mean():.4f}, σ={df_destandardized[col].std():.4f}")

        if anomaly_mask.any():
            anomalies_original = df_backup.loc[anomaly_mask, col]
            anomalies_modified = df_destandardized.loc[anomaly_mask, col]
            mean_change = (anomalies_modified - anomalies_original).abs().mean()
            print(f"    Modifica media su anomalie: {mean_change:.4f}")

    print(f"\n{'=' * 70}")
    print("✓ PROCESSO COMPLETATO CON SUCCESSO")
    print('=' * 70)
    print(f"\nFile finale: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Anomalies injection tool with WOMBAT pipeline and delta parameter',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--conf_file', '-c', type=str, default='./dataset_configuration/fiorire_1.yaml',
        help='Dataset configuration file path')

    parser.add_argument(
        '--plot', action='store_true',
        help='Crea visualizzazioni delle anomalie')

    args = parser.parse_args()
    main(args)