import pandas as pd
import numpy as np
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.tokenreplay.variants import tbr_prc
from pm4py.objects.petri_net.utils import petri_utils
from gplearn.genetic import SymbolicRegressor
from typing import Optional, Dict, Any, Union, List, Tuple
from collections import defaultdict
from datetime import datetime
import os

# Globale Liste für Datenbereinigungsstatistiken
data_cleaning_stats = []

def prc_bottleneckdetection_sr(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Erkennt Engpässe in einem Produktionsprozess mithilfe eines erweiterten Token-Based Replay (TBR)-Ansatzes,
    der eine Production Resource Constrained (PRC)-Analyse und Symbolische Regression (SR) integriert (Masterarbeit Fabian Altendorfer).
    Die SR findet dynamisch eine Formel für Engpässe, basierend auf Kapazitäten, Zeitdifferenzen und optionalen Faktoren
    wie Maschinenmeldungen und Speicherständen, und modelliert die relative Verzögerung von Abschnitten im Gesamtprozess.

    Parameter
    ----------
    log : EventLog
        Objektzentrierter Event-Log (OCEL), abgeflacht zu einem traditionellen Event-Log.
    net : PetriNet
        Petri-Netz, das den Produktionsprozess repräsentiert.
    initial_marking : Marking
        Anfangsmarkierung des Petri-Netzes.
    final_marking : Marking
        Endmarkierung des Petri-Netzes.
    parameters : Dict[str, Any], optional
        Parameter für den Algorithmus, einschließlich activity_key, timestamp_key, machine_alert_key, sr_weights, cleaning_stats_file.

    Rückgabe
    --------
    pd.DataFrame
        DataFrame mit Engpassdiagnosen, einschließlich Kapazitäten, Zeitdifferenzen, maximaler Verzögerung und SR-Scores.
    """
    global data_cleaning_stats
    data_cleaning_stats = []
    
    if parameters is None:
        parameters = {}
    
    activity_key = parameters.get('activity_key', 'concept:name')
    timestamp_key = parameters.get('timestamp_key', 'time:timestamp')
    machine_alert_key = parameters.get('machine_alert_key', None)
    sr_weights = parameters.get('sr_weights', {})
    cleaning_stats_file = parameters.get('cleaning_stats_file', '../Ergebnisse/data_cleaning_statistics.csv')
    
    # Step 1: Datenvorverarbeitung
    df_events, trace_total_times = _preprocess_log(log, activity_key, timestamp_key, machine_alert_key)
    
    # Step 2: PRC-Erweiterung des Token Based Replays mit neuer TBR-Variante
    tbr_results, place_fitness, transition_fitness, notexisting_activities, place_token_timeline, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts = tbr_prc.apply_log(
        log, net, initial_marking, final_marking, enable_pltr_fitness=True, parameters=parameters
    )
    
    # Step 3: Identifizierung interner und externer Verbindungen
    internal_places, external_places = _identify_connections(net, df_events[activity_key].unique())
    
    # Step 4: Analysieren der Einflussfaktoren
    capacity_info = _analyze_capacities(
        place_token_timeline, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts,
        internal_places, external_places
    )
    
    # Step 5: Datenvorbereitung für die symbolische Regression
    sr_data = _prepare_sr_data(tbr_results, capacity_info, df_events, trace_total_times)
    
    # Step 6: Berechnung von Engpass Scores durch Symbolic Regression
    bottleneck_scores = _compute_sr_scores(sr_data, sr_weights)
    
    # Step 7: Zusammenfügen der Ergebnisse
    results_df = pd.DataFrame({
        'place': list(capacity_info.keys()),
        'max_capacity': [info['max_tokens'] for info in capacity_info.values()],
        'bottleneck_frequency': [info['frequency'] for info in capacity_info.values()],
        'avg_time_diff': [info['avg_time_diff'] for info in capacity_info.values()],
        'max_time': [info['max_time'] for info in capacity_info.values()],
        'machine_alert_count': [info['alert_count'] for info in capacity_info.values()],
        'storage_level_ratio': [info['storage_level_ratio'] for info in capacity_info.values()],
        'sr_bottleneck_score': bottleneck_scores
    })
    
    # Step 8: Speichere Datenbereinigungsstatistiken als CSV
    if data_cleaning_stats:
        stats_df = pd.DataFrame(data_cleaning_stats)
        stats_df.to_csv(cleaning_stats_file, index=False)
        if os.path.exists(cleaning_stats_file):
            print(f"Datenbereinigungsstatistiken erfolgreich gespeichert in: {cleaning_stats_file}")
        else:
            print(f"Fehler: Konnte Datenbereinigungsstatistiken nicht in {cleaning_stats_file} speichern")
    
    return results_df

def _preprocess_log(log: EventLog, activity_key: str, timestamp_key: str, machine_alert_key: Optional[str]) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Vorverarbeitung des Event-Logs, inklusive Timestamp-Validierung, Ausreißerentfernung und Berechnung der Gesamtverzögerung pro Spur."""
    events = []
    trace_total_times = {}
    for trace in log:
        case_id = trace.attributes.get('concept:name', 'unknown')
        timestamps = [pd.to_datetime(event[timestamp_key]) for event in trace]
        if timestamps:
            total_time = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            trace_total_times[case_id] = total_time
        for event in trace:
            event_data = {
                'case_id': case_id,
                'activity': event[activity_key],
                'timestamp': pd.to_datetime(event[timestamp_key]),
                'time_diff': 0.0  # Wird von tbr_prc berechnet
            }
            event_data['machine_alert'] = event.get(machine_alert_key, 0) if machine_alert_key else 0
            events.append(event_data)
    df = pd.DataFrame(events).sort_values(['case_id', 'timestamp'])
    
    # Timestamp-Validierung
    initial_count = len(df)
    df = df.dropna(subset=['timestamp'])
    df = df[df['timestamp'] <= pd.Timestamp('2030-01-01', tz='UTC')]
    
    # Berechne Zeitdifferenzen und entferne Ausreißer
    df['next_timestamp'] = df.groupby('case_id')['timestamp'].shift(-1)
    df['next_activity'] = df.groupby('case_id')['activity'].shift(-1)
    df['time_diff'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds() / 3600
    df = df[df['time_diff'].notna()]
    df = df[df['time_diff'] <= 8760]
    
    # Entferne Zeiten 30% über dem Median pro Verbindung
    median_times = df.groupby(['activity', 'next_activity'])['time_diff'].median().reset_index(name='median_time')
    df = df.merge(median_times, on=['activity', 'next_activity'], how='left')
    df = df[df['time_diff'] <= df['median_time'] * 1.3]
    
    # Protokolliere Datenbereinigung
    removed_count = initial_count - len(df)
    log_cleaning_step(
        step_name="Datenbereinigung",
        initial_count=initial_count,
        removed_count=removed_count,
        remaining_count=len(df),
        details="Entfernen von NaT, ungültigen Timestamps und Zeitdifferenzen > 8760 Stunden oder 30% über Median."
    )
    
    return df[['case_id', 'activity', 'timestamp', 'time_diff', 'next_activity', 'machine_alert']], trace_total_times

def _identify_connections(net: PetriNet, activities: list) -> tuple:
    """Identifiziert interne und externe Plätze basierend auf Transitionen."""
    internal_places = set()
    external_places = set()
    
    for trans in net.transitions:
        if trans.label:
            activity = trans.label
            if activity.endswith('_1') or activity.endswith('_6'):
                for arc in trans.in_arcs:
                    internal_places.add(arc.source)
                for arc in trans.out_arcs:
                    internal_places.add(arc.target)
            elif activity.endswith('_f') or '0505' in activity:
                for arc in trans.in_arcs:
                    external_places.add(arc.source)
                for arc in trans.out_arcs:
                    external_places.add(arc.target)
    
    return internal_places, external_places

def _analyze_capacities(place_token_timeline: Dict[PetriNet.Place, List[Tuple[pd.Timestamp, int]]], place_time_diffs: Dict[PetriNet.Place, List[float]], place_alerts: Dict[PetriNet.Place, List[float]], place_storage_levels: Dict[PetriNet.Place, List[float]], place_frequency_counts: Dict[PetriNet.Place, int], internal_places: set, external_places: set) -> Dict[PetriNet.Place, Dict[str, Any]]:
    """Analysiert Kapazitäten, Häufigkeiten, Zeitdifferenzen, Maschinenmeldungen und Speicherstände."""
    capacity_info = {}
    total_traces = sum(place_frequency_counts.values()) / len(place_frequency_counts) if place_frequency_counts else 1
    for place in place_token_timeline:
        place_type = 'internal' if place in internal_places else 'external' if place in external_places else 'other'
        tokens = [t[1] for t in place_token_timeline[place]]
        max_tokens = max(tokens, default=0)
        frequency = place_frequency_counts.get(place, 0) / max(1, total_traces)
        avg_time_diff = np.mean(place_time_diffs.get(place, [0])) if place_time_diffs.get(place) else 0
        max_time = np.max(place_time_diffs.get(place, [0])) if place_time_diffs.get(place) else 0
        time_diff_variance = np.var(place_time_diffs.get(place, [0])) if place_time_diffs.get(place) else 0
        alert_count = sum(place_alerts.get(place, [0]))
        storage_level_ratio = np.mean(place_storage_levels.get(place, [1.0])) if place_storage_levels.get(place) else 1.0
        capacity_info[place] = {
            'max_tokens': max_tokens,
            'frequency': frequency,
            'avg_time_diff': avg_time_diff,
            'max_time': max_time,
            'time_diff_variance': time_diff_variance,
            'alert_count': alert_count,
            'storage_level_ratio': storage_level_ratio,
            'type': place_type
        }
    
    return capacity_info

def _prepare_sr_data(tbr_results: list, capacity_info: Dict[PetriNet.Place, Dict[str, Any]], df_events: pd.DataFrame, trace_total_times: Dict[str, float]) -> pd.DataFrame:
    """Bereitet Features für Symbolische Regression vor, einschließlich Gesamtverzögerung pro Spur."""
    sr_data = []
    for trace_result in tbr_results:
        case_id = trace_result.get('case_id', 'unknown')
        fitness = trace_result['trace_fitness']
        missing_tokens = trace_result['missing_tokens']
        total_time = trace_total_times.get(case_id, 0.0)
        for place, info in capacity_info.items():
            sr_data.append({
                'case_id': case_id,
                'place': place,
                'fitness': fitness,
                'missing_tokens': missing_tokens,
                'max_tokens': info['max_tokens'],
                'frequency': info['frequency'],
                'avg_time_diff': info['avg_time_diff'],
                'max_time': info['max_time'],
                'time_diff_variance': info['time_diff_variance'],
                'alert_count': info['alert_count'],
                'storage_level_ratio': info['storage_level_ratio'],
                'global_avg_time_diff': df_events['time_diff'].mean(),
                'total_trace_time': total_time
            })
    return pd.DataFrame(sr_data)

def _compute_sr_scores(sr_data: pd.DataFrame, sr_weights: Dict[str, float]) -> np.ndarray:
    """Berechnet Engpass-Scores mithilfe Symbolischer Regression mit benutzerdefinierter Fitness-Funktion."""
    X = sr_data[[
        'fitness', 'missing_tokens', 'max_tokens', 'frequency', 'avg_time_diff',
        'max_time', 'time_diff_variance', 'alert_count', 'storage_level_ratio',
        'global_avg_time_diff', 'total_trace_time'
    ]].values
    
    # Zielvariable: Fokus auf Gesamtverzögerung des Prozesses
    y = sr_data['total_trace_time']
    
    # Benutzerdefinierte Fitness-Funktion mit Gewichtung
    def custom_fitness(y_true, y_pred, sample_weight):
        error = np.abs(y_true - y_pred)
        weighted_error = error
        for feature, weight in sr_weights.items():
            if feature in sr_data.columns:
                weighted_error *= (1 + sr_data[feature] * weight)
        return np.mean(weighted_error)
    
    sr = SymbolicRegressor(
        population_size=1000,
        generations=20,
        random_state=42,
        metric=custom_fitness
    )
    sr.fit(X, y)
    scores = sr.predict(X)
    return scores

def log_cleaning_step(step_name: str, initial_count: int, removed_count: int, remaining_count: int, details: str = ""):
    """Protokolliert Datenbereinigungsschritte und speichert sie in einer globalen Liste."""
    global data_cleaning_stats
    print(f"Datenbereinigung: {step_name}, Initial: {initial_count}, Entfernt: {removed_count}, Verbleibend: {remaining_count}, Details: {details}")
    data_cleaning_stats.append({
        'step': step_name,
        'initial_count': initial_count,
        'removed_count': removed_count,
        'remaining_count': remaining_count,
        'details': details
    })