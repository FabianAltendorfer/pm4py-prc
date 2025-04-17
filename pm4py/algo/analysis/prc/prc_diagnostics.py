import pandas as pd
import numpy as np
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.petri_net.utils import petri_utils
from gplearn.genetic import SymbolicRegressor
from typing import Optional, Dict, Any, Union
from collections import defaultdict
from datetime import datetime

def prc_bottleneckdetection_sr(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Erkennt Engpässe in einem Produktionsprozess mithilfe eines erweiterten Token-Based Replay (TBR)-Ansatzes,
    der eine Production Resource Constrained (PRC)-Analyse und Symbolische Regression (SR) integriert.
    Die SR findet dynamisch eine Formel für Engpässe, basierend auf Kapazitäten, Zeitdifferenzen und optionalen Faktoren
    wie Maschinenmeldungen und Speicherständen.

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
        Parameter für den Algorithmus, einschließlich activity_key, timestamp_key, machine_alert_key, sr_weights, etc.

    Rückgabe
    --------
    pd.DataFrame
        DataFrame mit Engpassdiagnosen, einschließlich Kapazitäten, Zeitdifferenzen und SR-Scores.
    """
    if parameters is None:
        parameters = {}
    
    activity_key = parameters.get('activity_key', 'concept:name')
    timestamp_key = parameters.get('timestamp_key', 'time:timestamp')
    machine_alert_key = parameters.get('machine_alert_key', None)
    sr_weights = parameters.get('sr_weights', {})
    
    # Step 1: Data Preprocessing
    df_events = _preprocess_log(log, activity_key, timestamp_key, machine_alert_key)
    
    # Step 2: Extend Token-Based Replay to capture simultaneous token counts and metrics
    tbr_results, place_token_counts, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts = _extended_token_replay(
        log, net, initial_marking, final_marking, parameters, df_events
    )
    
    # Step 3: Identify internal and external connections
    internal_places, external_places = _identify_connections(net, df_events[activity_key].unique())
    
    # Step 4: Analyze capacities, frequencies, time differences, alerts, and storage levels
    capacity_info = _analyze_capacities(
        place_token_counts, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts,
        internal_places, external_places
    )
    
    # Step 5: Prepare data for Symbolic Regression
    sr_data = _prepare_sr_data(tbr_results, capacity_info, df_events)
    
    # Step 6: Apply Symbolic Regression to compute bottleneck scores
    bottleneck_scores = _compute_sr_scores(sr_data, sr_weights)
    
    # Step 7: Compile results
    results_df = pd.DataFrame({
        'place': list(capacity_info.keys()),
        'max_capacity': [info['max_tokens'] for info in capacity_info.values()],
        'bottleneck_frequency': [info['frequency'] for info in capacity_info.values()],
        'avg_time_diff': [info['avg_time_diff'] for info in capacity_info.values()],
        'machine_alert_count': [info['alert_count'] for info in capacity_info.values()],
        'storage_level_ratio': [info['storage_level_ratio'] for info in capacity_info.values()],
        'sr_bottleneck_score': bottleneck_scores
    })
    
    return results_df

def _preprocess_log(log: EventLog, activity_key: str, timestamp_key: str, machine_alert_key: Optional[str]) -> pd.DataFrame:
    """Vorverarbeitung des Event-Logs, inklusive Timestamp-Validierung und Ausreißerentfernung."""
    events = []
    for trace in log:
        case_id = trace.attributes.get('concept:name', 'unknown')
        for event in trace:
            event_data = {
                'case_id': case_id,
                'activity': event[activity_key],
                'timestamp': pd.to_datetime(event[timestamp_key])
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
    
    return df[['case_id', 'activity', 'timestamp', 'time_diff', 'next_activity', 'machine_alert']]

def _extended_token_replay(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Dict[str, Any], df_events: pd.DataFrame) -> tuple:
    """Erweitert TBR, um gleichzeitige Tokenanzahlen, Zeitdifferenzen, Maschinenmeldungen, Speicherstände und Frequenzen zu verfolgen."""
    tbr_params = parameters.copy()
    tbr_params['enable_pltr_fitness'] = True
    tbr_params['case_id_key'] = 'concept:name'
    
    # Dictionaries zur Verfolgung von Metriken
    place_token_timeline = defaultdict(list)
    place_time_diffs = defaultdict(list)
    place_alerts = defaultdict(list)
    place_storage_levels = defaultdict(list)
    place_frequency_counts = defaultdict(int)
    
    # TBR für jede Spur ausführen
    aligned_traces, place_fitness, _, _ = token_replay.apply_log(log, net, initial_marking, final_marking, parameters=tbr_params)
    
    # Aggregiere Metriken über alle Spuren hinweg
    for trace_idx, trace_result in enumerate(aligned_traces):
        case_id = trace_result.get('case_id', f'case_{trace_idx}')
        trace_events = df_events[df_events['case_id'] == case_id][['activity', 'timestamp', 'machine_alert']]
        
        # Verfolge Markierungen und Zeitstempel
        for marking in trace_result['reached_marking'].items():
            place, tokens = marking
            timestamp = trace_events.iloc[min(trace_idx, len(trace_events)-1)]['timestamp']
            place_token_timeline[place].append((timestamp, tokens))
        
        # Zeitdifferenzen, Maschinenmeldungen und Speicherstände pro Platz
        for _, event in trace_events.iterrows():
            activity = event['activity']
            for trans in net.transitions:
                if trans.label == activity:
                    for arc in trans.out_arcs:
                        place = arc.target
                        time_diff = df_events[
                            (df_events['case_id'] == case_id) & (df_events['activity'] == activity)
                        ]['time_diff'].mean()
                        if not np.isnan(time_diff):
                            place_time_diffs[place].append(time_diff)
                        alert = event['machine_alert']
                        place_alerts[place].append(alert)
                        if '0505' in activity:
                            tokens = place_token_counts.get(place, 1)
                            place_storage_levels[place].append(tokens / max(1, tokens))
    
    # Berechne maximale gleichzeitige Tokenanzahl und Frequenz
    place_token_counts = {}
    for place, timeline in place_token_timeline.items():
        timeline.sort(key=lambda x: x[0])
        max_tokens = 0
        current_tokens = 0
        for i, (timestamp, tokens) in enumerate(timeline):
            current_tokens += tokens
            max_tokens = max(max_tokens, current_tokens)
            # Zähle Frequenz, wenn nahe der maximalen Kapazität
            if tokens >= max_tokens * 0.9:
                place_frequency_counts[place] += 1
            for j in range(i + 1, len(timeline)):
                if timeline[j][0] > timestamp:
                    current_tokens -= tokens
                    break
        place_token_counts[place] = max_tokens
    
    return aligned_traces, place_token_counts, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts

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

def _analyze_capacities(place_token_counts: Dict[str, int], place_time_diffs: Dict[str, list], place_alerts: Dict[str, list], place_storage_levels: Dict[str, list], place_frequency_counts: Dict[str, int], internal_places: set, external_places: set) -> Dict[str, Dict[str, Any]]:
    """Analysiert Kapazitäten, Häufigkeiten, Zeitdifferenzen, Maschinenmeldungen und Speicherstände."""
    capacity_info = {}
    total_traces = len(place_frequency_counts)
    for place, max_tokens in place_token_counts.items():
        place_type = 'internal' if place in internal_places else 'external' if place in external_places else 'other'
        frequency = place_frequency_counts[place] / max(1, total_traces) if total_traces > 0 else 0.0
        avg_time_diff = np.mean(place_time_diffs.get(place, [0])) if place_time_diffs.get(place) else 0
        time_diff_variance = np.var(place_time_diffs.get(place, [0])) if place_time_diffs.get(place) else 0
        alert_count = sum(place_alerts.get(place, [0]))
        storage_level_ratio = np.mean(place_storage_levels.get(place, [1.0])) if place_storage_levels.get(place) else 1.0
        capacity_info[place] = {
            'max_tokens': max_tokens,
            'frequency': frequency,
            'avg_time_diff': avg_time_diff,
            'time_diff_variance': time_diff_variance,
            'alert_count': alert_count,
            'storage_level_ratio': storage_level_ratio,
            'type': place_type
        }
    
    return capacity_info

def _prepare_sr_data(tbr_results: list, capacity_info: Dict[str, Dict[str, Any]], df_events: pd.DataFrame) -> pd.DataFrame:
    """Bereitet Features für Symbolische Regression vor."""
    sr_data = []
    for trace_result in tbr_results:
        case_id = trace_result.get('case_id', 'unknown')
        fitness = trace_result['trace_fitness']
        missing_tokens = trace_result['missing_tokens']
        for place, info in capacity_info.items():
            sr_data.append({
                'case_id': case_id,
                'place': place,
                'fitness': fitness,
                'missing_tokens': missing_tokens,
                'max_tokens': info['max_tokens'],
                'frequency': info['frequency'],
                'avg_time_diff': info['avg_time_diff'],
                'time_diff_variance': info['time_diff_variance'],
                'alert_count': info['alert_count'],
                'storage_level_ratio': info['storage_level_ratio'],
                'global_avg_time_diff': df_events['time_diff'].mean()
            })
    return pd.DataFrame(sr_data)

def _compute_sr_scores(sr_data: pd.DataFrame, sr_weights: Dict[str, float]) -> np.ndarray:
    """Berechnet Engpass-Scores mithilfe Symbolischer Regression mit benutzerdefinierter Fitness-Funktion."""
    X = sr_data[[
        'fitness', 'missing_tokens', 'max_tokens', 'frequency', 'avg_time_diff',
        'time_diff_variance', 'alert_count', 'storage_level_ratio', 'global_avg_time_diff'
    ]].values
    
    # Zielvariable: Fokus auf Kapazitätsbeschränkung
    y = sr_data['max_tokens'] * sr_data['frequency'] * (1 + sr_data['avg_time_diff']) / (sr_data['storage_level_ratio'] + 0.1)
    
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
    """Protokolliert Datenbereinigungsschritte."""
    print(f"Datenbereinigung: {step_name}, Initial: {initial_count}, Entfernt: {removed_count}, Verbleibend: {remaining_count}, Details: {details}")

# Helper zur Sicherstellung der case_id in TBR-Ergebnissen
token_replay.apply_log.__defaults__ = (*token_replay.apply_log.__defaults__, 'case_id_key', 'concept:name')