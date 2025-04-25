from pm4py.util import xes_constants as xes_util
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils.petri_utils import get_places_shortest_path_by_hidden, get_s_components_from_petri
from copy import copy
from enum import Enum
from pm4py.util import exec_utils, constants
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from collections import Counter
from datetime import datetime

class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY

class TechnicalParameters(Enum):
    MAX_DEF_THR_EX_TIME = 10

def add_missing_tokens(t, marking):
    missing = 0
    tokens_added = {}
    for a in t.in_arcs:
        if marking[a.source] < a.weight:
            missing = missing + (a.weight - marking[a.source])
            tokens_added[a.source] = a.weight - marking[a.source]
            marking[a.source] = marking[a.source] + a.weight
    return [missing, tokens_added]

def get_consumed_tokens(t):
    consumed = 0
    consumed_map = {}
    for a in t.in_arcs:
        consumed = consumed + a.weight
        consumed_map[a.source] = a.weight
    return consumed, consumed_map

def get_produced_tokens(t):
    produced = 0
    produced_map = {}
    for a in t.out_arcs:
        produced = produced + a.weight
        produced_map[a.target] = a.weight
    return produced, produced_map

def compute_place_capacities(net, event_type_capacities):
    place_capacities = {}
    for place in net.places:
        incoming_transitions = [t for t in net.transitions if any(arc.source == place and arc.target == t for arc in t.in_arcs)]
        max_capacity = 0
        for trans in incoming_transitions:
            if trans.label and trans.label in event_type_capacities:
                max_capacity = max(max_capacity, event_type_capacities[trans.label])
        if max_capacity > 0:
            place_capacities[place] = max_capacity
    return place_capacities

def apply_trace(trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness,
                transition_fitness, notexisting_activities_in_model,
                places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key="concept:name",
                try_to_reach_final_marking_through_hidden=True, stop_immediately_unfit=False,
                walk_through_hidden_trans=True, post_fix_caching=None,
                marking_to_activity_caching=None, is_reduction=False,
                thread_maximum_ex_time=10, enable_postfix_cache=False, enable_marktoact_cache=False,
                cleaning_token_flood=False, s_components=None, trace_occurrences=1,
                consider_activities_not_in_model_in_fitness=False, events_by_timestamp=None,
                global_place_counts=None, timestamp_key="time:timestamp", place_capacities=None):
    sorted_events = sorted(trace, key=lambda x: x.get(timestamp_key, 0))

    act_trans = []
    transitions_with_problems = []
    vis_mark = []
    activating_transition_index = {}
    activating_transition_interval = []
    marking = copy(initial_marking)
    vis_mark.append(marking)
    missing = 0
    consumed = 0
    produced = sum(initial_marking[p] for p in initial_marking)
    current_event_map = {}
    current_remaining_map = {}
    capacity_exceeded_entries = []  # Liste für Einträge mit waiting_due_to_capacity und Zeitstempeln

    for i, event in enumerate(sorted_events):
        prev_len_activated_transitions = len(act_trans)
        if event[activity_key] in trans_map:
            t = trans_map[event[activity_key]]
            current_event_map.update(event)

            if not semantics.is_enabled(t, net, marking):
                if stop_immediately_unfit:
                    missing += 1
                    break
                [m, tokens_added] = add_missing_tokens(t, marking)
                missing += m
                if enable_pltr_fitness:
                    for place in tokens_added.keys():
                        if place in place_fitness:
                            place_fitness[place]["underfed_traces"].add(trace)
                        place_fitness[place]["m"] += tokens_added[place]
                    if trace not in transition_fitness[t]["underfed_traces"]:
                        transition_fitness[t]["underfed_traces"][trace] = []
                    transition_fitness[t]["underfed_traces"][trace].append(current_event_map)
            else:
                if enable_pltr_fitness:
                    if trace not in transition_fitness[t]["fit_traces"]:
                        transition_fitness[t]["fit_traces"][trace] = []
                    transition_fitness[t]["fit_traces"][trace].append(current_event_map)

            c, cmap = get_consumed_tokens(t)
            p, pmap = get_produced_tokens(t)
            consumed += c
            produced += p

            if semantics.is_enabled(t, net, marking):
                marking = semantics.execute(t, net, marking)
                act_trans.append(t)
                vis_mark.append(marking)

                # Prüfe Kapazitätsüberschreitung nach der Ausführung der Transition
                for place in marking:
                    if place in place_capacities and marking[place] >= place_capacities[place]:
                        # Erstelle einen Eintrag für die Überschreitung
                        entry = {
                            "waiting_due_to_capacity": True,
                            "timestamp": event[timestamp_key]
                        }
                        capacity_exceeded_entries.append(entry)
                        # Speichere detaillierte Informationen im Event
                        event["capacity_exceeded"] = {
                            "place": place.name,
                            "tokens": marking[place],
                            "capacity": place_capacities[place],
                            "timestamp": event[timestamp_key]
                        }

            if enable_pltr_fitness:
                for pl in cmap:
                    if pl in place_fitness:
                        place_fitness[pl]["c"] += cmap[pl] * trace_occurrences
                for pl in pmap:
                    if pl in place_fitness:
                        place_fitness[pl]["p"] += pmap[pl] * trace_occurrences

        else:
            if event[activity_key] not in notexisting_activities_in_model:
                notexisting_activities_in_model[event[activity_key]] = {}
            notexisting_activities_in_model[event[activity_key]][trace] = current_event_map

        trace_activities = [e[activity_key] for e in sorted_events[i:]]
        if len(trace_activities) < 20:
            activating_transition_index[str(trace_activities)] = {"index": len(act_trans), "marking": hash(marking)}
        if i > 0:
            activating_transition_interval.append([event[activity_key], prev_len_activated_transitions, len(act_trans), sorted_events[i-1][activity_key]])
        else:
            activating_transition_interval.append([event[activity_key], prev_len_activated_transitions, len(act_trans), ""])

    marking_before_cleaning = copy(marking)
    diff_fin_mark_mark = Marking()
    for p in final_marking:
        diff = final_marking[p] - marking[p]
        if diff > 0:
            diff_fin_mark_mark[p] = diff

    remaining = 0
    for p in marking:
        if p in final_marking:
            marking[p] = max(0, marking[p] - final_marking[p])
            if enable_pltr_fitness and marking[p] > 0:
                if p in place_fitness:
                    place_fitness[p]["overfed_traces"].add(trace)
                place_fitness[p]["r"] += marking[p] * trace_occurrences
        elif enable_pltr_fitness:
            if p in place_fitness:
                place_fitness[p]["overfed_traces"].add(trace)
            place_fitness[p]["r"] += marking[p] * trace_occurrences
        remaining += marking[p]

    for p in current_remaining_map:
        if enable_pltr_fitness:
            if p in place_fitness:
                place_fitness[p]["overfed_traces"].add(trace)
            place_fitness[p]["r"] += current_remaining_map[p] * trace_occurrences
        remaining += current_remaining_map[p]

    is_fit = (missing == 0) and (consider_remaining_in_fitness and remaining == 0)
    if consider_activities_not_in_model_in_fitness and notexisting_activities_in_model:
        is_fit = False

    for pl in final_marking:
        consumed += final_marking[pl]
    for pl in diff_fin_mark_mark:
        missing += diff_fin_mark_mark[pl]

    if enable_pltr_fitness:
        for pl in initial_marking:
            place_fitness[pl]["p"] += initial_marking[pl] * trace_occurrences
        for pl in final_marking:
            place_fitness[pl]["c"] += final_marking[pl] * trace_occurrences
        for pl in diff_fin_mark_mark:
            place_fitness[pl]["m"] += diff_fin_mark_mark[pl] * trace_occurrences

    trace_fitness = 0.5 * (1.0 - float(missing) / float(consumed)) + 0.5 * (1.0 - float(remaining) / float(produced)) if consumed > 0 and produced > 0 else 1.0

    # Speichere die Liste der Kapazitätsüberschreitungen im Trace (Case/Objekt)
    trace.attributes["capacity_exceeded_entries"] = capacity_exceeded_entries

    return [is_fit, trace_fitness, act_trans, transitions_with_problems, marking_before_cleaning,
            semantics.enabled_transitions(net, marking_before_cleaning), missing, consumed, remaining, produced, capacity_exceeded_entries]

class ApplyTraceTokenReplay:
    def __init__(self, trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness,
                 transition_fitness, notexisting_activities_in_model, places_shortest_path_by_hidden,
                 consider_remaining_in_fitness, activity_key="concept:name", reach_mark_through_hidden=True,
                 stop_immediately_unfit=False, walk_through_hidden_trans=True, post_fix_caching=None,
                 marking_to_activity_caching=None, is_reduction=False,
                 thread_maximum_ex_time=TechnicalParameters.MAX_DEF_THR_EX_TIME.value,
                 cleaning_token_flood=False, s_components=None, trace_occurrences=1,
                 consider_activities_not_in_model_in_fitness=False, events_by_timestamp=None,
                 global_place_counts=None, timestamp_key="time:timestamp", place_capacities=None):
        self.thread_is_alive = True
        self.trace = trace
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.trans_map = trans_map
        self.enable_pltr_fitness = enable_pltr_fitness
        self.place_fitness = place_fitness
        self.transition_fitness = transition_fitness
        self.notexisting_activities_in_model = notexisting_activities_in_model
        self.places_shortest_path_by_hidden = places_shortest_path_by_hidden
        self.consider_remaining_in_fitness = consider_remaining_in_fitness
        self.activity_key = activity_key
        self.reach_mark_through_hidden = reach_mark_through_hidden
        self.stop_immediately_unfit = stop_immediately_unfit
        self.walk_through_hidden_trans = walk_through_hidden_trans
        self.post_fix_caching = post_fix_caching
        self.marking_to_activity_caching = marking_to_activity_caching
        self.is_reduction = is_reduction
        self.thread_maximum_ex_time = thread_maximum_ex_time
        self.cleaning_token_flood = cleaning_token_flood
        self.s_components = s_components
        self.trace_occurrences = trace_occurrences
        self.consider_activities_not_in_model_in_fitness = consider_activities_not_in_model_in_fitness
        self.events_by_timestamp = events_by_timestamp
        self.global_place_counts = global_place_counts
        self.timestamp_key = timestamp_key
        self.place_capacities = place_capacities

    def run(self):
        self.t_fit, self.t_value, self.act_trans, self.trans_probl, self.reached_marking, self.enabled_trans_in_mark, self.missing, self.consumed, self.remaining, self.produced, self.capacity_exceeded_entries = \
            apply_trace(self.trace, self.net, self.initial_marking, self.final_marking, self.trans_map,
                        self.enable_pltr_fitness, self.place_fitness, self.transition_fitness,
                        self.notexisting_activities_in_model,
                        self.places_shortest_path_by_hidden, self.consider_remaining_in_fitness,
                        activity_key=self.activity_key,
                        try_to_reach_final_marking_through_hidden=self.reach_mark_through_hidden,
                        stop_immediately_unfit=self.stop_immediately_unfit,
                        walk_through_hidden_trans=self.walk_through_hidden_trans,
                        post_fix_caching=self.post_fix_caching,
                        marking_to_activity_caching=self.marking_to_activity_caching,
                        is_reduction=self.is_reduction,
                        thread_maximum_ex_time=self.thread_maximum_ex_time,
                        cleaning_token_flood=self.cleaning_token_flood,
                        s_components=self.s_components,
                        trace_occurrences=self.trace_occurrences,
                        consider_activities_not_in_model_in_fitness=self.consider_activities_not_in_model_in_fitness,
                        events_by_timestamp=self.events_by_timestamp,
                        global_place_counts=self.global_place_counts,
                        timestamp_key=self.timestamp_key,
                        place_capacities=self.place_capacities)
        self.thread_is_alive = False

def apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=False, consider_remaining_in_fitness=False,
              activity_key="concept:name", reach_mark_through_hidden=True, stop_immediately_unfit=False,
              walk_through_hidden_trans=True, places_shortest_path_by_hidden=None,
              is_reduction=False, thread_maximum_ex_time=10,
              cleaning_token_flood=False, disable_variants=False, return_object_names=True, show_progress_bar=True,
              consider_activities_not_in_model_in_fitness=False, case_id_key=constants.CASE_CONCEPT_NAME,
              timestamp_key="time:timestamp"):
    if places_shortest_path_by_hidden is None:
        places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(net, 50)

    place_fitness_per_trace = {}
    transition_fitness_per_trace = {}
    aligned_traces = []
    if enable_pltr_fitness:
        for place in net.places:
            place_fitness_per_trace[place] = {"underfed_traces": set(), "overfed_traces": set(), "m": 0, "r": 0, "c": 0, "p": 0}
        for transition in net.transitions:
            if transition.label:
                transition_fitness_per_trace[transition] = {"underfed_traces": {}, "fit_traces": {}}

    s_components = []
    if cleaning_token_flood:
        s_components = get_s_components_from_petri(net, initial_marking, final_marking)

    notexisting_activities_in_model = {}
    trans_map = {t.label: t for t in sorted(list(net.transitions), key=lambda x: x.name) if t.label}
    event_type_capacities = compute_global_place_capacities(log, net, initial_marking, trans_map, case_id_key, timestamp_key, activity_key)
    place_capacities = compute_place_capacities(net, event_type_capacities)

    events_by_timestamp = {}
    if isinstance(log, pd.DataFrame):
        for case_id, group in log.groupby(case_id_key):
            sorted_group = group.sort_values(timestamp_key)
            for idx, row in sorted_group.iterrows():
                event = row.to_dict()
                ts = event[timestamp_key]
                next_ts = sorted_group.iloc[idx + 1][timestamp_key] if idx + 1 < len(sorted_group) else None
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((case_id, idx, event, next_ts))
    else:
        for trace in log:
            case_id = trace.attributes[case_id_key]
            sorted_trace = sorted(trace, key=lambda x: x.get(timestamp_key, 0))
            for j, event in enumerate(sorted_trace):
                ts = event[timestamp_key]
                next_ts = sorted_trace[j + 1][timestamp_key] if j + 1 < len(sorted_trace) else None
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((case_id, j, event, next_ts))

    traces = [(trace.attributes[case_id_key], [event[activity_key] for event in trace], trace) for trace in log] if not isinstance(log, pd.DataFrame) else \
             [(case_id, list(group[activity_key]), group.to_dict('records')) for case_id, group in log.groupby(case_id_key)]

    for i, (case_id, activities, trace_data) in enumerate(traces):
        t = ApplyTraceTokenReplay(trace_data, net, initial_marking, final_marking,
                                  trans_map, enable_pltr_fitness, place_fitness_per_trace,
                                  transition_fitness_per_trace, notexisting_activities_in_model,
                                  places_shortest_path_by_hidden, consider_remaining_in_fitness,
                                  activity_key=activity_key, reach_mark_through_hidden=reach_mark_through_hidden,
                                  stop_immediately_unfit=stop_immediately_unfit,
                                  walk_through_hidden_trans=walk_through_hidden_trans,
                                  is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time,
                                  cleaning_token_flood=cleaning_token_flood, s_components=s_components,
                                  trace_occurrences=1, consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness,
                                  events_by_timestamp=events_by_timestamp, global_place_counts={},
                                  timestamp_key=timestamp_key, place_capacities=place_capacities)
        t.run()
        aligned_traces.append({
            "trace_is_fit": t.t_fit,
            "trace_fitness": t.t_value,
            "activated_transitions": [trans.name for trans in t.act_trans],
            "transitions_with_problems": [trans.name for trans in t.trans_probl],
            "reached_marking": {place.name: count for place, count in t.reached_marking.items()},
            "enabled_transitions_in_marking": [trans.name for trans in t.enabled_trans_in_mark],
            "missing_tokens": t.missing,
            "consumed_tokens": t.consumed,
            "remaining_tokens": t.remaining,
            "produced_tokens": t.produced,
            "capacity_exceeded_entries": t.capacity_exceeded_entries
        })

    if enable_pltr_fitness:
        return aligned_traces, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model, event_type_capacities
    return aligned_traces, event_type_capacities

def compute_global_place_capacities(log, net, initial_marking, trans_map, case_id_key, timestamp_key="time:timestamp", activity_key="concept:name"):
    all_events = []
    if isinstance(log, pd.DataFrame):
        for case_id, group in log.groupby(case_id_key):
            sorted_group = group.sort_values(timestamp_key)
            for idx, row in sorted_group.iterrows():
                event = row.to_dict()
                all_events.append((event[timestamp_key], case_id, event))
    else:
        for trace in log:
            case_id = trace.attributes[case_id_key]
            sorted_trace = sorted(trace, key=lambda x: x.get(timestamp_key, 0))
            for event in sorted_trace:
                all_events.append((event[timestamp_key], case_id, event))
    
    all_events.sort(key=lambda x: x[0])
    
    global_marking = copy(initial_marking)
    place_max_capacities = {place: global_marking.get(place, 0) for place in net.places}
    
    for ts, case_id, event in all_events:
        if event[activity_key] in trans_map:
            t = trans_map[event[activity_key]]
            if semantics.is_enabled(t, net, global_marking):
                global_marking = semantics.execute(t, net, global_marking)
                for place in global_marking:
                    if global_marking[place] > place_max_capacities[place]:
                        place_max_capacities[place] = global_marking[place]
    
    event_type_capacities = {}
    for place, capacity in place_max_capacities.items():
        incoming_transitions = [t for t in net.transitions if any(arc.source == place and arc.target == t for arc in t.in_arcs)]
        for trans in incoming_transitions:
            if trans.label:
                event_type = trans.label
                if event_type not in event_type_capacities:
                    event_type_capacities[event_type] = capacity
                else:
                    event_type_capacities[event_type] = max(event_type_capacities[event_type], capacity)
    
    return event_type_capacities