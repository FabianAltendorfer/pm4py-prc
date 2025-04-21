from pm4py.util import xes_constants as xes_util
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils.petri_utils import get_places_shortest_path_by_hidden, get_s_components_from_petri
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils
from copy import copy
from enum import Enum
from pm4py.util import exec_utils, constants
from pm4py.util import variants_util, pandas_utils
import importlib.util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing
from collections import Counter
from pm4py.objects.conversion.log import converter as log_converter

# [Existing enums and helper functions unchanged]
# ... (Parameters, TechnicalParameters, DebugConst, add_missing_tokens, get_consumed_tokens, get_produced_tokens, etc.)

def get_overlapping_events(event, events_by_timestamp, activity_key, timestamp_key="time:timestamp"):
    """
    Find events with the same timestamp as the given event, where the subsequent event
    (if any) has a later timestamp or does not exist.
    
    Parameters
    ----------
    event : dict
        The current event being processed.
    events_by_timestamp : dict
        Dictionary mapping timestamps to lists of (case_id, event_index, event, next_timestamp) tuples.
    activity_key : str
        Key for activity name.
    timestamp_key : str
        Key for timestamp.
    
    Returns
    -------
    list
        List of events with overlapping timestamps and unprocessed subsequent events.
    """
    current_timestamp = event[timestamp_key]
    if current_timestamp not in events_by_timestamp:
        return []
    
    overlapping = []
    for case_id, event_idx, evt, next_ts in events_by_timestamp[current_timestamp]:
        if next_ts is None or next_ts > current_timestamp:
            overlapping.append(evt)
    
    return overlapping

def apply_trace(trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness,
                transition_fitness, notexisting_activities_in_model,
                places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key="concept:name",
                try_to_reach_final_marking_through_hidden=True, stop_immediately_unfit=False,
                walk_through_hidden_trans=True, post_fix_caching=None,
                marking_to_activity_caching=None, is_reduction=False,
                thread_maximum_ex_time=10, enable_postfix_cache=False, enable_marktoact_cache=False,
                cleaning_token_flood=False, s_components=None, trace_occurrences=1,
                consider_activities_not_in_model_in_fitness=False, events_by_timestamp=None,
                global_place_counts=None, global_place_capacities=None, timestamp_key="time:timestamp"):
    """
    Modified apply_trace to update global place capacities based on overlapping timestamps.
    
    Parameters
    ----------
    [Existing parameters unchanged]
    events_by_timestamp : dict
        Dictionary mapping timestamps to lists of (case_id, event_index, event, next_timestamp) tuples.
    global_place_counts : dict
        Global dictionary to track current token counts per timestamp.
    global_place_capacities : dict
        Global dictionary to store maximum place capacities across all timestamps.
    timestamp_key : str
        Key for timestamp.
    """
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

    place_capacities = {place: {'current': 0, 'max': 0} for place in net.places}
    for place, tokens in initial_marking.items():
        place_capacities[place]['current'] = tokens
        place_capacities[place]['max'] = tokens

    for i, event in enumerate(sorted_events):
        prev_len_activated_transitions = len(act_trans)
        if event[activity_key] in trans_map:
            t = trans_map[event[activity_key]]
            current_event_map.update(event)

            # Get overlapping events
            overlapping_events = get_overlapping_events(event, events_by_timestamp, activity_key, timestamp_key)
            
            # Update global place counts for this timestamp
            current_timestamp = event[timestamp_key]
            if current_timestamp not in global_place_counts:
                global_place_counts[current_timestamp] = {place: 0 for place in net.places}
            
            temp_marking = copy(initial_marking)
            for overlap_event in overlapping_events:
                if overlap_event[activity_key] in trans_map:
                    overlap_t = trans_map[overlap_event[activity_key]]
                    if semantics.is_enabled(overlap_t, net, temp_marking):
                        c, cmap = get_consumed_tokens(overlap_t)
                        p, pmap = get_produced_tokens(overlap_t)
                        temp_marking = semantics.execute(overlap_t, net, temp_marking)
                        for place in cmap:
                            global_place_counts[current_timestamp][place] -= cmap[place]
                        for place in pmap:
                            global_place_counts[current_timestamp][place] += pmap[place]

            # Update global capacities if higher
            for place in net.places:
                if place not in global_place_capacities:
                    global_place_capacities[place] = {'max': 0}
                if global_place_counts[current_timestamp][place] > global_place_capacities[place]['max']:
                    global_place_capacities[place]['max'] = global_place_counts[current_timestamp][place]

            # Process current event
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
                for place in cmap:
                    place_capacities[place]['current'] -= cmap[place]
                marking = semantics.execute(t, net, marking)
                for place in pmap:
                    place_capacities[place]['current'] += pmap[place]
                    if place_capacities[place]['current'] > place_capacities[place]['max']:
                        place_capacities[place]['max'] = place_capacities[place]['current']
                act_trans.append(t)
                vis_mark.append(marking)

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

    # [Existing final marking and fitness logic unchanged]
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

    place_max_capacities = {place: capacities['max'] for place, capacities in place_capacities.items()}
    return [is_fit, trace_fitness, act_trans, transitions_with_problems, marking_before_cleaning,
            semantics.enabled_transitions(net, marking_before_cleaning), missing, consumed, remaining, produced,
            place_max_capacities]

# [Existing ApplyTraceTokenReplay class unchanged]
# ... (class ApplyTraceTokenReplay, PostFixCaching, MarkingToActivityCaching, get_variant_from_trace, transcribe_result)

def apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=False, consider_remaining_in_fitness=False,
              activity_key="concept:name", reach_mark_through_hidden=True, stop_immediately_unfit=False,
              walk_through_hidden_trans=True, places_shortest_path_by_hidden=None,
              is_reduction=False, thread_maximum_ex_time=10,
              cleaning_token_flood=False, disable_variants=False, return_object_names=False, show_progress_bar=True,
              consider_activities_not_in_model_in_fitness=False, case_id_key=constants.CASE_CONCEPT_NAME):
    post_fix_cache = PostFixCaching()
    marking_to_activity_cache = MarkingToActivityCaching()
    if places_shortest_path_by_hidden is None:
        places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(net, TechnicalParameters.MAX_REC_DEPTH.value)

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

    trans_map = {t.label: t for t in sorted(list(net.transitions), key=lambda x: x.name)}

    # Precompute events by timestamp
    events_by_timestamp = {}
    if pandas_utils.check_is_pandas_dataframe(log):
        for case_id, group in log.groupby(case_id_key):
            sorted_group = group.sort_values("time:timestamp")
            for idx, row in sorted_group.iterrows():
                event = row.to_dict()
                ts = event["time:timestamp"]
                # Find next event's timestamp
                next_ts = None
                if idx + 1 < len(sorted_group):
                    next_row = sorted_group.iloc[sorted_group.index.get_loc(idx) + 1]
                    next_ts = next_row["time:timestamp"]
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((case_id, idx, event, next_ts))
    else:
        for i, trace in enumerate(log):
            sorted_trace = sorted(trace, key=lambda x: x.get("time:timestamp", 0))
            for j, event in enumerate(sorted_trace):
                ts = event["time:timestamp"]
                next_ts = None
                if j + 1 < len(sorted_trace):
                    next_ts = sorted_trace[j + 1]["time:timestamp"]
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((trace.attributes[case_id_key], j, event, next_ts))

    # Initialize global place counts and capacities
    global_place_counts = {}
    global_place_capacities = {}

    # Process each trace individually
    threads_results = {}
    progress = None
    if importlib.util.find_spec("tqdm") and show_progress_bar and len(log) > 1:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(log), desc="replaying log with TBR, completed traces :: ")

    if pandas_utils.check_is_pandas_dataframe(log):
        traces = [(case_id, list(group[activity_key]), group) for case_id, group in log.groupby(case_id_key)]
    else:
        traces = [(trace.attributes[case_id_key], [event[activity_key] for event in trace], trace) for trace in log]

    for i, (case_id, activities, trace_data) in enumerate(traces):
        t = ApplyTraceTokenReplay(trace_data, net, initial_marking, final_marking,
                                  trans_map, enable_pltr_fitness, place_fitness_per_trace,
                                  transition_fitness_per_trace, notexisting_activities_in_model,
                                  places_shortest_path_by_hidden, consider_remaining_in_fitness,
                                  activity_key=activity_key, reach_mark_through_hidden=reach_mark_through_hidden,
                                  stop_immediately_when_unfit=stop_immediately_unfit,
                                  walk_through_hidden_trans=walk_through_hidden_trans,
                                  post_fix_caching=post_fix_cache, marking_to_activity_caching=marking_to_activity_cache,
                                  is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time,
                                  cleaning_token_flood=cleaning_token_flood, s_components=s_components,
                                  trace_occurrences=1, consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness,
                                  events_by_timestamp=events_by_timestamp, global_place_counts=global_place_counts,
                                  global_place_capacities=global_place_capacities)
        t.run()
        threads_results[i] = transcribe_result(t, return_object_names=return_object_names)
        if progress:
            progress.update()

    for i in range(len(traces)):
        aligned_traces.append(threads_results[i])

    if progress:
        progress.close()

    # Update aligned_traces with global place capacities
    for i in range(len(aligned_traces)):
        aligned_traces[i]["place_max_capacities"] = {place.name: capacities['max'] for place, capacities in global_place_capacities.items()}

    if enable_pltr_fitness:
        return aligned_traces, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model
    else:
        return aligned_traces

# [Existing apply and get_diagnostics_dataframe unchanged]
# ... (apply, get_diagnostics_dataframe)