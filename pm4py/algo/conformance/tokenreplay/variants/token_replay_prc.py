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

class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    VARIANTS = "variants"
    PLACES_SHORTEST_PATH_BY_HIDDEN = "places_shortest_path_by_hidden"
    THREAD_MAX_EX_TIME = "thread_maximum_ex_time"
    DISABLE_VARIANTS = "disable_variants"
    CLEANING_TOKEN_FLOOD = "cleaning_token_flood"
    IS_REDUCTION = "is_reduction"
    WALK_THROUGH_HIDDEN_TRANS = "walk_through_hidden_trans"
    RETURN_NAMES = "return_names"
    STOP_IMMEDIATELY_UNFIT = "stop_immediately_unfit"
    TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN = "try_to_reach_final_marking_through_hidden"
    CONSIDER_REMAINING_IN_FITNESS = "consider_remaining_in_fitness"
    CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS = "consider_activities_not_in_model_in_fitness"
    ENABLE_PLTR_FITNESS = "enable_pltr_fitness"
    SHOW_PROGRESS_BAR = "show_progress_bar"


class TechnicalParameters(Enum):
    MAX_REC_DEPTH = 50
    MAX_IT_FINAL1 = 5
    MAX_IT_FINAL2 = 5
    MAX_REC_DEPTH_HIDTRANSENABL = 2
    MAX_POSTFIX_SUFFIX_LENGTH = 20
    MAX_NO_THREADS = 1024
    MAX_DEF_THR_EX_TIME = 10
    ENABLE_POSTFIX_CACHE = False
    ENABLE_MARKTOACT_CACHE = False


class DebugConst:
    REACH_MRH = -1
    REACH_ITF1 = -1
    REACH_ITF2 = -1


class NoConceptNameException(Exception):
    def __init__(self, message):
        self.message = message


def add_missing_tokens(t, marking):
    """
    Adds missing tokens needed to activate a transition

    Parameters
    ----------
    t
        Transition that should be enabled
    marking
        Current marking
    """
    missing = 0
    tokens_added = {}
    for a in t.in_arcs:
        if marking[a.source] < a.weight:
            missing = missing + (a.weight - marking[a.source])
            tokens_added[a.source] = a.weight - marking[a.source]
            marking[a.source] = marking[a.source] + a.weight
    return [missing, tokens_added]


def get_consumed_tokens(t):
    """
    Get tokens consumed firing a transition

    Parameters
    ----------
    t
        Transition that should be enabled
    """
    consumed = 0
    consumed_map = {}
    for a in t.in_arcs:
        consumed = consumed + a.weight
        consumed_map[a.source] = a.weight
    return consumed, consumed_map


def get_produced_tokens(t):
    """
    Get tokens produced firing a transition

    Parameters
    ----------
    t
        Transition that should be enabled
    """
    produced = 0
    produced_map = {}
    for a in t.out_arcs:
        produced = produced + a.weight
        produced_map[a.target] = a.weight
    return produced, produced_map


def merge_dicts(x, y):
    """
    Merge two dictionaries keeping the least value

    Parameters
    ----------
    x
        First map (string, integer)
    y
        Second map (string, integer)
    """
    for key in y:
        if key not in x:
            x[key] = y[key]
        else:
            if y[key] < x[key]:
                x[key] = y[key]


def get_places_with_missing_tokens(t, marking):
    """
    Get places with missing tokens

    Parameters
    ----------
    t
        Transition to enable
    marking
        Current marking
    """
    places_with_missing = set()
    for a in t.in_arcs:
        if marking[a.source] < a.weight:
            places_with_missing.add(a.source)
    return places_with_missing


def get_hidden_transitions_to_enable(marking, places_with_missing, places_shortest_path_by_hidden):
    """
    Calculate an ordered list of transitions to visit in order to enable a given transition

    Parameters
    ----------
    marking
        Current marking
    places_with_missing
        List of places with missing tokens
    places_shortest_path_by_hidden
        Minimal connection between places by hidden transitions
    """
    hidden_transitions_to_enable = []

    marking_places = [x for x in marking]
    marking_places = sorted(marking_places, key=lambda x: x.name)
    places_with_missing_keys = [x for x in places_with_missing]
    places_with_missing_keys = sorted(places_with_missing_keys, key=lambda x: x.name)
    for p1 in marking_places:
        for p2 in places_with_missing_keys:
            if p1 in places_shortest_path_by_hidden and p2 in places_shortest_path_by_hidden[p1]:
                hidden_transitions_to_enable.append(places_shortest_path_by_hidden[p1][p2])
    hidden_transitions_to_enable = sorted(hidden_transitions_to_enable, key=lambda x: len(x))

    return hidden_transitions_to_enable


def get_req_transitions_for_final_marking(marking, final_marking, places_shortest_path_by_hidden):
    """
    Gets required transitions for final marking

    Parameters
    ----------
    marking
        Current marking
    final_marking
        Final marking assigned to the Petri net
    places_shortest_path_by_hidden
        Minimal connection between places by hidden transitions
    """
    hidden_transitions_to_enable = []

    marking_places = [x for x in marking]
    marking_places = sorted(marking_places, key=lambda x: x.name)
    final_marking_places = [x for x in final_marking]
    final_marking_places = sorted(final_marking_places, key=lambda x: x.name)
    for p1 in marking_places:
        for p2 in final_marking_places:
            if p1 in places_shortest_path_by_hidden and p2 in places_shortest_path_by_hidden[p1]:
                hidden_transitions_to_enable.append(places_shortest_path_by_hidden[p1][p2])
    hidden_transitions_to_enable = sorted(hidden_transitions_to_enable, key=lambda x: len(x))

    return hidden_transitions_to_enable


def enable_hidden_transitions(net, marking, activated_transitions, visited_transitions, all_visited_markings,
                              hidden_transitions_to_enable, t):
    """
    Actually enable hidden transitions on the Petri net

    Parameters
    -----------
    net
        Petri net
    marking
        Current marking
    activated_transitions
        All activated transitions during the replay
    visited_transitions
        All visited transitions by the recursion
    all_visited_markings
        All visited markings
    hidden_transitions_to_enable
        List of hidden transition to enable
    t
        Transition against we should check if they are enabled
    """
    j_indexes = [0] * len(hidden_transitions_to_enable)
    for z in range(10000000):
        something_changed = False
        for k in range(j_indexes[z % len(hidden_transitions_to_enable)], len(
                hidden_transitions_to_enable[z % len(hidden_transitions_to_enable)])):
            t3 = hidden_transitions_to_enable[z % len(hidden_transitions_to_enable)][
                j_indexes[z % len(hidden_transitions_to_enable)]]
            if not t3 == t:
                if semantics.is_enabled(t3, net, marking):
                    if t3 not in visited_transitions:
                        marking = semantics.execute(t3, net, marking)
                        activated_transitions.append(t3)
                        visited_transitions.add(t3)
                        all_visited_markings.append(marking)
                        something_changed = True
            j_indexes[z % len(hidden_transitions_to_enable)] = j_indexes[z % len(hidden_transitions_to_enable)] + 1
            if semantics.is_enabled(t, net, marking):
                break
        if semantics.is_enabled(t, net, marking):
            break
        if not something_changed:
            break
    return [marking, activated_transitions, visited_transitions, all_visited_markings]


def apply_hidden_trans(t, net, marking, places_shortest_paths_by_hidden, act_tr, rec_depth,
                       visit_trans,
                       vis_mark):
    """
    Apply hidden transitions in order to enable a given transition

    Parameters
    ----------
    t
        Transition to eventually enable
    net
        Petri net
    marking
        Marking
    places_shortest_paths_by_hidden
        Shortest paths between places connected by hidden transitions
    act_tr
        All activated transitions
    rec_depth
        Current recursion depth
    visit_trans
        All visited transitions by hiddenTrans method
    vis_mark
        All visited markings
    """
    if rec_depth >= TechnicalParameters.MAX_REC_DEPTH_HIDTRANSENABL.value or t in visit_trans:
        return [net, marking, act_tr, vis_mark]
    # if rec_depth > DebugConst.REACH_MRH:
    #    DebugConst.REACH_MRH = rec_depth
    visit_trans.add(t)
    marking_at_start = copy(marking)
    places_with_missing = get_places_with_missing_tokens(t, marking)
    hidden_transitions_to_enable = get_hidden_transitions_to_enable(marking, places_with_missing,
                                                                    places_shortest_paths_by_hidden)

    if hidden_transitions_to_enable:
        [marking, act_tr, visit_trans, vis_mark] = enable_hidden_transitions(net,
                                                                             marking,
                                                                             act_tr,
                                                                             visit_trans,
                                                                             vis_mark,
                                                                             hidden_transitions_to_enable,
                                                                             t)
        if not semantics.is_enabled(t, net, marking):
            hidden_transitions_to_enable = get_hidden_transitions_to_enable(marking, places_with_missing,
                                                                            places_shortest_paths_by_hidden)
            for z in range(len(hidden_transitions_to_enable)):
                for k in range(len(hidden_transitions_to_enable[z])):
                    t4 = hidden_transitions_to_enable[z][k]
                    if not t4 == t:
                        if t4 not in visit_trans:
                            if not semantics.is_enabled(t4, net, marking):
                                [net, marking, act_tr, vis_mark] = apply_hidden_trans(t4,
                                                                                      net,
                                                                                      marking,
                                                                                      places_shortest_paths_by_hidden,
                                                                                      act_tr,
                                                                                      rec_depth + 1,
                                                                                      visit_trans,
                                                                                      vis_mark)
                            if semantics.is_enabled(t4, net, marking):
                                marking = semantics.execute(t4, net, marking)
                                act_tr.append(t4)
                                visit_trans.add(t4)
                                vis_mark.append(marking)
        if not semantics.is_enabled(t, net, marking):
            if not (marking_at_start == marking):
                [net, marking, act_tr, vis_mark] = apply_hidden_trans(t, net, marking,
                                                                      places_shortest_paths_by_hidden,
                                                                      act_tr,
                                                                      rec_depth + 1,
                                                                      visit_trans,
                                                                      vis_mark)

    return [net, marking, act_tr, vis_mark]


def break_condition_final_marking(marking, final_marking):
    """
    Verify break condition for final marking

    Parameters
    -----------
    marking
        Current marking
    final_marking
        Target final marking
    """
    final_marking_dict = dict(final_marking)
    marking_dict = dict(marking)
    final_marking_dict_keys = set(final_marking_dict.keys())
    marking_dict_keys = set(marking_dict.keys())

    return final_marking_dict_keys.issubset(marking_dict_keys)

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

    # Debugging-Informationen sammeln
    debug_data = []
    place_activity_data = []  # Neu: Für Cases, die den Zielplatz passieren

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

            # Debugging für einen spezifischen Platz
            target_place_name = "({'E5GAG45_H5GGML56'}, {'E5GAG45_H5GGML5F'})"  # Ersetze mit deinem Platz
            for place in net.places:
                if place.name == target_place_name:
                    if place not in global_place_capacities:
                        global_place_capacities[place] = {'max': 0}
                    if global_place_counts[current_timestamp][place] > global_place_capacities[place]['max']:
                        global_place_capacities[place]['max'] = global_place_counts[current_timestamp][place]
                        # Sammle Debugging-Informationen
                        case_info = []
                        for evt in overlapping_events:
                            if evt[activity_key] in trans_map:
                                case_id = evt["case:concept:name"]
                                # Finde das nächste Event für diesen Case
                                next_activity = "None"
                                evt_timestamp = evt[timestamp_key]
                                for _, evt_idx, next_evt, next_ts in events_by_timestamp.get(evt_timestamp, []):
                                    if next_ts and next_ts > evt_timestamp and next_evt["case:concept:name"] == case_id:
                                        next_activity = next_evt[activity_key]
                                        break
                                case_info.append({"case_id": case_id, "next_activity": next_activity})
                        debug_data.append({
                            "place": place.name,
                            "timestamp": current_timestamp,
                            "max_capacity": global_place_counts[current_timestamp][place],
                            "contributing_cases": [info["case_id"] for info in case_info],
                            "next_activities": [info["next_activity"] for info in case_info]
                        })

            # Sammle Daten für alle Cases, die den Zielplatz passieren
            target_transition = "E5GAG45_H5GGML56"  # Eingehende Transition des Platzes
            if event[activity_key] == target_transition:
                case_id = event["case:concept:name"]
                next_activity = "None"
                if i + 1 < len(sorted_events):
                    next_activity = sorted_events[i + 1][activity_key]
                place_activity_data.append({
                    "case_id": case_id,
                    "timestamp": event[timestamp_key],
                    "activity": event[activity_key],
                    "next_activity": next_activity
                })

            # [Rest der bestehenden Logik unverändert]
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

    # [Restlicher Code unverändert]
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
            place_max_capacities, debug_data, place_activity_data]  # Füge place_activity_data hinzu

class ApplyTraceTokenReplay:
    def __init__(self, trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness,
                 transition_fitness, notexisting_activities_in_model, places_shortest_path_by_hidden,
                 consider_remaining_in_fitness, activity_key="concept:name", reach_mark_through_hidden=True,
                 stop_immediately_unfit=False, walk_through_hidden_trans=True, post_fix_caching=None,
                 marking_to_activity_caching=None, is_reduction=False,
                 thread_maximum_ex_time=TechnicalParameters.MAX_DEF_THR_EX_TIME.value,
                 cleaning_token_flood=False, s_components=None, trace_occurrences=1,
                 consider_activities_not_in_model_in_fitness=False, events_by_timestamp=None,
                 global_place_counts=None, global_place_capacities=None, timestamp_key="time:timestamp"):
        """
        Constructor
        """
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
        self.consider_activities_not_in_model_in_fitness = consider_activities_not_in_model_in_fitness
        self.activity_key = activity_key
        self.reach_mark_through_hidden = reach_mark_through_hidden  # Korrigiert
        self.stop_immediately_unfit = stop_immediately_unfit
        self.walk_through_hidden_trans = walk_through_hidden_trans
        self.post_fix_caching = post_fix_caching
        self.marking_to_activity_caching = marking_to_activity_caching
        self.is_reduction = is_reduction
        self.thread_maximum_ex_time = thread_maximum_ex_time
        self.cleaning_token_flood = cleaning_token_flood
        self.enable_postfix_cache = TechnicalParameters.ENABLE_POSTFIX_CACHE.value
        self.enable_marktoact_cache = TechnicalParameters.ENABLE_MARKTOACT_CACHE.value
        if self.is_reduction:
            self.enable_postfix_cache = True
            self.enable_marktoact_cache = True
        self.t_fit = None
        self.t_value = None
        self.act_trans = None
        self.trans_probl = None
        self.reached_marking = None
        self.enabled_trans_in_mark = None
        self.missing = None
        self.consumed = None
        self.remaining = None
        self.produced = None
        self.place_max_capacities = None
        self.debug_data = None
        self.place_activity_data = None
        self.s_components = s_components
        self.trace_occurrences = trace_occurrences
        self.events_by_timestamp = events_by_timestamp
        self.global_place_counts = global_place_counts
        self.global_place_capacities = global_place_capacities
        self.timestamp_key = timestamp_key

    def run(self):
        """
        Runs the thread and stores the results
        """
        self.t_fit, self.t_value, self.act_trans, self.trans_probl, self.reached_marking, self.enabled_trans_in_mark, self.missing, self.consumed, self.remaining, self.produced, self.place_max_capacities, self.debug_data, self.place_activity_data = \
            apply_trace(self.trace, self.net, self.initial_marking, self.final_marking, self.trans_map,
                        self.enable_pltr_fitness, self.place_fitness, self.transition_fitness,
                        self.notexisting_activities_in_model,
                        self.places_shortest_path_by_hidden, self.consider_remaining_in_fitness,
                        activity_key=self.activity_key,
                        try_to_reach_final_marking_through_hidden=self.reach_mark_through_hidden,  # Korrigiert
                        stop_immediately_unfit=self.stop_immediately_unfit,
                        walk_through_hidden_trans=self.walk_through_hidden_trans,
                        post_fix_caching=self.post_fix_caching,
                        marking_to_activity_caching=self.marking_to_activity_caching,
                        is_reduction=self.is_reduction,
                        thread_maximum_ex_time=self.thread_maximum_ex_time,
                        enable_postfix_cache=self.enable_postfix_cache,
                        enable_marktoact_cache=self.enable_marktoact_cache,
                        cleaning_token_flood=self.cleaning_token_flood,
                        s_components=self.s_components,
                        trace_occurrences=self.trace_occurrences,
                        consider_activities_not_in_model_in_fitness=self.consider_activities_not_in_model_in_fitness,
                        events_by_timestamp=self.events_by_timestamp,
                        global_place_counts=self.global_place_counts,
                        global_place_capacities=self.global_place_capacities,
                        timestamp_key=self.timestamp_key)
        self.thread_is_alive = False


class PostFixCaching:
    """
    Post fix caching object
    """

    def __init__(self):
        self.cache = 0
        self.cache = {}


class MarkingToActivityCaching:
    """
    Marking to activity caching
    """

    def __init__(self):
        self.cache = 0
        self.cache = {}


def get_variant_from_trace(trace, activity_key, disable_variants=False):
    """
    Gets the variant from the trace (allow disabling)

    Parameters
    ------------
    trace
        Trace
    activity_key
        Attribute that is the activity
    disable_variants
        Boolean value that disable variants

    Returns
    -------------
    variant
        Variant describing the trace
    """
    if disable_variants:
        return str(hash(trace))
    parameters = {}
    parameters[variants_util.Parameters.ACTIVITY_KEY] = activity_key
    return variants_util.get_variant_from_trace(trace, parameters=parameters)


def transcribe_result(t, return_object_names=True):
    corr_value = {
        "trace_is_fit": copy(t.t_fit),
        "trace_fitness": float(copy(t.t_value)),
        "activated_transitions": copy(t.act_trans),
        "reached_marking": copy(t.reached_marking),
        "enabled_transitions_in_marking": copy(t.enabled_trans_in_mark),
        "transitions_with_problems": copy(t.trans_probl),
        "missing_tokens": int(t.missing),
        "consumed_tokens": int(t.consumed),
        "remaining_tokens": int(t.remaining),
        "produced_tokens": int(t.produced),
        "place_max_capacities": copy(t.place_max_capacities),
        "debug_data": copy(t.debug_data),
        "place_activity_data": copy(t.place_activity_data)  # Neu hinzugefügt
    }

    if return_object_names:
        corr_value["activated_transitions_labels"] = [x.label for x in corr_value["activated_transitions"]]
        corr_value["activated_transitions"] = [x.name for x in corr_value["activated_transitions"]]
        corr_value["enabled_transitions_in_marking_labels"] = [x.label for x in corr_value["enabled_transitions_in_marking"]]
        corr_value["enabled_transitions_in_marking"] = [x.name for x in corr_value["enabled_transitions_in_marking"]]
        corr_value["transitions_with_problems"] = [x.name for x in corr_value["transitions_with_problems"]]
        corr_value["reached_marking"] = {x.name: y for x, y in corr_value["reached_marking"].items()}

    return corr_value


def apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=False, consider_remaining_in_fitness=False,
              activity_key="concept:name", reach_mark_through_hidden=True, stop_immediately_unfit=False,
              walk_through_hidden_trans=True, places_shortest_path_by_hidden=None,
              is_reduction=False, thread_maximum_ex_time=10,
              cleaning_token_flood=False, disable_variants=False, return_object_names=False, show_progress_bar=True,
              consider_activities_not_in_model_in_fitness=False, case_id_key=constants.CASE_CONCEPT_NAME, 
              timestamp_key="time:timestamp"):
    import pandas as pd
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
            sorted_group = group.sort_values(timestamp_key)
            for idx, row in sorted_group.iterrows():
                event = row.to_dict()
                ts = event[timestamp_key]
                next_ts = None
                if idx + 1 < len(sorted_group):
                    next_row = sorted_group.iloc[sorted_group.index.get_loc(idx) + 1]
                    next_ts = next_row[timestamp_key]
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((case_id, idx, event, next_ts))
    else:
        for i, trace in enumerate(log):
            sorted_trace = sorted(trace, key=lambda x: x.get(timestamp_key, 0))
            for j, event in enumerate(sorted_trace):
                ts = event[timestamp_key]
                next_ts = None
                if j + 1 < len(sorted_trace):
                    next_ts = sorted_trace[j + 1][timestamp_key]
                if ts not in events_by_timestamp:
                    events_by_timestamp[ts] = []
                events_by_timestamp[ts].append((trace.attributes[case_id_key], j, event, next_ts))

    # Initialize global place counts, capacities, and debug data
    global_place_counts = {}
    global_place_capacities = {}
    all_debug_data = []
    all_place_activity_data = []

    # Process each trace individually
    threads_results = {}
    progress = None
    if importlib.util.find_spec("tqdm") and show_progress_bar and len(log) > 1:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(log), desc="replaying log with TBR, completed traces :: ")

    if pandas_utils.check_is_pandas_dataframe(log):
        traces = [(case_id, list(group[activity_key]), group.to_dict('records')) for case_id, group in log.groupby(case_id_key)]
    else:
        traces = [(trace.attributes[case_id_key], [event[activity_key] for event in trace], trace) for trace in log]

    for i, (case_id, activities, trace_data) in enumerate(traces):
        t = ApplyTraceTokenReplay(trace_data, net, initial_marking, final_marking,
                                  trans_map, enable_pltr_fitness, place_fitness_per_trace,
                                  transition_fitness_per_trace, notexisting_activities_in_model,
                                  places_shortest_path_by_hidden, consider_remaining_in_fitness,
                                  activity_key=activity_key, reach_mark_through_hidden=reach_mark_through_hidden,
                                  stop_immediately_unfit=stop_immediately_unfit,
                                  walk_through_hidden_trans=walk_through_hidden_trans,
                                  post_fix_caching=post_fix_cache, marking_to_activity_caching=marking_to_activity_cache,
                                  is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time,
                                  cleaning_token_flood=cleaning_token_flood, s_components=s_components,
                                  trace_occurrences=1, consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness,
                                  events_by_timestamp=events_by_timestamp, global_place_counts=global_place_counts,
                                  global_place_capacities=global_place_capacities, timestamp_key=timestamp_key)
        t.run()
        threads_results[i] = transcribe_result(t, return_object_names=return_object_names)
        all_debug_data.extend(threads_results[i]["debug_data"])
        all_place_activity_data.extend(threads_results[i]["place_activity_data"])
        if progress:
            progress.update()

    for i in range(len(traces)):
        aligned_traces.append(threads_results[i])

    # Speichere Debugging-Informationen als CSV
    if all_debug_data:
        debug_df = pd.DataFrame(all_debug_data)
        debug_df.to_csv("../Ergebnisse_PRC/place_capacity_debug.csv", index=False)
        print(f"Debugging data saved to '../Ergebnisse_PRC/place_capacity_debug.csv'")

    # Speichere place_activity_data als CSV
    if all_place_activity_data:
        place_activity_df = pd.DataFrame(all_place_activity_data)
        place_activity_df.to_csv("../Ergebnisse_PRC/place_activity_log.csv", index=False)
        print(f"Place activity log saved to '../Ergebnisse_PRC/place_activity_log.csv'")

    # Erstelle CSV mit der letzten Aktivität der contributing_cases
    if all_debug_data:
        contributing_cases = set()
        for debug_entry in all_debug_data:
            for case_id in debug_entry["contributing_cases"]:
                contributing_cases.add(case_id)
        last_activity_data = []
        event_sequence_data = []  # Neu: Für die Event-Sequenzen der contributing_cases
        if pandas_utils.check_is_pandas_dataframe(log):
            for case_id, group in log.groupby(case_id_key):
                if case_id in contributing_cases:
                    # Letzte Aktivität
                    last_event = group.sort_values(timestamp_key).iloc[-1]
                    last_activity_data.append({
                        "case_id": case_id,
                        "last_activity": last_event[activity_key],
                        "last_timestamp": last_event[timestamp_key]
                    })
                    # Event-Sequenz
                    sorted_group = group.sort_values(timestamp_key)
                    for idx, row in sorted_group.iterrows():
                        event = row.to_dict()
                        next_activity = "None"
                        if idx + 1 < len(sorted_group):
                            next_row = sorted_group.iloc[sorted_group.index.get_loc(idx) + 1]
                            next_activity = next_row[activity_key]
                        event_sequence_data.append({
                            "case_id": case_id,
                            "activity": event[activity_key],
                            "timestamp": event[timestamp_key],
                            "next_activity": next_activity
                        })
        else:
            for trace in log:
                case_id = trace.attributes[case_id_key]
                if case_id in contributing_cases:
                    # Letzte Aktivität
                    last_event = max(trace, key=lambda x: x.get(timestamp_key, 0))
                    last_activity_data.append({
                        "case_id": case_id,
                        "last_activity": last_event[activity_key],
                        "last_timestamp": last_event[timestamp_key]
                    })
                    # Event-Sequenz
                    sorted_trace = sorted(trace, key=lambda x: x.get(timestamp_key, 0))
                    for j, event in enumerate(sorted_trace):
                        next_activity = "None"
                        if j + 1 < len(sorted_trace):
                            next_activity = sorted_trace[j + 1][activity_key]
                        event_sequence_data.append({
                            "case_id": case_id,
                            "activity": event[activity_key],
                            "timestamp": event[timestamp_key],
                            "next_activity": next_activity
                        })
        if last_activity_data:
            last_activity_df = pd.DataFrame(last_activity_data)
            last_activity_df.to_csv("../Ergebnisse_PRC/contributing_cases_last_activity.csv", index=False)
            print(f"Last activity data saved to '../Ergebnisse_PRC/contributing_cases_last_activity.csv'")
        if event_sequence_data:
            event_sequence_df = pd.DataFrame(event_sequence_data)
            event_sequence_df.to_csv("../Ergebnisse_PRC/contributing_cases_event_sequence.csv", index=False)
            print(f"Event sequence data saved to '../Ergebnisse_PRC/contributing_cases_event_sequence.csv'")

    if progress:
        progress.close()

    # Update aligned_traces with global place capacities
    for i in range(len(aligned_traces)):
        aligned_traces[i]["place_max_capacities"] = {place.name: capacities['max'] for place, capacities in global_place_capacities.items()}

    if enable_pltr_fitness:
        return aligned_traces, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model
    else:
        return aligned_traces


def apply(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.ListAlignments:
    """
    Method to apply token-based replay

    Parameters
    -----------
    log
        Log
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm
    """
    if parameters is None:
        parameters = {}

    enable_pltr_fitness = exec_utils.get_param_value(Parameters.ENABLE_PLTR_FITNESS, parameters, False)
    # changed default to uniform behavior with token-based replay fitness
    consider_remaining_in_fitness = exec_utils.get_param_value(Parameters.CONSIDER_REMAINING_IN_FITNESS, parameters,
                                                               True)
    try_to_reach_final_marking_through_hidden = exec_utils.get_param_value(
        Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN, parameters, True)
    stop_immediately_unfit = exec_utils.get_param_value(Parameters.STOP_IMMEDIATELY_UNFIT, parameters, False)
    walk_through_hidden_trans = exec_utils.get_param_value(Parameters.WALK_THROUGH_HIDDEN_TRANS, parameters, True)
    is_reduction = exec_utils.get_param_value(Parameters.IS_REDUCTION, parameters, False)
    cleaning_token_flood = exec_utils.get_param_value(Parameters.CLEANING_TOKEN_FLOOD, parameters, False)
    disable_variants = exec_utils.get_param_value(Parameters.DISABLE_VARIANTS, parameters, enable_pltr_fitness)
    return_names = exec_utils.get_param_value(Parameters.RETURN_NAMES, parameters, False)
    thread_maximum_ex_time = exec_utils.get_param_value(Parameters.THREAD_MAX_EX_TIME, parameters,
                                                        TechnicalParameters.MAX_DEF_THR_EX_TIME.value)
    places_shortest_path_by_hidden = exec_utils.get_param_value(Parameters.PLACES_SHORTEST_PATH_BY_HIDDEN, parameters,
                                                                None)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    consider_activities_not_in_model_in_fitness = exec_utils.get_param_value(Parameters.CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS, parameters, False)

    show_progress_bar = exec_utils.get_param_value(Parameters.SHOW_PROGRESS_BAR, parameters, constants.SHOW_PROGRESS_BAR)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)

    if type(log) is not pd.DataFrame:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    return apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=enable_pltr_fitness,
                     consider_remaining_in_fitness=consider_remaining_in_fitness,
                     reach_mark_through_hidden=try_to_reach_final_marking_through_hidden,
                     stop_immediately_unfit=stop_immediately_unfit,
                     walk_through_hidden_trans=walk_through_hidden_trans,
                     places_shortest_path_by_hidden=places_shortest_path_by_hidden, activity_key=activity_key,
                     is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time,
                     cleaning_token_flood=cleaning_token_flood, disable_variants=disable_variants,
                     return_object_names=return_names, show_progress_bar=show_progress_bar,
                     consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness,
                     case_id_key=case_id_key)


def get_diagnostics_dataframe(log: Union[EventLog, pd.DataFrame], tbr_output: typing.ListAlignments, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> pd.DataFrame:
    if parameters is None:
        parameters = {}
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, xes_util.DEFAULT_TRACEID_KEY)
    import pandas as pd
    diagn_stream = []
    
    if isinstance(log, pd.DataFrame):
        for index, row in log.groupby(case_id_key).first().reset_index().iterrows():
            case_id = row[case_id_key]
            is_fit = tbr_output[index]["trace_is_fit"]
            trace_fitness = tbr_output[index]["trace_fitness"]
            missing = tbr_output[index]["missing_tokens"]
            remaining = tbr_output[index]["remaining_tokens"]
            produced = tbr_output[index]["produced_tokens"]
            consumed = tbr_output[index]["consumed_tokens"]
            place_max_capacities = tbr_output[index]["place_max_capacities"]
            diagn_stream.append({
                "case_id": case_id,
                "is_fit": is_fit,
                "trace_fitness": trace_fitness,
                "missing": missing,
                "remaining": remaining,
                "produced": produced,
                "consumed": consumed,
                "place_max_capacities": place_max_capacities
            })
    else:
        for index in range(len(log)):
            case_id = log[index].attributes[case_id_key]
            is_fit = tbr_output[index]["trace_is_fit"]
            trace_fitness = tbr_output[index]["trace_fitness"]
            missing = tbr_output[index]["missing_tokens"]
            remaining = tbr_output[index]["remaining_tokens"]
            produced = tbr_output[index]["produced_tokens"]
            consumed = tbr_output[index]["consumed_tokens"]
            place_max_capacities = tbr_output[index]["place_max_capacities"]
            diagn_stream.append({
                "case_id": case_id,
                "is_fit": is_fit,
                "trace_fitness": trace_fitness,
                "missing": missing,
                "remaining": remaining,
                "produced": produced,
                "consumed": consumed,
                "place_max_capacities": place_max_capacities
            })
    
    return pandas_utils.instantiate_dataframe(diagn_stream)