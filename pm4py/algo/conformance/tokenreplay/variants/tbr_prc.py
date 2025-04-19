'''
    PM4Py – A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschränkt)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''
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
from typing import Optional, Dict, Any, Union, List, Tuple
from pm4py.objects.log.obj import EventLog
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing
from collections import Counter, defaultdict
from pm4py.objects.conversion.log import converter as log_converter
import numpy as np

class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
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
    MACHINE_ALERT_KEY = "machine_alert_key"

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

def merge_dicts(x, y):
    for key in y:
        if key not in x:
            x[key] = y[key]
        else:
            if y[key] < x[key]:
                x[key] = y[key]

def get_places_with_missing_tokens(t, marking):
    places_with_missing = set()
    for a in t.in_arcs:
        if marking[a.source] < a.weight:
            places_with_missing.add(a.source)
    return places_with_missing

def get_hidden_transitions_to_enable(marking, places_with_missing, places_shortest_path_by_hidden):
    hidden_transitions = []
    marking_places = sorted([x for x in marking], key=lambda x: x.name)
    places_with_missing_keys = sorted([x for x in places_with_missing], key=lambda x: x.name)
    for p1 in marking_places:
        for p2 in places_with_missing_keys:
            if p1 in places_shortest_path_by_hidden and p2 in places_shortest_path_by_hidden[p1]:
                path = places_shortest_path_by_hidden[p1][p2]
                transitions = [trans for trans, weight in path if trans is not None and weight is not None]
                if transitions:
                    hidden_transitions.append(transitions)
    hidden_transitions = sorted(hidden_transitions, key=lambda x: len(x))
    print(f"get_hidden_transitions_to_enable returns: {hidden_transitions}")
    return hidden_transitions

def get_req_transitions_for_final_marking(marking, final_marking, places_shortest_path_by_hidden):
    hidden_transitions = []
    marking_places = sorted([x for x in marking], key=lambda x: x.name)
    final_marking_places = sorted([x for x in final_marking], key=lambda x: x.name)
    for p1 in marking_places:
        for p2 in final_marking_places:
            if p1 in places_shortest_path_by_hidden and p2 in places_shortest_path_by_hidden[p1]:
                path = places_shortest_path_by_hidden[p1][p2]
                transitions = [trans for trans, weight in path if trans is not None and weight is not None]
                if transitions:
                    hidden_transitions.append(transitions)
    hidden_transitions = sorted(hidden_transitions, key=lambda x: len(x))
    return hidden_transitions

def enable_hidden_transitions(net, marking, activated_transitions, visited_transitions, all_visited_markings, hidden_transitions, t):
    print(f"enable_hidden_transitions aufgerufen für Transition {t}, hidden_transitions: {hidden_transitions}")
    j_indexes = [0] * len(hidden_transitions)
    for z in range(10000000):
        something_changed = False
        path_index = z % len(hidden_transitions)
        if j_indexes[path_index] < len(hidden_transitions[path_index]):
            t3 = hidden_transitions[path_index][j_indexes[path_index]]
            print(f"Verarbeite Transition {t3} in enable_hidden_transitions")
            if t3 != t and t3 not in visited_transitions and semantics.is_enabled(t3, net, marking):
                marking = semantics.execute(t3, net, marking)
                activated_transitions.append(t3)
                visited_transitions.add(t3)
                all_visited_markings.append(marking)
                something_changed = True
            j_indexes[path_index] += 1
            if t is not None and semantics.is_enabled(t, net, marking):
                break
        if not something_changed:
            break
    result = [marking, activated_transitions, visited_transitions, all_visited_markings]
    print(f"enable_hidden_transitions Rückgabewert für Transition {t}: {result}")
    if not isinstance(result, list) or len(result) != 4:
        print(f"Fehler: enable_hidden_transitions returned invalid object {result} für Transition {t}")
        return [marking, activated_transitions, visited_transitions, all_visited_markings]
    return result

def apply_hidden_trans(t, net, marking, places_shortest_paths_by_hidden, act_tr, rec_depth, visit_trans, vis_mark):
    print(f"apply_hidden_trans aufgerufen für Transition {t}, rec_depth: {rec_depth}, marking: {marking}")
    if rec_depth >= TechnicalParameters.MAX_REC_DEPTH_HIDTRANSENABL.value or t in visit_trans:
        print(f"Rückgabe wegen max Rekursionstiefe oder besuchter Transition {t}")
        return [net, marking, act_tr, vis_mark]
    visit_trans.add(t)
    marking_at_start = copy(marking)
    places_with_missing = get_places_with_missing_tokens(t, marking)
    print(f"places_with_missing für Transition {t}: {places_with_missing}")
    hidden_transitions = get_hidden_transitions_to_enable(marking, places_with_missing, places_shortest_paths_by_hidden)
    print(f"hidden_transitions für Transition {t}: {hidden_transitions}")
    if hidden_transitions:
        result = enable_hidden_transitions(net, marking, act_tr, visit_trans, vis_mark, hidden_transitions, t)
        print(f"enable_hidden_transitions result für Transition {t}: {result}")
        if not isinstance(result, list) or len(result) != 4:
            print(f"Fehler: enable_hidden_transitions returned invalid object {result} für Transition {t}")
            return [net, marking, act_tr, vis_mark]
        [marking, act_tr, visit_trans, vis_mark] = result
        if not semantics.is_enabled(t, net, marking):
            hidden_transitions = get_hidden_transitions_to_enable(marking, places_with_missing, places_shortest_paths_by_hidden)
            print(f"Erneute hidden_transitions für Transition {t}: {hidden_transitions}")
            for z in range(len(hidden_transitions)):
                for k in range(len(hidden_transitions[z])):
                    t4 = hidden_transitions[z][k]
                    print(f"Verarbeite Transition {t4} in Schleife")
                    if t4 != t and t4 not in visit_trans:
                        if not semantics.is_enabled(t4, net, marking):
                            result = apply_hidden_trans(t4, net, marking, places_shortest_paths_by_hidden, act_tr, rec_depth + 1, visit_trans, vis_mark)
                            print(f"Rekursiver Aufruf result für Transition {t4}: {result}")
                            if not isinstance(result, list) or len(result) != 4:
                                print(f"Fehler: Rekursiver Aufruf von apply_hidden_trans returned invalid object {result} für Transition {t4}")
                                return [net, marking, act_tr, vis_mark]
                            [net, marking, act_tr, vis_mark] = result
                        if semantics.is_enabled(t4, net, marking):
                            marking = semantics.execute(t4, net, marking)
                            act_tr.append(t4)
                            visit_trans.add(t4)
                            vis_mark.append(marking)
        if not semantics.is_enabled(t, net, marking) and marking != marking_at_start:
            result = apply_hidden_trans(t, net, marking, places_shortest_paths_by_hidden, act_tr, rec_depth + 1, visit_trans, vis_mark)
            print(f"Rekursiver Aufruf result für Transition {t}: {result}")
            if not isinstance(result, list) or len(result) != 4:
                print(f"Fehler: Rekursiver Aufruf von apply_hidden_trans returned invalid object {result} für Transition {t}")
                return [net, marking, act_tr, vis_mark]
            [net, marking, act_tr, vis_mark] = result
    print(f"Rückgabe von apply_hidden_trans für Transition {t}: [net, {marking}, {act_tr}, {vis_mark}]")
    return [net, marking, act_tr, vis_mark]

def break_condition_final_marking(marking, final_marking):
    final_marking_dict = dict(final_marking)
    marking_dict = dict(marking)
    final_marking_dict_keys = set(final_marking_dict.keys())
    marking_dict_keys = set(marking_dict.keys())
    return final_marking_dict_keys.issubset(marking_dict_keys)

def apply_trace(trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness, transition_fitness, notexisting_activities_in_model, places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key="concept:name", try_to_reach_final_marking_through_hidden=True, stop_immediately_unfit=False, walk_through_hidden_trans=True, post_fix_caching=None, marking_to_activity_caching=None, is_reduction=False, thread_maximum_ex_time=TechnicalParameters.MAX_DEF_THR_EX_TIME.value, enable_postfix_cache=False, enable_marktoact_cache=False, cleaning_token_flood=False, s_components=None, trace_occurrences=1, consider_activities_not_in_model_in_fitness=False, timestamp_key=xes_util.DEFAULT_TIMESTAMP_KEY, machine_alert_key=None, place_token_timeline=None, place_time_diffs=None, place_alerts=None, place_storage_levels=None, place_frequency_counts=None):
    trace_activities = [event[activity_key] for event in trace]
    act_trans = []
    transitions_with_problems = []
    vis_mark = []
    activating_transition_index = {}
    activating_transition_interval = []
    used_postfix_cache = False
    marking = copy(initial_marking)
    vis_mark.append(marking)
    missing = 0
    consumed = 0
    sum_tokens_im = 0
    for place in initial_marking:
        sum_tokens_im += initial_marking[place]
    sum_tokens_fm = 0
    for place in final_marking:
        sum_tokens_fm += final_marking[place]
    produced = sum_tokens_im
    current_event_map = {}
    current_remaining_map = {}
    
    for i in range(len(trace)):
        prev_len_activated_transitions = len(act_trans)
        if enable_postfix_cache and (str(trace_activities) in post_fix_caching.cache and hash(marking) in post_fix_caching.cache[str(trace_activities)]):
            trans_to_act = post_fix_caching.cache[str(trace_activities)][hash(marking)]["trans_to_activate"]
            for z in range(len(trans_to_act)):
                t = trans_to_act[z]
                act_trans.append(t)
            used_postfix_cache = True
            marking = post_fix_caching.cache[str(trace_activities)][hash(marking)]["final_marking"]
            break
        else:
            if enable_marktoact_cache and (hash(marking) in marking_to_activity_caching.cache and trace[i][activity_key] in marking_to_activity_caching.cache[hash(marking)] and trace[i - 1][activity_key] == marking_to_activity_caching.cache[hash(marking)][trace[i][activity_key]]["previousActivity"]):
                this_end_marking = marking_to_activity_caching.cache[hash(marking)][trace[i][activity_key]]["end_marking"]
                this_act_trans = marking_to_activity_caching.cache[hash(marking)][trace[i][activity_key]]["this_activated_transitions"]
                this_vis_markings = marking_to_activity_caching.cache[hash(marking)][trace[i][activity_key]]["this_visited_markings"]
                act_trans = act_trans + this_act_trans
                vis_mark = vis_mark + this_vis_markings
                marking = copy(this_end_marking)
            else:
                if trace[i][activity_key] in trans_map:
                    current_event_map.update(trace[i])
                    corr_en_t = [x for x in semantics.enabled_transitions(net, marking) if x.label == trace[i][activity_key]]
                    if corr_en_t:
                        t = corr_en_t[0]
                    else:
                        t = trans_map[trace[i][activity_key]]
                    if walk_through_hidden_trans and not semantics.is_enabled(t, net, marking):
                        visited_transitions = set()
                        prev_len_activated_transitions = len(act_trans)
                        print(f"Aufruf von apply_hidden_trans für Transition {t}, marking: {marking}")
                        result = apply_hidden_trans(t, net, copy(marking), places_shortest_path_by_hidden, copy(act_trans), 0, copy(visited_transitions), copy(vis_mark))
                        print(f"apply_hidden_trans result für Transition {t}: {result}")
                        if not isinstance(result, list) or len(result) != 4:
                            print(f"Fehler: apply_hidden_trans returned invalid object {result} für Transition {t}")
                            continue  # Überspringe die Iteration, um den Fehler zu vermeiden
                        [net, new_marking, new_act_trans, new_vis_mark] = result
                        for jj5 in range(len(act_trans), len(new_act_trans)):
                            tt5 = new_act_trans[jj5]
                            c, cmap = get_consumed_tokens(tt5)
                            p, pmap = get_produced_tokens(tt5)
                            if enable_pltr_fitness:
                                for pl2 in cmap:
                                    if pl2 in place_fitness:
                                        place_fitness[pl2]["c"] += cmap[pl2] * trace_occurrences
                                for pl2 in pmap:
                                    if pl2 in place_fitness:
                                        place_fitness[pl2]["p"] += pmap[pl2] * trace_occurrences
                            consumed = consumed + c
                            produced = produced + p
                        marking, act_trans, vis_mark = new_marking, new_act_trans, new_vis_mark
                    is_initially_enabled = True
                    old_marking_names = [x.name for x in list(marking.keys())]
                    if not semantics.is_enabled(t, net, marking):
                        is_initially_enabled = False
                        transitions_with_problems.append(t)
                        if stop_immediately_unfit:
                            missing = missing + 1
                            break
                        [m, tokens_added] = add_missing_tokens(t, marking)
                        missing = missing + m
                        if enable_pltr_fitness:
                            for place in tokens_added.keys():
                                if place in place_fitness:
                                    place_fitness[place]["underfed_traces"].add(trace)
                                place_fitness[place]["m"] += tokens_added[place]
                            if trace not in transition_fitness[t]["underfed_traces"]:
                                transition_fitness[t]["underfed_traces"][trace] = list()
                            transition_fitness[t]["underfed_traces"][trace].append(current_event_map)
                    elif enable_pltr_fitness:
                        if trace not in transition_fitness[t]["fit_traces"]:
                            transition_fitness[t]["fit_traces"][trace] = list()
                        transition_fitness[t]["fit_traces"][trace].append(current_event_map)
                    c, cmap = get_consumed_tokens(t)
                    p, pmap = get_produced_tokens(t)
                    consumed = consumed + c
                    produced = produced + p
                    if enable_pltr_fitness:
                        for pl2 in cmap:
                            if pl2 in place_fitness:
                                place_fitness[pl2]["c"] += cmap[pl2] * trace_occurrences
                        for pl2 in pmap:
                            if pl2 in place_fitness:
                                place_fitness[pl2]["p"] += pmap[pl2] * trace_occurrences
                    if semantics.is_enabled(t, net, marking):
                        try:
                            if 'timestamp' not in trace[i]:
                                raise KeyError(f"Zeitstempel-Schlüssel 'timestamp' nicht im Ereignis gefunden: {trace[i]}")
                            timestamp = trace[i]['timestamp']
                        except KeyError as e:
                            print(f"Fehler beim Zugriff auf Zeitstempel in Spur {i}: {str(e)}")
                            timestamp = pd.Timestamp.now()  # Fallback-Wert
                        for place in marking:
                            place_token_timeline[place].append((timestamp, marking[place]))
                        time_diff = trace[i].get('time_diff', 0)
                        alert = trace[i].get(machine_alert_key, 0) if machine_alert_key else 0
                        for arc in t.out_arcs:
                            place = arc.target
                            if not np.isnan(time_diff):
                                place_time_diffs[place].append(time_diff)
                            place_alerts[place].append(alert)
                            if '0505' in trace[i][activity_key]:
                                tokens = marking[place]
                                max_tokens = max([t[1] for t in place_token_timeline[place]] + [1])
                                place_storage_levels[place].append(tokens / max_tokens)
                        marking = semantics.execute(t, net, marking)
                        act_trans.append(t)
                        vis_mark.append(marking)
                    if not is_initially_enabled and cleaning_token_flood:
                        new_marking_names = [x.name for x in list(marking.keys())]
                        new_marking_names_diff = [x for x in new_marking_names if x not in old_marking_names]
                        new_marking_names_inte = [x for x in new_marking_names if x in old_marking_names]
                        for p1 in new_marking_names_inte:
                            for p2 in new_marking_names_diff:
                                for comp in s_components:
                                    if p1 in comp and p2 in comp:
                                        place_to_delete = [place for place in list(marking.keys()) if place.name == p1]
                                        if len(place_to_delete) == 1:
                                            del marking[place_to_delete[0]]
                                            if not place_to_delete[0] in current_remaining_map:
                                                current_remaining_map[place_to_delete[0]] = 0
                                            current_remaining_map[place_to_delete[0]] = current_remaining_map[place_to_delete[0]] + 1
                else:
                    if not trace[i][activity_key] in notexisting_activities_in_model:
                        notexisting_activities_in_model[trace[i][activity_key]] = {}
                    notexisting_activities_in_model[trace[i][activity_key]][trace] = current_event_map
            del trace_activities[0]
            if len(trace_activities) < TechnicalParameters.MAX_POSTFIX_SUFFIX_LENGTH.value:
                activating_transition_index[str(trace_activities)] = {"index": len(act_trans), "marking": hash(marking)}
            if i > 0:
                activating_transition_interval.append([trace[i][activity_key], prev_len_activated_transitions, len(act_trans), trace[i - 1][activity_key]])
            else:
                activating_transition_interval.append([trace[i][activity_key], prev_len_activated_transitions, len(act_trans), ""])

    if try_to_reach_final_marking_through_hidden and not break_condition_final_marking(marking, final_marking):
        hidden_transitions = get_req_transitions_for_final_marking(marking, final_marking, places_shortest_path_by_hidden)
        if hidden_transitions:
            result = enable_hidden_transitions(net, marking, act_trans, visit_trans, vis_mark, hidden_transitions, None)
            print(f"enable_hidden_transitions result für Endmarkierung: {result}")
            if not isinstance(result, list) or len(result) != 4:
                print(f"Fehler: enable_hidden_transitions returned invalid object {result} für Endmarkierung")
                return [False, 0.0, act_trans, transitions_with_problems, marking, [], missing, consumed, 0, produced]
            [marking, act_trans, visit_trans, vis_mark] = result

    remaining = 0
    for place in marking:
        if place in final_marking:
            marking[place] = max(0, marking[place] - final_marking[place])
        if marking[place] > 0:
            remaining += marking[place]
            if enable_pltr_fitness:
                place_fitness[place]["r"] += marking[place]
                place_fitness[place]["overfed_traces"].add(trace)

    trace_is_fit = len(transitions_with_problems) == 0 and missing == 0 and remaining == 0
    trace_fitness = 0.5 * (1 - missing / consumed) + 0.5 * (1 - remaining / produced) if consumed > 0 and produced > 0 else 0.0

    enabled_trans_in_mark = list(semantics.enabled_transitions(net, marking))
    return [trace_is_fit, trace_fitness, act_trans, transitions_with_problems, marking, enabled_trans_in_mark, missing, consumed, remaining, produced]

class ApplyTraceTokenReplay:
    def __init__(self, trace, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness, transition_fitness, notexisting_activities_in_model, places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key="concept:name", reach_mark_through_hidden=True, stop_immediately_unfit=False, walk_through_hidden_trans=True, post_fix_caching=None, marking_to_activity_caching=None, is_reduction=False, thread_maximum_ex_time=TechnicalParameters.MAX_DEF_THR_EX_TIME.value, cleaning_token_flood=False, s_components=None, trace_occurrences=1, consider_activities_not_in_model_in_fitness=False, timestamp_key=xes_util.DEFAULT_TIMESTAMP_KEY, machine_alert_key=None, place_token_timeline=None, place_time_diffs=None, place_alerts=None, place_storage_levels=None, place_frequency_counts=None):
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
        self.try_to_reach_final_marking_through_hidden = reach_mark_through_hidden
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
        self.s_components = s_components
        self.trace_occurrences = trace_occurrences
        self.timestamp_key = timestamp_key
        self.machine_alert_key = machine_alert_key
        self.place_token_timeline = place_token_timeline
        self.place_time_diffs = place_time_diffs
        self.place_alerts = place_alerts
        self.place_storage_levels = place_storage_levels
        self.place_frequency_counts = place_frequency_counts

    def run(self):
        self.t_fit, self.t_value, self.act_trans, self.trans_probl, self.reached_marking, self.enabled_trans_in_mark, self.missing, self.consumed, self.remaining, self.produced = apply_trace(
            self.trace, self.net, self.initial_marking, self.final_marking, self.trans_map, self.enable_pltr_fitness,
            self.place_fitness, self.transition_fitness, self.notexisting_activities_in_model, self.places_shortest_path_by_hidden,
            self.consider_remaining_in_fitness, activity_key=self.activity_key, try_to_reach_final_marking_through_hidden=self.try_to_reach_final_marking_through_hidden,
            stop_immediately_unfit=self.stop_immediately_unfit, walk_through_hidden_trans=self.walk_through_hidden_trans,
            post_fix_caching=self.post_fix_caching, marking_to_activity_caching=self.marking_to_activity_caching,
            is_reduction=self.is_reduction, thread_maximum_ex_time=self.thread_maximum_ex_time, cleaning_token_flood=self.cleaning_token_flood,
            s_components=self.s_components, trace_occurrences=self.trace_occurrences, consider_activities_not_in_model_in_fitness=self.consider_activities_not_in_model_in_fitness,
            timestamp_key=self.timestamp_key, machine_alert_key=self.machine_alert_key, place_token_timeline=self.place_token_timeline,
            place_time_diffs=self.place_time_diffs, place_alerts=self.place_alerts, place_storage_levels=self.place_storage_levels,
            place_frequency_counts=self.place_frequency_counts)
        self.thread_is_alive = False

class PostFixCaching:
    def __init__(self):
        self.cache = {}

class MarkingToActivityCaching:
    def __init__(self):
        self.cache = {}

def get_variant_from_trace(trace, activity_key, disable_variants=False):
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
        "produced_tokens": int(t.produced)
    }
    if return_object_names:
        corr_value["activated_transitions_labels"] = [x.label for x in corr_value["activated_transitions"]]
        corr_value["activated_transitions"] = [x.name for x in corr_value["activated_transitions"]]
        corr_value["enabled_transitions_in_marking_labels"] = [x.label for x in corr_value["enabled_transitions_in_marking"]]
        corr_value["enabled_transitions_in_marking"] = [x.name for x in corr_value["enabled_transitions_in_marking"]]
        corr_value["transitions_with_problems"] = [x.name for x in corr_value["transitions_with_problems"]]
        corr_value["reached_marking"] = {x.name: y for x, y in corr_value["reached_marking"].items()}
    return corr_value

def apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=False, consider_remaining_in_fitness=False, activity_key="concept:name", reach_mark_through_hidden=True, stop_immediately_unfit=False, walk_through_hidden_trans=True, places_shortest_path_by_hidden=None, is_reduction=False, thread_maximum_ex_time=TechnicalParameters.MAX_DEF_THR_EX_TIME.value, cleaning_token_flood=False, disable_variants=False, return_object_names=False, show_progress_bar=True, consider_activities_not_in_model_in_fitness=False, case_id_key=constants.CASE_CONCEPT_NAME, timestamp_key=xes_util.DEFAULT_TIMESTAMP_KEY, machine_alert_key=None):
    post_fix_cache = PostFixCaching()
    marking_to_activity_cache = MarkingToActivityCaching()
    if places_shortest_path_by_hidden is None:
        places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(net, TechnicalParameters.MAX_REC_DEPTH.value)
    
    print("Validiere places_shortest_path_by_hidden...")
    for source, targets in places_shortest_path_by_hidden.items():
        for target, transitions in targets.items():
            for trans, weight in transitions:
                if trans is None or weight is None:
                    print(f"Warnung: Ungültiger Pfad in places_shortest_path_by_hidden: {source} -> {target}, Transition: {trans}, Gewicht: {weight}")
                    transitions[:] = [(t, w) for t, w in transitions if t is not None and w is not None]
    
    place_fitness_per_trace = {}
    transition_fitness_per_trace = {}
    aligned_traces = []
    place_token_timeline = defaultdict(list)
    place_time_diffs = defaultdict(list)
    place_alerts = defaultdict(list)
    place_storage_levels = defaultdict(list)
    place_frequency_counts = defaultdict(int)
    
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
    trans_map = {}
    for t in sorted(list(net.transitions), key=lambda x: x.name):
        trans_map[t.label] = t
    
    if pandas_utils.check_is_pandas_dataframe(log):
        traces = [(tuple(x), y) for y, x in log.groupby(case_id_key)[activity_key].agg(list).to_dict().items()]
        traces = [(traces[i][0], i) for i in range(len(traces))]
    else:
        traces = [(tuple(x[activity_key] for x in log[i]), i) for i in range(len(log))]
    
    variants = dict()
    for t in traces:
        if t[0] not in variants:
            variants[t[0]] = list()
        variants[t[0]].append(t[1])
    
    traces = [t[0] for t in traces]
    vc = [(k, v) for k, v in variants.items()]
    vc = list(sorted(vc, key=lambda x: (len(x[1]), x[0]), reverse=True))
    
    threads_results = {}
    progress = None
    
    if importlib.util.find_spec("tqdm") and show_progress_bar and len(variants) > 1:
        from tqdm.auto import tqdm
        if disable_variants and not pandas_utils.check_is_pandas_dataframe(log):
            progress = tqdm(total=len(traces), desc="replaying log with TBR-PRC, completed traces :: ")
        else:
            progress = tqdm(total=len(variants), desc="replaying log with TBR-PRC, completed traces :: ")
    
    for i in range(len(vc)):
        variant = vc[i][0]
        all_cases = vc[i][1]
        
        if disable_variants and not pandas_utils.check_is_pandas_dataframe(log):
            for j in range(len(all_cases)):
                case_position = all_cases[j]
                considered_case = log[case_position]
                t = ApplyTraceTokenReplay(considered_case, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model, places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key=activity_key, reach_mark_through_hidden=reach_mark_through_hidden, stop_immediately_unfit=stop_immediately_unfit, walk_through_hidden_trans=walk_through_hidden_trans, post_fix_caching=post_fix_cache, marking_to_activity_caching=marking_to_activity_cache, is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time, cleaning_token_flood=cleaning_token_flood, s_components=s_components, trace_occurrences=1, consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness, timestamp_key=timestamp_key, machine_alert_key=machine_alert_key, place_token_timeline=place_token_timeline, place_time_diffs=place_time_diffs, place_alerts=place_alerts, place_storage_levels=place_storage_levels, place_frequency_counts=place_frequency_counts)
                t.run()
                threads_results[case_position] = transcribe_result(t, return_object_names=return_object_names)
                if progress is not None:
                    progress.update()
        else:
            considered_case = variants_util.variant_to_trace(variant, parameters={constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_key})
            t = ApplyTraceTokenReplay(considered_case, net, initial_marking, final_marking, trans_map, enable_pltr_fitness, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model, places_shortest_path_by_hidden, consider_remaining_in_fitness, activity_key=activity_key, reach_mark_through_hidden=reach_mark_through_hidden, stop_immediately_unfit=stop_immediately_unfit, walk_through_hidden_trans=walk_through_hidden_trans, post_fix_caching=post_fix_cache, marking_to_activity_caching=marking_to_activity_cache, is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time, cleaning_token_flood=cleaning_token_flood, s_components=s_components, trace_occurrences=len(vc[i][1]), consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness, timestamp_key=timestamp_key, machine_alert_key=machine_alert_key, place_token_timeline=place_token_timeline, place_time_diffs=place_time_diffs, place_alerts=place_alerts, place_storage_levels=place_storage_levels, place_frequency_counts=place_frequency_counts)
            t.run()
            for j in range(len(all_cases)):
                case_position = all_cases[j]
                threads_results[case_position] = transcribe_result(t, return_object_names=return_object_names)
            if progress is not None:
                progress.update()
    
    for i in range(len(traces)):
        aligned_traces.append(threads_results[i])
    
    if progress is not None:
        progress.close()
    del progress
    
    if enable_pltr_fitness:
        return aligned_traces, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model, place_token_timeline, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts
    else:
        return aligned_traces, place_token_timeline, place_time_diffs, place_alerts, place_storage_levels, place_frequency_counts

def apply(log: EventLog, net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[PetriNet.Place, Dict[str, Any]], Dict[PetriNet.Transition, Dict[str, Any]], Dict[str, Dict], Dict[PetriNet.Place, List[Tuple[pd.Timestamp, int]]], Dict[PetriNet.Place, List[float]], Dict[PetriNet.Place, List[float]], Dict[PetriNet.Place, List[float]], Dict[PetriNet.Place, int]]:
    if parameters is None:
        parameters = {}
    
    enable_pltr_fitness = exec_utils.get_param_value(Parameters.ENABLE_PLTR_FITNESS, parameters, False)
    consider_remaining_in_fitness = exec_utils.get_param_value(Parameters.CONSIDER_REMAINING_IN_FITNESS, parameters, True)
    try_to_reach_final_marking_through_hidden = exec_utils.get_param_value(Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN, parameters, True)
    stop_immediately_unfit = exec_utils.get_param_value(Parameters.STOP_IMMEDIATELY_UNFIT, parameters, False)
    walk_through_hidden_trans = exec_utils.get_param_value(Parameters.WALK_THROUGH_HIDDEN_TRANS, parameters, True)
    is_reduction = exec_utils.get_param_value(Parameters.IS_REDUCTION, parameters, False)
    cleaning_token_flood = exec_utils.get_param_value(Parameters.CLEANING_TOKEN_FLOOD, parameters, False)
    disable_variants = exec_utils.get_param_value(Parameters.DISABLE_VARIANTS, parameters, enable_pltr_fitness)
    return_names = exec_utils.get_param_value(Parameters.RETURN_NAMES, parameters, False)
    thread_maximum_ex_time = exec_utils.get_param_value(Parameters.THREAD_MAX_EX_TIME, parameters, TechnicalParameters.MAX_DEF_THR_EX_TIME.value)
    places_shortest_path_by_hidden = exec_utils.get_param_value(Parameters.PLACES_SHORTEST_PATH_BY_HIDDEN, parameters, None)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY)
    machine_alert_key = exec_utils.get_param_value(Parameters.MACHINE_ALERT_KEY, parameters, None)
    consider_activities_not_in_model_in_fitness = exec_utils.get_param_value(Parameters.CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS, parameters, False)
    show_progress_bar = exec_utils.get_param_value(Parameters.SHOW_PROGRESS_BAR, parameters, constants.SHOW_PROGRESS_BAR)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)
    
    if not isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    
    return apply_log(log, net, initial_marking, final_marking, enable_pltr_fitness=enable_pltr_fitness, consider_remaining_in_fitness=consider_remaining_in_fitness, reach_mark_through_hidden=try_to_reach_final_marking_through_hidden, stop_immediately_unfit=stop_immediately_unfit, walk_through_hidden_trans=walk_through_hidden_trans, places_shortest_path_by_hidden=places_shortest_path_by_hidden, activity_key=activity_key, is_reduction=is_reduction, thread_maximum_ex_time=thread_maximum_ex_time, cleaning_token_flood=cleaning_token_flood, disable_variants=disable_variants, return_object_names=return_names, show_progress_bar=show_progress_bar, consider_activities_not_in_model_in_fitness=consider_activities_not_in_model_in_fitness, case_id_key=case_id_key, timestamp_key=timestamp_key, machine_alert_key=machine_alert_key)