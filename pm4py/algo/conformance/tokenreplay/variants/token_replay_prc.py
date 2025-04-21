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
    import pandas as pd  # Für CSV-Ausgabe
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

    # Speichere Debugging-Informationen als CSV
    if debug_data:
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv("../Ergebnisse_PRC/place_capacity_debug.csv", index=False)
        print(f"Debugging data saved to '../Ergebnisse_PRC/place_capacity_debug.csv'")

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
            place_max_capacities]