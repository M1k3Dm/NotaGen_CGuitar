# © 2025 Mike Michail-Angelos Dimopoulos. Licensed under the MIT License.
# See LICENSE file for details.

import argparse
import math
from collections import Counter
from utils import *
from constants import *
from pathlib import Path
from itertools import product

D2_6TH_STRING = False

def generate_candidates_for_event(start_list):
    """
    Generate all valid string–fret combinations for a given list of pitch 
    names, applying guitar-specific constraints (no duplicate strings, max span,
    max fingers, and barre chord allowance).
    """
    global D2_6TH_STRING

    pos_lists = []
    for p in start_list:
        base = NOTE_TO_POSITIONS.get(p, [])
        if D2_6TH_STRING:
            # If 6th-string is tuned down (D2), shift any 6th-string fret up by 2
            shifted = [(s, f+2 if s == 6 else f) for s, f in base]
            pos_lists.append(shifted)
        else:
            pos_lists.append(base)

    # Compute Cartesian product: every possible assignment of one position per pitch
    raw_combos = product(*pos_lists)

    valid = []
    for combo in raw_combos:
        strings = [s for s,f in combo]
        frets   = [f for s,f in combo]

        # No duplicate strings (can't play two notes on the same string)
        if len(set(strings)) != len(combo): continue
        # Span constraint (frets must fit within MAX_SPAN), ignore open strings (fret 0)
        non_open = [f for f in frets if f > 0]
        span = max(non_open) - min(non_open) if non_open else 0
        if span > MAX_SPAN:
            continue

        # Max fingers: 4, open strings (fret 0) don't count against finger limit
        n = len(combo)
        open_strings_count = frets.count(0)
        fret_notes = n - open_strings_count

        if fret_notes <= MAX_FINGERS:
            valid.append(combo)
        else:
            # Barre allowance if enough notes share a single fret 
            count = Counter(frets)
            for fret, c in count.items():
                if fret > 0 and c >= BARRE_MIN_NOTES:
                    if min(frets) < fret:
                        continue
                    valid.append(combo)
                    break        
    return valid

def find_candidates(timeline):
    """
    For each guitar part in the timeline, generate valid fingering candidates
    at each time offset based on active note events.
    """
    result = {}
    # Iterate over each guitar and its timeline of note events
    for guitar, per_tl in timeline.items():
        cand_list = []
        # per_tl is an OrderedDict mapping offset to {'start': ..., 'end': ...}
        active_events = get_active_events(per_tl)
        for offset, active_notes in active_events:
            if not active_notes:
                cand_list.append((offset, []))
            else:
                # Generate candidates only for the start_list
                combos = generate_candidates_for_event(active_notes)       
                cand_list.append((offset, combos))

        result[guitar] = cand_list
    return result

def compute_base_score(combo, w_max_fret=0.5, w_span=1, w_opstrings=1, w_barre=2):
    """
    Compute a base score for a single fingering combination.
    combo: tuple of (string, fret) pairs
    w_max_fret: weight for using higher frets
    w_span: weight for the fret span (max - min)
    w_opstrings: reward weight for open strings (subtracted)
    w_barre: reward weight for using a barre (subtracted)
    """
    frets = [f for _, f in combo]
    # Open Strings
    open_strings_count = frets.count(0)
    # Fret span
    span = max(frets) - min(frets)
    # Barre 
    has_barre = any(f > 0 and frets.count(f) >= BARRE_MIN_NOTES for f in set(frets))
    # Maximum fret
    max_fret = max(frets)
    
    score = (
        w_opstrings * open_strings_count
        + w_barre * int(has_barre)
        - w_max_fret * max_fret
        - w_span * span 
    )
    return score
    

def compute_transition_score(prev_combo, curr_combo, w_jump=1.0, w_common=1.0, w_avg_move=0.2):
    """
    Compute a transition score between two successive combos.
    prev_combo: tuple of (string, fret) pairs for previous chord/note
    curr_combo: same for current chord/note
    w_jump: weight for overall fretboard jump
    w_common: reward weight for reusing exact same string–fret pairs
    w_avg_move: weight for average finger movement on shared strings
    """
    prev_map = {s: f for s, f in prev_combo}
    curr_map = {s: f for s, f in curr_combo}

    # Jump cost
    prev_max = max(prev_map.values())
    curr_max = max(curr_map.values())
    jump = abs(curr_max - prev_max)

    # Reward for exact matches
    common = set(prev_combo) & set(curr_combo)

    # Fret movement on shared strings
    shared = set(prev_map) & set(curr_map)
    if shared:
        moves = [abs(prev_map[s] - curr_map[s]) for s in shared]
        avg_move = sum(moves) / len(moves)
    else:
        avg_move = 0
    
    score = (
        w_common * len(common)
        - w_jump * jump 
        - w_avg_move * avg_move
    )
        
    return score

def find_optimal_sequence(candidates):
    """
    Find the optimal sequence of combos that maximizes the sum
    of base and transition scores. Returns a list of chosen combos
    in chronological order.
    """
    
    if isinstance(candidates, dict):
        items = sorted(candidates.items(), key=lambda x: x[0])
        candidates = items

    candidates = [(off, cmbs) for off, cmbs in candidates if cmbs]
    T = len(candidates)
    # dp[i] maps each combo at time i to its maximal score
    dp = [dict() for _ in range(T)]
    # bp[i] maps each combo at time i to the best previous combo for backtracking
    bp = [dict() for _ in range(T)]

    # Initialize the first timestep with base scores
    _, first_combos = candidates[0]
    for c in first_combos:
        dp[0][c] = compute_base_score(c)
        bp[0][c] = None # no previous combo

    # Fill DP table for each subsequent timestep
    for i in range(1, T):
        _, combos = candidates[i]
        for combo in combos:
            best_score = -math.inf
            best_prev = None
            # Try every possible previous combo
            for prev_combo, prev_score in dp[i-1].items():
                # Score for transitioning from prev_combo to combo
                trans_score = compute_transition_score(prev_combo, combo)
                # Total score if we choose this path
                total_score = prev_score + trans_score + compute_base_score(combo)
                if total_score > best_score:
                    best_score = total_score
                    best_prev = prev_combo 
            dp[i][combo] = best_score
            bp[i][combo] = best_prev 

    # Pick the ending combo with the highest total score
    last_dp = dp[T-1]
    end_combo = max(last_dp, key=lambda c: last_dp[c])

    # Backtrack to reconstruct the optimal sequence
    sequence = [end_combo]
    for i in range(T-1, 0, -1):
        sequence.append(bp[i][sequence[-1]])
    sequence.reverse() # chronological order
    total_score = last_dp[end_combo]
    return sequence        

def get_guitars_optimal_sequences(abc_path):
    """
    Parse an ABC file, detect tuning shifts, generate fingering candidates
    and compute the optimal fingering sequence for each guitar.
    """
    global D2_6TH_STRING

    timeline, octave_shift_counter, _ = get_timeline(abc_path)

    # FOR DEBUG:
    #print_timeline(timeline)

    for guitar, events in timeline.items():
        for offset, ev in events.items():
            if any(n in ('D2', 'D#2') for n in ev['start']):
                D2_6TH_STRING = True
                break
        if D2_6TH_STRING:
            break

    candidates = find_candidates(timeline)
    # FOR DEBUG:
    #print_candidates(candidates)

    optimal_by_guitar = {
        g: find_optimal_sequence(cands)
            for g, cands in candidates.items()
    }

    # FOR DEBUG:
    #print_optimal_sequences(candidates, optimal_by_guitar)
    
    return optimal_by_guitar, octave_shift_counter

def compute_playability_score(optimal_sequences, octave_shift_counter):
    """
    Calculate an overall playability score based on optimal fingering sequences
    and the number of octave shifts applied.
    """
    total_score = 0.0
    total_transitions = 0

    # For each guitar's optimal sequence
    for seq in optimal_sequences.values():
        if not seq:
            continue

        # Add base score of the first chord/note
        total_score += compute_base_score(seq[0])
        
        # For each consecutive pair, add transition and base scores
        for prev, curr in zip(seq, seq[1:]):
            total_score += compute_transition_score(prev, curr)
            total_score += compute_base_score(curr)
            total_transitions += 1

    if total_transitions == 0:
        return 0.0

    # 3) Normalized average score per event, minus octave shifts penalty
    avg_score = total_score / total_transitions
    penalty = octave_shift_counter * 0.25
    return avg_score - penalty

def min_max_normalize(x, min_s, max_s):
    if max_s == min_s:
        return 1.0  
    return (x - min_s) / (max_s - min_s)

def main():
    p = argparse.ArgumentParser(
        description="Playability score standard abc guitar files."
    )
    p.add_argument('abc_dir', help='.abc dir')
    args = p.parse_args()

    abc_path = Path(args.abc_dir)
    if not abc_path.is_dir():
        print(f"{abc_path} not a folder.")
        exit(1)

    abc_files = sorted(abc_path.glob('*.abc'))

    scores = []
    for abc_file in abc_files:
        print(f"Preprocessing: {abc_file.name}")
        optimal_sequences, octave_shift_counter = get_guitars_optimal_sequences(str(abc_file))
        score = compute_playability_score(optimal_sequences, octave_shift_counter)
        scores.append(score)
        print(f"{abc_file.name} | Score: {score:.3f}")

    # Min–Max normalization
    if scores:
        min_s = min(scores)
        max_s = max(scores)
        normalized_scores = [
            min_max_normalize(x, min_s, max_s) for x in scores
        ]
        avg_score = sum(normalized_scores) / len(normalized_scores)
        print(f"\nAverage Playability Score: {avg_score:.3f}")

if __name__ == '__main__':
    main()