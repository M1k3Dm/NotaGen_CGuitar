# © 2025 Mike Michail-Angelos Dimopoulos. Licensed under the MIT License.
# See LICENSE file for details.

import re
from constants import *
from collections import defaultdict, OrderedDict
from music21 import stream, note, chord, pitch, interval, converter
from playability import compute_base_score

"""
STANDARD ABC NOTATION !!!
"""

def split_abc_lines(abc_data):
    """
    Split the ABC file into header (metadata) and body (tunebody).
    """
    lines = abc_data.splitlines()

    # Look for a 'V:n' line without any extra tokens
    # to mark the start of the actual tunebody
    for i, line in enumerate(lines):
        if re.match(r'^V:\d+\s*$', line.strip()):
            return lines[:i], lines[i:]
 
    raise ValueError(f"No 'V:n' line found'; cannot split metadata and body.")

def get_guitar_groups(metadata_lines):
    """
    Parse the metadata header to group voices into guitars.
    Look for snm="Guit" to start a new guitar group, then 
    collect subsequent V:n lines into the same group.
    """
    groups = OrderedDict()
    current = None
    counter = 0

    for L in metadata_lines:
        m = re.match(r'^\s*V:(\d+)', L)
        if not m:
            continue
        vnum = int(m.group(1))
        # detect voice-name marker for a guitar
        snm = re.search(r'snm="([^"]+)"', L)
        if snm and 'Guit' in snm.group(1):
            counter += 1
            current = counter
            groups[current] = [vnum]
        else:
            # if within an existing guitar group, append the voice
            if current is not None:
                groups[current].append(vnum)

    return groups

def handle_pitch(p: pitch.Pitch, carried: dict):
    """
    Manage accidentals for a single pitch based on stored accidentals 
    within the same measure.
    """
    key = (p.step, p.octave)
    if p.accidental is not None:
        carried[key] = p.accidental
    else:
        if key in carried:
            p.accidental = carried[key]

def carry_accidentals_by_measure(score):
    """
    Apply measure-based accidental carrying for each part in the score.
    For each measure reset 'carried' accidentals and process all notes and chords.
    """
    for part in score.parts:
        for meas in part.getElementsByClass(stream.Measure):
            carried = {}  
            elems = meas.recurse().getElementsByClass((note.Note, chord.Chord))
            for el in elems:
                if isinstance(el, note.Note):
                    handle_pitch(el.pitch, carried)
                else:  
                    for p in el.pitches:
                        handle_pitch(p, carried)

def shift_pitch_to_range(p):
    """
    Try to transpose a pitch by whole octaves (±3 octaves max) so that
    its nameWithOctave (without '-') appears in NOTE_TO_POSITIONS.
    """
    key0 = p.nameWithOctave.replace('-', '')
    if key0 in NOTE_TO_POSITIONS:
        return p, 0

    shifts = [0]
    for i in range(1, 4):
        shifts += [ i, -i ]

    for s in shifts:
        candidate = p.transpose(interval.Interval(s * 12))
        key = candidate.nameWithOctave.replace('-', '')
        if key in NOTE_TO_POSITIONS:
            return candidate, s

    return p, 0

def build_timeline(score, voice_to_group):
    """
    Build a shared timeline of note start/end events for each guitar.
    """
    #score.show('text')

    # Select only the parts that hold music (have Measure objects)
    body_parts = [p for p in score.parts if p.getElementsByClass(stream.Measure)]

    # Shared events per guitar: { guitar_name: { offset: {start:[], end:[]} } }
    shared = defaultdict(lambda: defaultdict(lambda: {'start': [], 'end': []}))

    for voice_num, part in enumerate(body_parts, start=1):
        guitar = voice_to_group.get(voice_num)
        if not guitar:
            continue

        # Find both notes and chords
        for el in part.recurse().getElementsByClass((note.Note, chord.Chord)):
            # Compute absolute start and end times in quarterLength units
            start = float(el.getOffsetInHierarchy(score))
            dur   = float(el.duration.quarterLength)
            end   = start + dur

            # Extract pitch names
            if isinstance(el, chord.Chord):
                pitches = [p.nameWithOctave for p in el.pitches]
            else:
                pitches = [el.nameWithOctave]

            for p in pitches:
                p = p.replace('-', '')
                shared[guitar][start]['start'].append(p)
                shared[guitar][end]  ['end'].append(p)

    # Convert to OrderedDict per guitar, sorted by offset
    final = {}
    for guitar, events in shared.items():
        ord_ev = OrderedDict()
        for off in sorted(events.keys()):
            ord_ev[off] = events[off]
        final[guitar] = ord_ev

    return final

def get_active_events(per_tl):
    """
    per_tl: OrderedDict[offset → {'start':[…], 'end':[…]}]
    Returns: list of (offset, active_notes_list)
    """
    active = []
    events = []
    for offset, ev in per_tl.items():
        # Remove notes that end here
        for n in ev.get('end', []):
            if n in active:
                active.remove(n)
        # Add notes that start here
        for n in ev.get('start', []):
            active.append(n)
        # Append the full active chord at this offset
        events.append((offset, list(active)))
    return events

def get_timeline(abc_path):
    """
    Load an ABC file, map voices to guitar groups, parse into a Score,
    apply measure-based accidentals and an octave transpose, shift pitches
    into the allowed range, count total shifts, then build and return
    a timeline of note events along with shift count and metadata.
    """

    with open(abc_path, encoding='utf-8') as f:
        abc_data = f.read()

    # Split into metadata(header) and tunebody
    metadata, _ = split_abc_lines(abc_data)

    # Map each voice number to a guitar group
    groups = get_guitar_groups(metadata)
    voice_to_group = {}
    for gid, vs in groups.items():
        name = f'Guitar {gid}'
        for v in vs:
            voice_to_group[v] = name        

    # Parse ABC file with music21 into a Score object
    score = converter.parse(abc_path, format='abc')
    carry_accidentals_by_measure(score)
    score = score.transpose(interval.Interval('-P8'))

    total_shift_counter = 0
    for part in score.parts:
        for el in part.recurse().getElementsByClass((note.Note, chord.Chord)):
            if isinstance(el, note.Note):
                new_pitch, shift_count = shift_pitch_to_range(el.pitch)
                el.pitch = new_pitch
                total_shift_counter += abs(shift_count)
            else:
                new_pitches = []
                for p in el.pitches:
                    np, sc = shift_pitch_to_range(p)
                    new_pitches.append(np)
                    total_shift_counter += abs(sc)
                el.pitches = new_pitches

    # Build the timeline of note events
    timeline = build_timeline(score, voice_to_group)
    return timeline, total_shift_counter, metadata

#--- FOR DEBUG ---#

def print_timeline(timeline):
    """
    Print the timeline in a compact, one-line-per-offset format:
      offset │ start: [..]  end: [..] 
    """
    for guitar, events in timeline.items():
        print(f"\n===== {guitar} TIMELINE =====")
        for off in sorted(events.keys()):
            starts = ', '.join(events[off]['start']) or '-'
            ends   = ', '.join(events[off]['end'])   or '-'
            print(f"{off:>5.2f} │ start: [{starts}]  end: [{ends}]")

def print_candidates(candidates_by_guitar):
    """
    Print candidate combos in two columns: OFFSET and Candidates.
    """
    for guitar, cand_list in candidates_by_guitar.items():
        print(f"\n===== {guitar} CANDIDATES =====")
        for offset, combos in cand_list:
            off_str = f"{offset:5.2f}"
            if not combos:
                line = "(no candidates)"
            else:
                combo_strs = [
                    "(" + " ".join(f"{s}:{f}" for s, f in combo) + ")"
                    for combo in combos
                ]
                line = " OR ".join(combo_strs)
            print(f"{off_str}  {line}")

def print_optimal_sequences(candidates_by_guitar, optimal_by_guitar):
    """
    Print optimal fingering sequence per guitar.
    """
    for guitar, cand_list in candidates_by_guitar.items():
        print(f"\n===== {guitar} Optimal Fingering =====")
        filtered = [(off, combos) for off, combos in cand_list if combos]
        best_seq = optimal_by_guitar.get(guitar, [])
        for (offset, _), combo in zip(filtered, best_seq):
            combo_str = "(" + " ".join(f"{s}:{f}" for s, f in combo) + ")"
            print(f"{offset:6.2f}  {combo_str}")