# Basic
import os
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from itertools import chain
from scipy.spatial.distance import euclidean

# For
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helpers import read_json
from eyemovements.eyemovements_utils import (get_movement_indexes, GazeState,
                                 get_amplitude_and_angle, get_path_and_centroid)


def estimate_quality(data: List[pd.DataFrame]):
    movements = [df.movements.values for df in data]
    gaze_data = [df[['gaze_X', 'gaze_Y']].values for df in data]
    velocity = [df['velocity_sqrt'].values for df in data]
    # All stimulus movements - are smooth persuites
    stimulus_movements = [np.full_like(df.movements.values, 3) for df in data]
    stimulus_data = [df[['stim_X', 'stim_Y']].values for df in data]
    stimulus_velocity = [df[['stimulus_velocity']].values for df in data]

    metrics = {
        # Saccades
        "ANS": average_number_of_saccades(movements, to_average=True),
        "ASA": average_saccades_amplitude(movements, gaze_data, to_average=True),
        "ASM": average_saccades_magnitude(movements, gaze_data, to_average=True),
        # SP
        "ANSP": average_number_of_sp(movements, to_average=True),
        "PQlS": PQlS(stimulus_movements, stimulus_velocity,
                     movements, velocity,
                     gaze_data, gaze_data, to_average=True),
        "PQnS": PQnS(stimulus_movements, movements,
                     gaze_data, stimulus_data, to_average=True),
        "MisFix": MisFix(stimulus_movements, movements,
                         gaze_data, to_average=True)
    }
    return metrics

def visualize_and_save(data: pd.DataFrame, fn: str,
                       x_col: str="x", y_col: str="y",
                       time_col: str='timestamps',
                       color: str="movement_name"):
    assert ((x_col in data.columns) and (y_col in data.columns)
            and (color in data.columns) and (time_col in data.columns))

    color_mapping = dict(read_json(os.path.join(sys.path[0], "settings", "color_mappings.json")))
    data['color'] = data[color].apply(lambda x: color_mapping.get(x, "black"))

    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.06,
        specs=[[{}, {}],
               [{"colspan": 2}, None]],
        row_heights=[0.4, 0.6],
        subplot_titles=("Gaze X Axis", "Gaze Y Axis", "Gaze X-Y Axis")
    )

    min_ts = np.min(data[time_col])
    for movement_type, df in data.groupby(by=color):
        fig.add_trace(go.Scatter(x=df[time_col] - min_ts,
                                 y=df[x_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=df[time_col] - min_ts,
                                 y=df[y_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=df[x_col],
                                 y=df[y_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type), row=2, col=1)

    fig.update_traces(mode='markers', marker_line_width=0.1, marker_size=4)
    fig.update_layout(height=800, width=1000,
                      title_text="Eyemovements classified")

    fig.update_layout(legend_title_text='Eyemovements Types',
                      legend=dict(font=dict(family="Arial", size=12)))
    fig.update_layout(showlegend=True)
    # fig = px.scatter(data, color=color, y=y, x=x)
    plotly.offline.plot(fig, filename='../output/'+ fn + '.html')


# --------------- Fixations metrics ------------------------------
def average_number_of_fixations(movements_dataset: List[np.ndarray], to_average: bool):
    """
    Count average number of fixations.
    """
    anf_list = [len(get_movement_indexes(movements, GazeState.fixation)) for movements in movements_dataset]
    if to_average:
        return np.mean(anf_list)
    else:
        return anf_list


def average_fixations_duration(movements_dataset: List[np.ndarray],
                               timestamps_dataset: List[np.ndarray],
                               to_average: bool):
    """
    Count average duration of fixations.
    """
    afd_list = list(chain.from_iterable([[timestamps[fix[-1]] - timestamps[fix[0]]
                                        for fix in get_movement_indexes(movements, GazeState.fixation)]
                                        for movements, timestamps in zip(movements_dataset, timestamps_dataset)]))
    if to_average:
        return np.mean(afd_list)
    else:
        return afd_list


def FQnS(stimulus_dataset: List[np.ndarray],
         gaze_dataset: List[np.ndarray],
         coords_dataset: List[np.ndarray],
         amplitude_coef: float):
    """
    Fixation Quantitative Score (FQnS) compares the amount of
    detected fixational behavior to the amount of
    fixational behavior encoded in the stimuli.
    """
    fqns_estims = []
    for stimulus, gaze, coords in tqdm(zip(stimulus_dataset, gaze_dataset, coords_dataset),
                                                total=len(stimulus_dataset)):
        # Stimulus
        stimulus_fixes = get_movement_indexes(stimulus, GazeState.fixation)
        stimulus_fix_points_num = len([s for s in stimulus if s == GazeState.fixation])
        stimulus_fix_centroids = [get_path_and_centroid(coords[fix])[1:] for fix in stimulus_fixes]
        stimulus_prev_sac = [get_prev_saccade(stimulus, fix[0]) for fix in stimulus_fixes]
        default_sac_amplitude = np.mean([get_amplitude_and_angle(coords[prev_sac])[0]
                                         for prev_sac in stimulus_prev_sac if len(prev_sac) > 0])

        fixations_detected_cnt = 0
        for fix_idx, centroid, prev_sac in zip(stimulus_fixes, stimulus_fix_centroids, stimulus_prev_sac):
            if len(prev_sac) > 0:
                sac_amplitude = get_amplitude_and_angle(coords[prev_sac])[0]
            else:
                sac_amplitude = default_sac_amplitude

            for p_idx in fix_idx:
                if gaze[p_idx] == GazeState.fixation:
                    detected_fix = get_movement_for_index(p_idx, gaze)
                    if len(detected_fix) > 0:
                        (xc, yc) = get_path_and_centroid(coords[detected_fix])[1:]
                        (xs, ys) = coords[p_idx]
                        if euclidean((xs, ys), (xc, yc)) <= amplitude_coef * sac_amplitude:
                            fixations_detected_cnt += 1

        print(f"[INFO-METRICS]: Fixations_detected_cnt: {fixations_detected_cnt}")
        fqns = 100 * (fixations_detected_cnt / stimulus_fix_points_num)
        fqns_estims.append(fqns)
    return fqns_estims


def FQlS(stimulus_dataset: List[np.ndarray],
         gaze_dataset: List[np.ndarray],
         coords_dataset: List[np.ndarray], to_average: bool):
    """
    The Fixation Qualitative Score (FQlS) compares the spatial proximity
    of the classified eye fixation signal to the presented stimulus signal,
    therefore indicating the positional accuracy or error of the classified fixations
    """
    fqls_estims = []
    for stimulus, gaze, coords in tqdm(zip(stimulus_dataset, gaze_dataset, coords_dataset)):
        # Stimulus
        stimulus_fixes = get_movement_indexes(stimulus, GazeState.fixation)
        # Gaze
        gaze_fixes = get_movement_indexes(gaze, GazeState.fixation)
        gaze_fix_centroids = [get_path_and_centroid(coords[fix])[1:] for fix in gaze_fixes]

        fixations_dists_list = []
        for fix_idx in stimulus_fixes:
            for p_idx in fix_idx:
                if gaze[p_idx] == GazeState.fixation:
                    detected_fix = get_movement_for_index(p_idx, gaze)
                    if len(detected_fix) > 0:
                        (xc, yc) = get_path_and_centroid(coords[detected_fix])[1:]
                        (xs, ys) = coords[p_idx]
                        fixations_dists_list.append(euclidean((xs, ys), (xc, yc)))
        if to_average:
            fqls_estims.append(np.mean(fixations_dists_list))
        else:
            fqls_estims.append(fixations_dists_list)
    return fqls_estims


# ----------------------- Saccades metrics -----------------------

def average_number_of_saccades(movements_dataset: List[np.ndarray],
                               to_average: bool):
    """
    Count average number of saccades.
    """
    ans_list = [len(get_movement_indexes(movements, GazeState.saccade)) for movements in movements_dataset]
    if to_average:
        return np.mean(ans_list)
    else:
        return ans_list


def average_saccades_amplitude(movements_dataset: List[np.ndarray],
                               gaze_dataset: List[np.ndarray], to_average: bool):
    """
    Count average amplitude of saccades.
    """
    asa_list = (list([get_amplitude_and_angle(gaze[sac])[0]
                         for movements, gaze in zip(movements_dataset, gaze_dataset)
                         for sac in get_movement_indexes(movements, GazeState.saccade)]))
    if to_average:
        return np.mean(asa_list)
    else:
        return asa_list


def average_saccades_magnitude(movements_dataset: List[np.ndarray],
                               gaze_dataset: List[np.ndarray], to_average: bool):
    """
    Count average magnitude of saccades.
    """
    asm_list = (list([get_amplitude_and_angle(gaze[sac])[1]
                         for movements, gaze in zip(movements_dataset, gaze_dataset)
                         for sac in get_movement_indexes(movements, GazeState.saccade)]))
    if to_average:
        return np.mean(asm_list)
    else:
        return asm_list



def SQnS(stimulus_dataset: List[np.ndarray],
         gaze_dataset: List[np.ndarray],
         coords_dataset: List[np.ndarray]):
    """
    The Saccade Quantitative Score (SQnS) represents
    the amount of classified saccadic behavior given
    the amount of saccadic behavior encoded in the stimuli.
    """
    sqns_estims = []
    for stimulus, gaze, coords in tqdm(zip(stimulus_dataset, gaze_dataset, coords_dataset)):
        # Stimulus
        stimulus_sac = get_movement_indexes(stimulus, GazeState.saccade)
        stimulus_ampl_sum = (np.sum([get_amplitude_and_angle(coords[sac])[0]
                                     for sac in stimulus_sac]) + sys.float_info.epsilon)
        # Gaze
        gaze_sac = get_movement_indexes(gaze, GazeState.saccade)
        gaze_ampl_sum = np.sum([get_amplitude_and_angle(coords[sac])[0] for sac in gaze_sac])

        sqns = 100 * (gaze_ampl_sum / stimulus_ampl_sum)
        sqns_estims.append(sqns)
        # print(f"[INFO-METRICS]: Saccade Quantitative Score: {sqns}")
    return sqns_estims

# -------------------- Smooth Persuite ----------------

def average_number_of_sp(movements_dataset: List[np.ndarray], to_average: bool):
    """
    Count average number of smooth persuit.
    """
    ans_list = [len(get_movement_indexes(movements, GazeState.sp)) for movements in movements_dataset]
    if to_average:
        return np.mean(ans_list)
    else:
        return ans_list


def PQlS(stimulus_dataset: List[np.ndarray], stim_velocity: List[np.ndarray],
         gaze_dataset: List[np.ndarray], gaze_velocity: List[np.ndarray],
         stim_coords_dataset: List[np.ndarray], gaze_coords_dataset: List[np.ndarray],
         to_average: bool):
    """
    The intuitive idea behind the smooth pursuit qualitative scores (PQlS)
    is to compare the proximity of the detected SP signal with the signal
    presented in the stimuli.
    Two scores are indicative of positional (PQlS_P) and velocity (PQlS_V) accuracy.
    """
    pqls_p_estims = []
    pqls_v_estims = []
    for stimulus, stim_vel, gaze, gaze_vel, scoords, gcoords in tqdm(zip(stimulus_dataset, stim_velocity,
                                                                                  gaze_dataset, gaze_velocity,
                                                                                  stim_coords_dataset,
                                                                                  gaze_coords_dataset)):
        # Stimulus sp
        stimulus_sp = get_movement_indexes(stimulus, GazeState.sp)
        # Gaze
        gaze_sp = get_movement_indexes(gaze, GazeState.sp)

        sp_detected_cnt = 0
        v_diff = 0
        p_diff = 0
        for sp_idx in stimulus_sp:
            for p_idx in sp_idx:
                if gaze[p_idx] == GazeState.sp:
                    sp_detected_cnt += 1
                    v_diff += np.abs(stim_vel[p_idx] - gaze_vel[p_idx])
                    p_diff += euclidean(scoords[p_idx], gcoords[p_idx])
        if sp_detected_cnt > 0:
            pqls_p = p_diff / sp_detected_cnt
            pqls_v = v_diff / sp_detected_cnt
        else:
            pqls_p = np.inf
            pqls_v = np.inf
        # print(f"[INFO-METRICS]: PQlS_p: {pqls_p}, PQlS_v: {pqls_v}")
        pqls_p_estims.append(pqls_p)
        pqls_v_estims.append(pqls_v)
    if to_average:
        return np.mean(pqls_p_estims), np.mean(pqls_v_estims)
    else:
        return pqls_p_estims, pqls_v_estims


def PQnS(stimulus_dataset: List[np.ndarray], gaze_dataset: List[np.ndarray],
         stim_coords_dataset: List[np.ndarray], gaze_coords_dataset: List[np.ndarray],
         to_average: bool):
    """
    The smooth pursuit quantitative score (PQnS) measures
    the amount of detected SP behavior given
    the SP behavior encoded in the stimuli.
    """
    pqns_estims = []
    for stimulus, gaze, scoords, gcoords in tqdm(zip(stimulus_dataset, gaze_dataset,
                                                              stim_coords_dataset, gaze_coords_dataset)):
        # Stimulus sp
        stimulus_sp = get_movement_indexes(stimulus, GazeState.sp)
        stim_paths = [get_path_and_centroid(scoords[sp])[0] for sp in stimulus_sp]
        # Gaze
        gaze_sp = get_movement_indexes(gaze, GazeState.sp)
        gaze_paths = [get_path_and_centroid(scoords[sp])[0] for sp in gaze_sp]

        pqns = 100 * (np.sum(gaze_paths) / np.sum(stim_paths))
        # print(f"[INFO-METRICS]: Gaze paths: {np.sum(gaze_paths)}, Stim paths: {np.sum(stim_paths)}, PQnS: {pqns}")
        pqns_estims.append(pqns)

    if to_average:
        return np.mean(pqns_estims)
    else:
        return pqns_estims


def MisFix(stimulus_dataset: List[np.ndarray], gaze_dataset: List[np.ndarray],
           coords_dataset: List[np.ndarray], to_average: bool):
    """
    Misclassification error of the SP can be determined during a fixation stimulus,
    when correct classification is most challenging.
    """
    misfix_estims = []
    for stimulus, gaze, coords in tqdm(zip(stimulus_dataset, gaze_dataset, coords_dataset)):
        # Stimulus sp
        stimulus_all_fix = list(chain.from_iterable(get_movement_indexes(stimulus, GazeState.fixation)))
        stimulus_fix_points_num = len([s for s in stimulus if s == GazeState.fixation])
        # Gaze
        gaze_sp = list(chain.from_iterable(get_movement_indexes(gaze, GazeState.sp)))

        mis_class = len([stim_fix_p for stim_fix_p in stimulus_all_fix if stim_fix_p in gaze_sp])
        misfix = 100 * (mis_class / (stimulus_fix_points_num + sys.float_info.epsilon))
        misfix_estims.append(misfix)
        # print(f"[INFO-METRICS]: Misfix: {misfix}, fix classfied as sp: {mis_class} from total {stimulus_fix_points_num} fix points.")

    if to_average:
        return np.mean(misfix_estims)
    else:
        return misfix_estims

# --------------------- Utils ---------------------

def get_prev_saccade(movements: np.ndarray, fix_start_index: int):
    i = fix_start_index - 1
    saccade = []
    while i >= 0:
        # yet another point to saccade
        if movements[i] == GazeState.saccade:
            saccade.append(i)
            i -= 1
        # found saccade ended, return
        elif (movements[i] != GazeState.saccade) and (len(saccade) > 0):
            return saccade
        else:
            i -= 1
    print("[INFO]: Not saccades found!")
    return saccade

def get_closest_centroid(centroid: List[float],
                         centroids_list: List[List[float]]):
    dists = [euclidean(centroid, cc) for cc in centroids_list]
    return np.argmax(dists), np.max(dists)


def get_movement_for_index(index: Tuple[float], movements: np.ndarray):
    moves = get_movement_indexes(movements, movements[index])
    for m in moves:
        if index in m:
            return m
    return []