
from config import DEVICE
from models import ClinicalECGClassifier
from utils import test_dataset, all_labels
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

device = torch.device(DEVICE)
model = ClinicalECGClassifier(num_classes=len(all_labels)).to(device)

def add_ecg_grid(ax, duration_sec=10, mv_range=2, mm_per_sec=20, mm_per_mv=10):
    small_box_time = 1 / mm_per_sec  
    small_box_mv = 1 / mm_per_mv    

    big_box_time = 5 * small_box_time  
    big_box_mv = 5 * small_box_mv   

    ymin, ymax = ax.get_ylim()
    y_small = np.arange(np.floor(ymin / small_box_mv) * small_box_mv,
                        np.ceil(ymax / small_box_mv) * small_box_mv,
                        small_box_mv)
    for y in y_small:
        ax.axhline(y, color='#ed70bc', linewidth=0.3, zorder=0)

    y_big = np.arange(np.floor(ymin / big_box_mv) * big_box_mv,
                      np.ceil(ymax / big_box_mv) * big_box_mv,
                      big_box_mv)
    for y in y_big:
        ax.axhline(y, color='#b51777', linewidth=0.6, zorder=0)

    xmax = duration_sec
    x_small = np.arange(0, xmax, small_box_time)
    for x in x_small:
        ax.axvline(x, color='#ed70bc', linewidth=0.3, zorder=0)

    x_big = np.arange(0, xmax, big_box_time)
    for x in x_big:
        ax.axvline(x, color='#b51777', linewidth=0.6, zorder=0)


class ECGClinicalAnalyzer:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = 1000
        self.scaling_factor = 500
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        self.clinical_criteria = {
            'AFIB': {
                'name': 'Atrial Fibrillation',
                'criteria': [
                    {'type': 'rr_irregularity', 'threshold': 0.3, 'description': 'Irregular R-R intervals'},
                    {'type': 'p_wave_absent', 'threshold': 0.7, 'description': 'Absent P waves'},
                    {'type': 'fibrillatory_waves', 'threshold': 0.1, 'description': 'Fibrillatory waves'},
                    {'type': 'ventricular_rate', 'range': [60, 180], 'description': 'Heart rate 60-180 bpm'}
                ]
            },
            'AFL': {
                'name': 'Atrial Flutter',
                'criteria': [
                    {'type': 'flutter_waves', 'rate': [240, 350], 'description': 'Flutter waves 240-350 bpm'},
                    {'type': 'sawtooth_pattern', 'leads': ['II', 'III', 'aVF'], 'description': 'Sawtooth pattern in inferior leads'},
                    {'type': 'av_conduction', 'ratio': [2, 3, 4], 'description': 'Fixed AV conduction ratio'}
                ]
            },
            'AFLU': {
                'name': 'Atrial Flutter (variant)',
                'criteria': [
                    {'type': 'flutter_waves', 'rate': [240, 350], 'description': 'Flutter waves 240-350 bpm'},
                    {'type': 'variable_conduction', 'description': 'Variable AV conduction'}
                ]
            },
            'SVTAC': {
                'name': 'Supraventricular Tachycardia',
                'criteria': [
                    {'type': 'heart_rate', 'range': [150, 250], 'description': 'Heart rate 150-250 bpm'},
                    {'type': 'narrow_qrs', 'threshold': 0.12, 'description': 'QRS < 120ms'},
                    {'type': 'regular_rhythm', 'description': 'Regular rhythm'}
                ]
            },
            'STACH': {
                'name': 'Sinus Tachycardia',
                'criteria': [
                    {'type': 'heart_rate', 'range': [100, 150], 'description': 'Heart rate 100-150 bpm'},
                    {'type': 'normal_p_waves', 'description': 'Normal P waves'},
                    {'type': 'regular_rhythm', 'description': 'Regular sinus rhythm'}
                ]
            },
            'SBRAD': {
                'name': 'Sinus Bradycardia',
                'criteria': [
                    {'type': 'heart_rate', 'range': [40, 60], 'description': 'Heart rate < 60 bpm'},
                    {'type': 'normal_p_waves', 'description': 'Normal P waves'},
                    {'type': 'regular_rhythm', 'description': 'Regular sinus rhythm'}
                ]
            },
            'SARRH': {
                'name': 'Sinus Arrhythmia',
                'criteria': [
                    {'type': 'rr_variability', 'threshold': 0.16, 'description': 'R-R interval variability > 0.16s'},
                    {'type': 'normal_p_waves', 'description': 'Normal P waves'},
                    {'type': 'respiratory_variation', 'description': 'Respiratory variation'}
                ]
            },
            'PACE': {
                'name': 'Paced Rhythm',
                'criteria': [
                    {'type': 'pacing_spikes', 'description': 'Pacing spikes present'},
                    {'type': 'wide_qrs', 'threshold': 0.12, 'description': 'Wide QRS > 120ms'},
                    {'type': 'av_pacing', 'description': 'Atrial/Ventricular pacing'}
                ]
            },
            'BIGU': {
                'name': 'Bigeminy',
                'criteria': [
                    {'type': 'alternating_pattern', 'description': 'Alternating normal-PVC pattern'},
                    {'type': 'premature_beats', 'description': 'Every other beat is premature'}
                ]
            },
            'TRIGU': {
                'name': 'Trigeminy',
                'criteria': [
                    {'type': 'every_third_beat', 'description': 'Every third beat is premature'},
                    {'type': 'regular_pattern', 'description': 'Regular trigeminal pattern'}
                ]
            },

            # CONDUCTION ABNORMALITIES
            'IAVB': {
                'name': '1st Degree AV Block',
                'criteria': [
                    {'type': 'pr_interval', 'threshold': 0.20, 'description': 'PR interval > 200ms'},
                    {'type': 'all_p_conducted', 'description': 'All P waves conducted'}
                ]
            },
            'IIAVB': {
                'name': '2nd Degree AV Block',
                'criteria': [
                    {'type': 'dropped_qrs', 'description': 'Some P waves not conducted'},
                    {'type': 'wenckebach_pattern', 'description': 'Progressive PR prolongation'}
                ]
            },
            'IIAVBII': {
                'name': '2nd Degree AV Block Type II',
                'criteria': [
                    {'type': 'sudden_dropped_qrs', 'description': 'Sudden dropped QRS'},
                    {'type': 'constant_pr', 'description': 'Constant PR interval'}
                ]
            },
            'IIAVBI': {
                'name': '2nd Degree AV Block Type I',
                'criteria': [
                    {'type': 'wenckebach_pattern', 'description': 'Progressive PR prolongation'},
                    {'type': 'grouped_beating', 'description': 'Grouped beating pattern'}
                ]
            },
            'IIAVB': {
                'name': '3rd Degree AV Block',
                'criteria': [
                    {'type': 'av_dissociation', 'description': 'Complete AV dissociation'},
                    {'type': 'escape_rhythm', 'description': 'Junctional/ventricular escape'}
                ]
            },
            'LAFB': {
                'name': 'Left Anterior Fascicular Block',
                'criteria': [
                    {'type': 'left_axis_deviation', 'threshold': -45, 'description': 'Left axis deviation < -45°'},
                    {'type': 'qr_pattern', 'leads': ['I', 'aVL'], 'description': 'qR pattern in I, aVL'},
                    {'type': 'rs_pattern', 'leads': ['II', 'III', 'aVF'], 'description': 'rS pattern in inferior leads'}
                ]
            },
            'LPFB': {
                'name': 'Left Posterior Fascicular Block',
                'criteria': [
                    {'type': 'right_axis_deviation', 'threshold': 90, 'description': 'Right axis deviation > 90°'},
                    {'type': 'rs_pattern', 'leads': ['I', 'aVL'], 'description': 'rS pattern in I, aVL'},
                    {'type': 'qr_pattern', 'leads': ['II', 'III', 'aVF'], 'description': 'qR pattern in inferior leads'}
                ]
            },
            'LPR': {
                'name': 'Long PR Interval',
                'criteria': [
                    {'type': 'pr_interval', 'threshold': 0.20, 'description': 'PR interval > 200ms'},
                    {'type': 'all_p_conducted', 'description': 'All P waves conducted'}
                ]
            },
            'CLBBB': {
                'name': 'Complete Left Bundle Branch Block',
                'criteria': [
                    {'type': 'qrs_width', 'threshold': 0.12, 'description': 'QRS ≥ 120ms'},
                    {'type': 'broad_r_wave', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'Broad R wave in I, aVL, V5, V6'},
                    {'type': 'absent_q_waves', 'leads': ['I', 'V5', 'V6'], 'description': 'Absent Q waves in I, V5, V6'}
                ]
            },
            'CRBBB': {
                'name': 'Complete Right Bundle Branch Block',
                'criteria': [
                    {'type': 'qrs_width', 'threshold': 0.12, 'description': 'QRS ≥ 120ms'},
                    {'type': 'rsr_pattern', 'leads': ['V1', 'V2'], 'description': 'rSR\' pattern in V1, V2'},
                    {'type': 'wide_s_wave', 'leads': ['I', 'V6'], 'description': 'Wide S wave in I, V6'}
                ]
            },
            'ILBBB': {
                'name': 'Incomplete Left Bundle Branch Block',
                'criteria': [
                    {'type': 'qrs_width', 'range': [0.10, 0.12], 'description': 'QRS 100-120ms'},
                    {'type': 'left_axis_deviation', 'description': 'Left axis deviation'},
                    {'type': 'delayed_r_wave', 'leads': ['V5', 'V6'], 'description': 'Delayed R wave in V5, V6'}
                ]
            },
            'IRBBB': {
                'name': 'Incomplete Right Bundle Branch Block',
                'criteria': [
                    {'type': 'qrs_width', 'range': [0.10, 0.12], 'description': 'QRS 100-120ms'},
                    {'type': 'rsr_pattern', 'leads': ['V1', 'V2'], 'description': 'rSR\' pattern in V1, V2'},
                    {'type': 'terminal_r_wave', 'leads': ['V1'], 'description': 'Terminal R wave in V1'}
                ]
            },
            'ISCAL': {
                'name': 'Ischemia Anterolateral',
                'criteria': [
                    {'type': 'st_depression', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'ST depression in I, aVL, V5, V6'},
                    {'type': 't_wave_inversion', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'T wave inversion'}
                ]
            },
            'ISCAN': {
                'name': 'Ischemia Anterior',
                'criteria': [
                    {'type': 'st_depression', 'leads': ['V1', 'V2', 'V3', 'V4'], 'description': 'ST depression in V1-V4'},
                    {'type': 't_wave_inversion', 'leads': ['V1', 'V2', 'V3', 'V4'], 'description': 'T wave inversion in precordial leads'}
                ]
            },
            'ISCIN': {
                'name': 'Ischemia Inferior',
                'criteria': [
                    {'type': 'st_depression', 'leads': ['II', 'III', 'aVF'], 'description': 'ST depression in II, III, aVF'},
                    {'type': 't_wave_inversion', 'leads': ['II', 'III', 'aVF'], 'description': 'T wave inversion in inferior leads'}
                ]
            },
            'ISCLA': {
                'name': 'Ischemia Lateral',
                'criteria': [
                    {'type': 'st_depression', 'leads': ['I', 'aVL', 'V6'], 'description': 'ST depression in I, aVL, V6'},
                    {'type': 't_wave_inversion', 'leads': ['I', 'aVL', 'V6'], 'description': 'T wave inversion in lateral leads'}
                ]
            },
            'ISCI': {
                'name': 'Ischemia Inferolateral',
                'criteria': [
                    {'type': 'st_depression', 'leads': ['II', 'III', 'aVF', 'V5', 'V6'], 'description': 'ST depression in inferior and lateral leads'},
                    {'type': 't_wave_inversion', 'leads': ['II', 'III', 'aVF', 'V5', 'V6'], 'description': 'T wave inversion'}
                ]
            },

            # MYOCARDIAL INFARCTION
            'STTC': {
                'name': 'ST-T Changes',
                'criteria': [
                    {'type': 'st_elevation', 'threshold': 0.1, 'description': 'ST elevation > 1mm'},
                    {'type': 'st_depression', 'threshold': 0.1, 'description': 'ST depression > 1mm'},
                    {'type': 't_wave_changes', 'description': 'T wave abnormalities'}
                ]
            },
            'NST_': {
                'name': 'Nonspecific ST-T Changes',
                'criteria': [
                    {'type': 'minor_st_changes', 'description': 'Minor ST changes'},
                    {'type': 'nonspecific_t_changes', 'description': 'Nonspecific T wave changes'}
                ]
            },
            'IMI': {
                'name': 'Inferior Myocardial Infarction',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['II', 'III', 'aVF'], 'description': 'Q waves in II, III, aVF'},
                    {'type': 'st_elevation', 'leads': ['II', 'III', 'aVF'], 'description': 'ST elevation in inferior leads'},
                    {'type': 'reciprocal_changes', 'leads': ['I', 'aVL'], 'description': 'Reciprocal changes in I, aVL'}
                ]
            },
            'AMI': {
                'name': 'Anterior Myocardial Infarction',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['V1', 'V2', 'V3', 'V4'], 'description': 'Q waves in V1-V4'},
                    {'type': 'st_elevation', 'leads': ['V1', 'V2', 'V3', 'V4'], 'description': 'ST elevation in precordial leads'},
                    {'type': 'poor_r_progression', 'leads': ['V1', 'V2', 'V3'], 'description': 'Poor R wave progression'}
                ]
            },
            'ALMI': {
                'name': 'Anterolateral Myocardial Infarction',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['I', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 'description': 'Q waves in anterior and lateral leads'},
                    {'type': 'st_elevation', 'leads': ['I', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 'description': 'ST elevation'}
                ]
            },
            'LMI': {
                'name': 'Lateral Myocardial Infarction',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'Q waves in I, aVL, V5, V6'},
                    {'type': 'st_elevation', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'ST elevation in lateral leads'}
                ]
            },
            'ASMI': {
                'name': 'Anteroseptal Myocardial Infarction',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['V1', 'V2', 'V3'], 'description': 'Q waves in V1-V3'},
                    {'type': 'poor_r_progression', 'leads': ['V1', 'V2', 'V3'], 'description': 'Poor R wave progression'}
                ]
            },
            'INJAS': {
                'name': 'Injury/Ischemia',
                'criteria': [
                    {'type': 'st_elevation', 'description': 'ST elevation'},
                    {'type': 'hyperacute_t_waves', 'description': 'Hyperacute T waves'},
                    {'type': 'reciprocal_changes', 'description': 'Reciprocal ST depression'}
                ]
            },
            'INJAL': {
                'name': 'Injury Anterolateral',
                'criteria': [
                    {'type': 'st_elevation', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'ST elevation in I, aVL, V5, V6'},
                    {'type': 'hyperacute_t_waves', 'leads': ['I', 'aVL', 'V5', 'V6'], 'description': 'Hyperacute T waves'}
                ]
            },
            'INJIN': {
                'name': 'Injury Inferior',
                'criteria': [
                    {'type': 'st_elevation', 'leads': ['II', 'III', 'aVF'], 'description': 'ST elevation in II, III, aVF'},
                    {'type': 'hyperacute_t_waves', 'leads': ['II', 'III', 'aVF'], 'description': 'Hyperacute T waves'}
                ]
            },
            'INJLA': {
                'name': 'Injury Lateral',
                'criteria': [
                    {'type': 'st_elevation', 'leads': ['I', 'aVL', 'V6'], 'description': 'ST elevation in I, aVL, V6'},
                    {'type': 'hyperacute_t_waves', 'leads': ['I', 'aVL', 'V6'], 'description': 'Hyperacute T waves'}
                ]
            },
            'IPLMI': {
                'name': 'Inferoposterolateral MI',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['II', 'III', 'aVF', 'V5', 'V6'], 'description': 'Q waves in inferior and lateral leads'},
                    {'type': 'tall_r_waves', 'leads': ['V1', 'V2'], 'description': 'Tall R waves in V1, V2 (posterior)'}
                ]
            },
            'IPMI': {
                'name': 'Inferoposterior MI',
                'criteria': [
                    {'type': 'q_waves', 'leads': ['II', 'III', 'aVF'], 'description': 'Q waves in inferior leads'},
                    {'type': 'tall_r_waves', 'leads': ['V1', 'V2'], 'description': 'Tall R waves in V1, V2'}
                ]
            },

            # HYPERTROPHY
            'LVH': {
                'name': 'Left Ventricular Hypertrophy',
                'criteria': [
                    {'type': 'sokolow_lyon', 'threshold': 3.5, 'description': 'S(V1) + R(V5/V6) > 3.5mV'},
                    {'type': 'cornell_criteria', 'description': 'Cornell voltage criteria'},
                    {'type': 'st_depression', 'leads': ['V5', 'V6'], 'description': 'ST depression with strain pattern'}
                ]
            },
            'RVH': {
                'name': 'Right Ventricular Hypertrophy',
                'criteria': [
                    {'type': 'tall_r_wave', 'leads': ['V1'], 'description': 'Tall R wave in V1'},
                    {'type': 'right_axis_deviation', 'description': 'Right axis deviation'},
                    {'type': 'rv_strain', 'leads': ['V1', 'V2'], 'description': 'RV strain pattern'}
                ]
            },
            'HVOLT': {
                'name': 'High Voltage',
                'criteria': [
                    {'type': 'high_voltage', 'description': 'High voltage QRS complexes'},
                    {'type': 'voltage_criteria', 'description': 'Meets voltage criteria for hypertrophy'}
                ]
            },
            'LAO/LAE': {
                'name': 'Left Atrial Enlargement',
                'criteria': [
                    {'type': 'p_wave_duration', 'threshold': 0.12, 'description': 'P wave duration > 120ms'},
                    {'type': 'bifid_p_wave', 'leads': ['II'], 'description': 'Bifid P wave in II'},
                    {'type': 'terminal_p_force', 'leads': ['V1'], 'description': 'Terminal P force in V1'}
                ]
            },
            'RAO/RAE': {
                'name': 'Right Atrial Enlargement',
                'criteria': [
                    {'type': 'tall_p_wave', 'leads': ['II'], 'threshold': 0.25, 'description': 'Tall P wave in II > 2.5mm'},
                    {'type': 'peaked_p_wave', 'description': 'Peaked P waves'},
                    {'type': 'right_axis_p', 'description': 'Right axis deviation of P wave'}
                ]
            },

            # ARRHYTHMIAS
            'SVARR': {
                'name': 'Supraventricular Arrhythmia',
                'criteria': [
                    {'type': 'irregular_rhythm', 'description': 'Irregular supraventricular rhythm'},
                    {'type': 'narrow_qrs', 'description': 'Narrow QRS complexes'},
                    {'type': 'variable_morphology', 'description': 'Variable P wave morphology'}
                ]
            },
            'VCLVH': {
                'name': 'Voltage Criteria for LVH',
                'criteria': [
                    {'type': 'voltage_criteria', 'description': 'Meets voltage criteria for LVH'},
                    {'type': 'no_strain_pattern', 'description': 'No strain pattern present'}
                ]
            },
            'QWAVE': {
                'name': 'Q Wave Abnormality',
                'criteria': [
                    {'type': 'pathological_q_waves', 'description': 'Pathological Q waves present'},
                    {'type': 'q_wave_duration', 'threshold': 0.04, 'description': 'Q wave duration > 40ms'},
                    {'type': 'q_wave_depth', 'threshold': 0.25, 'description': 'Q wave depth > 25% of R wave'}
                ]
            },
            'PRWP': {
                'name': 'Poor R Wave Progression',
                'criteria': [
                    {'type': 'poor_r_progression', 'leads': ['V1', 'V2', 'V3'], 'description': 'Poor R wave progression V1-V3'},
                    {'type': 'rv_transition', 'description': 'Delayed R/S transition'}
                ]
            },
            'LOWT': {
                'name': 'Low T Wave Amplitude',
                'criteria': [
                    {'type': 'low_t_waves', 'description': 'Low amplitude T waves'},
                    {'type': 'flat_t_waves', 'description': 'Flat T waves'}
                ]
            },
            'NT_': {
                'name': 'Nonspecific T Wave Abnormality',
                'criteria': [
                    {'type': 'nonspecific_t_changes', 'description': 'Nonspecific T wave changes'},
                    {'type': 'minor_t_abnormalities', 'description': 'Minor T wave abnormalities'}
                ]
            },
            'TAB_': {
                'name': 'T Wave Abnormality',
                'criteria': [
                    {'type': 't_wave_inversion', 'description': 'T wave inversion'},
                    {'type': 't_wave_flattening', 'description': 'T wave flattening'},
                    {'type': 'biphasic_t_waves', 'description': 'Biphasic T waves'}
                ]
            },
            'INVT': {
                'name': 'Inverted T Waves',
                'criteria': [
                    {'type': 't_wave_inversion', 'description': 'T wave inversion'},
                    {'type': 'deep_t_inversion', 'description': 'Deep T wave inversion'}
                ]
            },
            'LVOLT': {
                'name': 'Low Voltage',
                'criteria': [
                    {'type': 'low_voltage_limb', 'threshold': 0.5, 'description': 'Low voltage in limb leads < 5mm'},
                    {'type': 'low_voltage_precordial', 'threshold': 1.0, 'description': 'Low voltage in precordial leads < 10mm'}
                ]
            },
            'NORM': {
                'name': 'Normal ECG',
                'criteria': [
                    {'type': 'normal_rhythm', 'description': 'Normal sinus rhythm'},
                    {'type': 'normal_axis', 'description': 'Normal axis'},
                    {'type': 'normal_intervals', 'description': 'Normal intervals'},
                    {'type': 'normal_morphology', 'description': 'Normal QRS and T wave morphology'}
                ]
            },
            'SR': {
              'name': 'Sinus Rhythm',
              'criteria': [
                  {'type': 'p_wave_present', 'description': 'P wave present before every QRS'},
                  {'type': 'rr_regular', 'description': 'Regular RR intervals'},
                  {'type': 'normal_heart_rate', 'description': 'Heart rate between 60–100 bpm'},
                  {'type': 'normal_axis', 'description': 'Normal axis (QRS axis between -30° and +90°)'}
              ]
          },
            'ABQRS': {
                'name': 'Abnormal QRS',
                'criteria': [
                    {'type': 'abnormal_qrs_morphology', 'description': 'Abnormal QRS morphology'},
                    {'type': 'wide_qrs', 'description': 'Wide QRS complexes'}
                ]
            },
            'PVC': {
                'name': 'Premature Ventricular Contractions',
                'criteria': [
                    {'type': 'premature_beats', 'description': 'Premature ventricular beats'},
                    {'type': 'wide_qrs', 'description': 'Wide QRS > 120ms'},
                    {'type': 'compensatory_pause', 'description': 'Compensatory pause'}
                ]
            },
            'SPB': {
                'name': 'Supraventricular Premature Beats',
                'criteria': [
                    {'type': 'premature_beats', 'description': 'Premature supraventricular beats'},
                    {'type': 'narrow_qrs', 'description': 'Narrow QRS complexes'},
                    {'type': 'abnormal_p_wave', 'description': 'Abnormal P wave morphology'}
                ]
            },
            'PSVT': {
                'name': 'Paroxysmal Supraventricular Tachycardia',
                'criteria': [
                    {'type': 'rapid_rate', 'range': [150, 250], 'description': 'Rapid heart rate 150-250 bpm'},
                    {'type': 'narrow_qrs', 'description': 'Narrow QRS complexes'},
                    {'type': 'regular_rhythm', 'description': 'Regular rhythm'}
                ]
            },
            'VTAC': {
                'name': 'Ventricular Tachycardia',
                'criteria': [
                    {'type': 'rapid_rate', 'range': [150, 250], 'description': 'Rapid heart rate 150-250 bpm'},
                    {'type': 'wide_qrs', 'threshold': 0.12, 'description': 'Wide QRS > 120ms'},
                    {'type': 'av_dissociation', 'description': 'AV dissociation'}
                ]
            },
            'VFIB': {
                'name': 'Ventricular Fibrillation',
                'criteria': [
                    {'type': 'chaotic_rhythm', 'description': 'Chaotic ventricular rhythm'},
                    {'type': 'no_qrs_complexes', 'description': 'No identifiable QRS complexes'},
                    {'type': 'irregular_waves', 'description': 'Irregular fibrillatory waves'}
                ]
            },
            'ASYS': {
                'name': 'Asystole',
                'criteria': [
                    {'type': 'no_electrical_activity', 'description': 'No electrical activity'},
                    {'type': 'flat_line', 'description': 'Flat line or minimal'},
                    {'type': 'activity', 'description': 'Minimal electrical activity'}
                ]
            }
        }

    def check_scaling(self, ecg_signal):

        for factor in [1, 10, 100, 1000]:
            scaled = ecg_signal / factor


    def detect_r_peaks(self, ecg_signal, lead=1):
        """Detect R peaks in ECG signal"""
        from scipy.signal import find_peaks

        if lead >= len(ecg_signal):
            lead = 1 

        signal_data = ecg_signal[lead]

        height = np.max(signal_data) * 0.5 
        distance = int(0.6 * self.sampling_rate) 

        peaks, _ = find_peaks(signal_data, height=height, distance=distance)

        return peaks


    def calculate_rr_intervals(self, r_peaks):
        """Calculate R-R intervals"""
        if len(r_peaks) < 2:
            return []

        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        return rr_intervals

    def calculate_heart_rate(self, rr_intervals):
        """Calculate heart rate from R-R intervals"""
        if len(rr_intervals) == 0:
            return 0

        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr if avg_rr > 0 else 0
        return heart_rate

    def detect_qrs_width(self, ecg_signal, r_peaks, lead=1):
        """Detect QRS width"""
        if len(r_peaks) < 2:
            return []

        qrs_widths = []
        signal_data = ecg_signal[lead]

        for peak in r_peaks:
            q_search_start = max(0, peak - int(0.08 * self.sampling_rate))
            q_search_end = peak

            s_search_start = peak
            s_search_end = min(len(signal_data), peak + int(0.08 * self.sampling_rate))
            
            qrs_width = (s_search_end - q_search_start) / self.sampling_rate
            qrs_widths.append(qrs_width)

        return qrs_widths

    def detect_st_changes(self, ecg_signal, r_peaks, lead=1):
        """Detect ST elevation/depression with refined criteria"""
        if len(r_peaks) < 2:
            return {'elevation': [], 'depression': []}

        signal_data = ecg_signal[lead]
        st_changes = {'elevation': [], 'depression': []}

        for peak in r_peaks:
            j_point = peak + int(0.06 * self.sampling_rate) 

            if j_point < len(signal_data):
                baseline_start = max(0, peak - int(0.25 * self.sampling_rate))
                baseline_end = max(0, peak - int(0.15 * self.sampling_rate))

                if baseline_end > baseline_start:
                    baseline = np.mean(signal_data[baseline_start:baseline_end])
                    st_value = signal_data[j_point]
                    deviation = st_value - baseline

                    if deviation > 0.2:
                        st_changes['elevation'].append((j_point, deviation))
                    elif deviation < -0.2:
                        st_changes['depression'].append((j_point, deviation))

        return st_changes


    def detect_t_wave_changes(self, ecg_signal, r_peaks, lead=1):
        """Detect T wave abnormalities"""
        if len(r_peaks) < 2:
            return {'inverted': [], 'flat': [], 'peaked': []}

        signal_data = ecg_signal[lead]
        t_changes = {'inverted': [], 'flat': [], 'peaked': []}

        for i, peak in enumerate(r_peaks[:-1]):
            t_start = peak + int(0.15 * self.sampling_rate)
            t_end = min(len(signal_data), peak + int(0.4 * self.sampling_rate))

            if t_end > t_start:
                t_segment = signal_data[t_start:t_end]

                t_peak_idx = np.argmax(np.abs(t_segment))
                t_peak_value = t_segment[t_peak_idx]
                t_peak_pos = t_start + t_peak_idx

                baseline_start = max(0, peak - int(0.2 * self.sampling_rate))
                baseline_end = max(0, peak - int(0.1 * self.sampling_rate))

                if baseline_end > baseline_start:
                    baseline = np.mean(signal_data[baseline_start:baseline_end])
                    
                    if t_peak_value < baseline - 0.1:
                        t_changes['inverted'].append((t_peak_pos, t_peak_value))
                    elif abs(t_peak_value - baseline) < 0.05:
                        t_changes['flat'].append((t_peak_pos, t_peak_value))
                    elif t_peak_value > baseline + 0.5:
                        t_changes['peaked'].append((t_peak_pos, t_peak_value))

        return t_changes

    def detect_p_waves(self, ecg_signal, r_peaks, lead=1):
        """Detect P waves and abnormalities"""
        if len(r_peaks) < 2:
            return {'present': [], 'absent': [], 'abnormal': []}

        signal_data = ecg_signal[lead]
        p_waves = {'present': [], 'absent': [], 'abnormal': []}

        for peak in r_peaks:
            p_search_start = max(0, peak - int(0.25 * self.sampling_rate))
            p_search_end = max(0, peak - int(0.08 * self.sampling_rate))

            if p_search_end > p_search_start:
                p_segment = signal_data[p_search_start:p_search_end]

                p_peak_idx = np.argmax(p_segment)
                p_peak_value = p_segment[p_peak_idx]
                p_peak_pos = p_search_start + p_peak_idx
                baseline_start = max(0, peak - int(0.4 * self.sampling_rate))
                baseline_end = max(0, peak - int(0.3 * self.sampling_rate))

                if baseline_end > baseline_start:
                    baseline = np.mean(signal_data[baseline_start:baseline_end])

                    # P wave detection
                    if p_peak_value > baseline + 0.05:
                        p_duration = len(p_segment) / self.sampling_rate

                        if p_duration > 0.12:
                            p_waves['abnormal'].append((p_peak_pos, p_peak_value, 'prolonged'))
                        elif p_peak_value > baseline + 0.25:
                            p_waves['abnormal'].append((p_peak_pos, p_peak_value, 'tall'))
                        else:
                            p_waves['present'].append((p_peak_pos, p_peak_value))
                    else:
                        p_waves['absent'].append((p_peak_pos, 0))

        return p_waves


    def analyze_axis_deviation(self, ecg_signal):
        lead_i = ecg_signal[0] if len(ecg_signal) > 0 else np.zeros(1000)
        lead_avf = ecg_signal[5] if len(ecg_signal) > 5 else np.zeros(1000)

        net_i = np.mean(lead_i)
        net_avf = np.mean(lead_avf)

        if net_i > 0 and net_avf > 0:
            axis = "Normal (0° to +90°)"
        elif net_i > 0 and net_avf < 0:
            axis = "Left axis deviation (-30° to -90°)"
        elif net_i < 0 and net_avf > 0:
            axis = "Right axis deviation (+90° to +180°)"
        else:
            axis = "Extreme axis deviation"

        return axis, net_i, net_avf

    def evaluate_clinical_criteria(self, ecg_signal, condition_code):
        """Evaluate clinical criteria for a specific condition"""
        if condition_code not in self.clinical_criteria:
            return {"error": f"Unknown condition: {condition_code}"}

        criteria = self.clinical_criteria[condition_code]
        results = {
            'condition': criteria['name'],
            'criteria_met': [],
            'criteria_not_met': [],
            'clinical_evidence': {},
            'confidence': 0
        }

        r_peaks = self.detect_r_peaks(ecg_signal)
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        heart_rate = self.calculate_heart_rate(rr_intervals)
        qrs_widths = self.detect_qrs_width(ecg_signal, r_peaks)
        st_changes = self.detect_st_changes(ecg_signal, r_peaks)
        t_changes = self.detect_t_wave_changes(ecg_signal, r_peaks)
        p_waves = self.detect_p_waves(ecg_signal, r_peaks)
        axis, net_i, net_avf = self.analyze_axis_deviation(ecg_signal)

        results['clinical_evidence'] = {
            'r_peaks': r_peaks,
            'rr_intervals': rr_intervals,
            'heart_rate': heart_rate,
            'qrs_widths': qrs_widths,
            'st_changes': st_changes,
            't_changes': t_changes,
            'p_waves': p_waves,
            'axis': axis,
            'net_i': net_i,
            'net_avf': net_avf
        }

        criteria_met = 0
        total_criteria = len(criteria['criteria'])

        for criterion in criteria['criteria']:
            criterion_met = False

            if criterion['type'] == 'heart_rate':
                if 'range' in criterion:
                    if criterion['range'][0] <= heart_rate <= criterion['range'][1]:
                        criterion_met = True

            elif criterion['type'] == 'rr_irregularity':
                if len(rr_intervals) > 1:
                    rr_std = np.std(rr_intervals)
                    if rr_std > criterion['threshold']:
                        criterion_met = True

            elif criterion['type'] == 'qrs_width':
                if len(qrs_widths) > 0:
                    avg_qrs = np.mean(qrs_widths)
                    if 'threshold' in criterion:
                        if avg_qrs >= criterion['threshold']:
                            criterion_met = True
                    elif 'range' in criterion:
                        if criterion['range'][0] <= avg_qrs <= criterion['range'][1]:
                            criterion_met = True

            elif criterion['type'] == 'st_elevation':
                if len(st_changes['elevation']) > 0:
                    criterion_met = True

            elif criterion['type'] == 'st_depression':
                if len(st_changes['depression']) > 0:
                    criterion_met = True

            elif criterion['type'] == 't_wave_inversion':
                if len(t_changes['inverted']) > 0:
                    criterion_met = True

            elif criterion['type'] == 'p_wave_absent':
                if len(p_waves['absent']) > len(p_waves['present']):
                    criterion_met = True

            elif criterion['type'] == 'left_axis_deviation':
                if "Left axis" in axis:
                    criterion_met = True

            elif criterion['type'] == 'right_axis_deviation':
                if "Right axis" in axis:
                    criterion_met = True

            if criterion_met:
                results['criteria_met'].append(criterion['description'])
                criteria_met += 1
            else:
                results['criteria_not_met'].append(criterion['description'])

        results['confidence'] = criteria_met / total_criteria if total_criteria > 0 else 0
        return results


    def visualize_clinical_evidence(self, ecg_signal, predictions_dict, threshold=0.5):
        top_predictions = sorted([(code, prob) for code, prob in predictions_dict.items()
                                if prob >= threshold], key=lambda x: x[1], reverse=True)[:3]

        if not top_predictions:
            top_predictions = sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True)[:1]
            threshold = 0
        fig = plt.figure(figsize=(30, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.1)

        # Plot 12-lead ECG
        lead_positions = [
            (0, 0), (0, 1), (0, 2),  # I, II, III
            (1, 0), (1, 1), (1, 2),  # aVR, aVL, aVF
            (2, 0), (2, 1), (2, 2),  # V1, V2, V3
            (3, 0), (3, 1), (3, 2)   # V4, V5, V6
        ]

        all_evidence = {}
        for condition_code, prob in top_predictions:
            evaluation = self.evaluate_clinical_criteria(ecg_signal, condition_code)
            if 'error' not in evaluation:
                all_evidence[condition_code] = evaluation['clinical_evidence']

        combined_evidence = {
            'r_peaks': [],
            'rr_intervals': [],
            'heart_rate': 0,
            'qrs_widths': [],
            'st_changes': {'elevation': [], 'depression': []},
            't_changes': {'inverted': [], 'flat': [], 'peaked': []},
            'p_waves': {'present': [], 'absent': [], 'abnormal': []},
            'axis': "Normal",
            'net_i': 0,
            'net_avf': 0
        }

        for evidence in all_evidence.values():
            if len(evidence['r_peaks']) > 0 and len(combined_evidence['r_peaks']) == 0:
                combined_evidence['r_peaks'] = evidence['r_peaks']
            if len(evidence['rr_intervals']) > 0 and len(combined_evidence['rr_intervals']) == 0:
                combined_evidence['rr_intervals'] = evidence['rr_intervals']
            if evidence['heart_rate'] > 0 and combined_evidence['heart_rate'] == 0:
                combined_evidence['heart_rate'] = evidence['heart_rate']
            if len(evidence['qrs_widths']) > 0 and len(combined_evidence['qrs_widths']) == 0:
                combined_evidence['qrs_widths'] = evidence['qrs_widths']
            if len(evidence['st_changes']['elevation']) > 0 and len(combined_evidence['st_changes']['elevation']) == 0:
                combined_evidence['st_changes']['elevation'] = evidence['st_changes']['elevation']
            if len(evidence['st_changes']['depression']) > 0 and len(combined_evidence['st_changes']['depression']) == 0:
                combined_evidence['st_changes']['depression'] = evidence['st_changes']['depression']
            if len(evidence['t_changes']['inverted']) > 0 and len(combined_evidence['t_changes']['inverted']) == 0:
                combined_evidence['t_changes']['inverted'] = evidence['t_changes']['inverted']
            if evidence['axis'] != "Normal" and combined_evidence['axis'] == "Normal":
                combined_evidence['axis'] = evidence['axis']

        for i, (row, col) in enumerate(lead_positions):
            if i >= len(ecg_signal):
                break

            ax = fig.add_subplot(gs[row, col])
            time_axis = np.arange(len(ecg_signal[i])) / self.sampling_rate
            time_mm = time_axis * 10 


            ecg_mv = ecg_signal[i] / 5

            ax.plot(time_mm, ecg_mv, color='#44062a', linewidth=1, alpha=0.8)
            add_ecg_grid(ax, duration_sec=10, mv_range=2)
            ax.set_xticks(np.arange(0, 11, 1))
            self._annotate_common_findings(ax, time_axis, ecg_signal[i], combined_evidence, i)

            ax.set_title(f'{self.lead_names[i]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('mV', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)



        fig.suptitle('Diagnosis Based on ECG', fontsize= 24, fontweight='bold',  y=0.93)

        combined_criteria = []
        for condition_code, prob in top_predictions:
            evaluation = self.evaluate_clinical_criteria(ecg_signal, condition_code)
            if 'error' not in evaluation:
                all_evidence[condition_code] = evaluation['clinical_evidence']
                combined_criteria.extend(evaluation['criteria_met'])

        summary_text = self._create_simplified_summary(top_predictions, combined_evidence, combined_criteria, threshold)

        fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.savefig("models/sample_prediction.png", dpi=300, bbox_inches='tight')
        plt.show()

        return all_evidence

    def _annotate_common_findings(self, ax, time_axis, signal, evidence, lead_idx):
          signal_mv = signal / 5

          if len(evidence['r_peaks']) > 0:
              r_times = evidence['r_peaks'] / self.sampling_rate * 10
              r_values = signal_mv[evidence['r_peaks']]
              # ax.scatter(r_times, r_values, color='orange', s=50, alpha=0.7,
              #           marker='o', label='R peaks', zorder=5)

          if lead_idx < 6:
              # ST elevations
              for st_point, deviation in evidence['st_changes']['elevation']:
                  st_time = st_point / self.sampling_rate * 10
                  if st_time < len(time_axis):
                      ax.annotate(f'ST↑ {deviation:.1f}mV',
                                xy=(st_time, signal_mv[st_point]),
                                xytext=(st_time, signal_mv[st_point] + 0.352),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                                fontsize=9, color='#0e7a23', fontweight='bold')

              # ST depressions
              for st_point, deviation in evidence['st_changes']['depression']:
                  st_time = st_point / self.sampling_rate * 10
                  if st_time < len(time_axis):
                      ax.annotate(f'ST↓ {deviation:.1f}mV',
                                xy=(st_time, signal_mv[st_point]),
                                xytext=(st_time, signal_mv[st_point] - 0.302),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                                fontsize=9, color='blue', fontweight='bold')

          # Annotate T wave inversions
          for t_point, value in evidence['t_changes']['inverted']:
              t_time = t_point / self.sampling_rate * 10
              if t_time < len(time_axis):
                  ax.annotate('T inv',
                            xy=(t_time, signal_mv[t_point]),
                            xytext=(t_time, signal_mv[t_point] - 0.315),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1),
                            fontsize=9, color='red', fontweight='bold')

          if lead_idx == 1:
              ax.text(0.02, 0.02, f'HR: {evidence["heart_rate"]:.0f} bpm',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))



    def _create_simplified_summary(self, predictions, evidence,criteria_met=None, threshold=0.3):

        summary = "CLINICAL SUMMARY\n"
        summary += "=" * 40 + "\n"

        filtered_predictions = [(code, prob) for code, prob in predictions if prob >= threshold]

        if filtered_predictions:
            summary += "DIAGNOSTIC PREDICTION:\n".format(int(threshold*100))
            for condition_code, prob in filtered_predictions:
                condition_name = self.clinical_criteria.get(condition_code, {}).get('name', condition_code)
                summary += f"  • {condition_name}: {prob:.1%}\n"
        else:
            summary += "NO SIGNIFICANT PREDICTIONS\n"

        summary += "\nKEY MEASUREMENTS:\n"
        summary += f"  • Heart Rate: {evidence['heart_rate']:.0f} bpm\n"
        summary += f"  • Axis: {evidence['axis']}\n"

        if len(evidence['qrs_widths']) > 0:
            summary += f"  • Avg QRS Width: {np.mean(evidence['qrs_widths'])*1000:.0f} ms\n"

        if len(evidence['rr_intervals']) > 1:
            summary += f"  • RR Variability: {np.std(evidence['rr_intervals']):.3f} s\n"

        if evidence['st_changes']['elevation']:
            summary += f"  • ST Elevation: Present\n"

        if evidence['st_changes']['depression']:
            summary += f"  • ST Depression: Present\n"

        if evidence['t_changes']['inverted']:
            summary += f"  • T Wave Inversion: Present\n"
        if criteria_met:
            for c in criteria_met[:5]:
                summary += f"  • {c}\n"
        return summary


def analyze_ecg_with_clinical_evidence(model, ecg_signal, all_labels, device, analyzer, threshold=0.3):

    model.eval()

    with torch.no_grad():
        if len(ecg_signal.shape) == 2:
            ecg_tensor = ecg_signal.unsqueeze(0).to(device)
        else:
            ecg_tensor = ecg_signal.to(device)

        logits = model(ecg_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    predictions_dict = {all_labels[i]: probs[i] for i in range(len(all_labels))}
    ecg_numpy = ecg_signal.cpu().numpy() if torch.is_tensor(ecg_signal) else ecg_signal
    if len(ecg_numpy.shape) == 3:
        ecg_numpy = ecg_numpy[0]

    clinical_results = analyzer.visualize_clinical_evidence(ecg_numpy, predictions_dict, threshold)

    return clinical_results

analyzer = ECGClinicalAnalyzer()

# Test
print("\n TESTING WITH SAMPLE ECG...")
test_sample = test_dataset[1967]
sample_signal = test_sample['signal']
true_labels = test_sample['labels'].numpy()

analyzer.check_scaling(sample_signal[0])

clinical_results = analyze_ecg_with_clinical_evidence(
    model, sample_signal, all_labels, device, analyzer, threshold=0.5
)