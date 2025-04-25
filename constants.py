import numpy as np

# Model dimensions
MODEL_WIDTH = 512
MODEL_HEIGHT = 256

# Output array structure constants
TRAJECTORY_SIZE = 33
DESIRE_LEN = 8
TRAFFIC_CONVENTION_LEN = 2
PLAN_MHP_N = 5
PLAN_MHP_COLUMNS = 15
LEAD_MHP_N = 2
LEAD_TRAJ_LEN = 6
LEAD_PRED_DIM = 4
STOP_LINE_MHP_N = 3
STOP_LINE_PRED_DIM = 8
META_STRIDE = 7
NUM_META_INTERVALS = 5
TEMPORAL_SIZE = 512

# Output indices in array - must match Phantom exactly
PLAN_MHP_GROUP_SIZE = (2*PLAN_MHP_COLUMNS*TRAJECTORY_SIZE + 1)
LEAD_MHP_GROUP_SIZE = (2*LEAD_PRED_DIM*LEAD_TRAJ_LEN + 3)
STOP_LINE_MHP_GROUP_SIZE = (2*STOP_LINE_PRED_DIM + 1)

PLAN_IDX = 0
LL_IDX = PLAN_IDX + PLAN_MHP_N*PLAN_MHP_GROUP_SIZE
LL_PROB_IDX = LL_IDX + 4*2*2*TRAJECTORY_SIZE
RE_IDX = LL_PROB_IDX + 8
LEAD_IDX = RE_IDX + 2*2*2*TRAJECTORY_SIZE
LEAD_PROB_IDX = LEAD_IDX + LEAD_MHP_N*(LEAD_MHP_GROUP_SIZE)
STOP_LINE_IDX = LEAD_PROB_IDX + 3
STOP_LINE_PROB_IDX = STOP_LINE_IDX + STOP_LINE_MHP_N*STOP_LINE_MHP_GROUP_SIZE
DESIRE_STATE_IDX = STOP_LINE_PROB_IDX + 1
META_IDX = DESIRE_STATE_IDX + DESIRE_LEN
OTHER_META_SIZE = 48
DESIRE_PRED_SIZE = 32
POSE_IDX = META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE
POSE_SIZE = 12
OUTPUT_SIZE = POSE_IDX + POSE_SIZE

# Cityscapes class definitions
CITYSCAPES_CLASSES = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle'
}

# Class groups for Cityscapes
ROAD_CLASSES = [7, 8, 9, 10]  # road, sidewalk, parking, rail track
VEHICLE_CLASSES = [26, 27, 28, 29, 30, 31, 32, 33]  # car, truck, bus, etc.
HUMAN_CLASSES = [24, 25]  # person, rider
TRAFFIC_CONTROL_CLASSES = [19, 20]  # traffic light, traffic sign

# Comma10k class definitions
COMMA10K_CLASSES = {
    0: 'none',
    1: 'road',  # road (all parts, anywhere nobody would look at you funny for driving)
    2: 'lane_markings',  # lane markings (don't include non lane markings like turn arrows and crosswalks)
    3: 'undrivable',  # undrivable
    4: 'movable',  # movable (vehicles and people/animals)
    5: 'my_car',  # my car (and anything inside it, including wires, mounts, etc. No reflections)
    6: 'movable_in_car'  # movable in my car (people inside the car, imgsd only)
}

# Helper functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=0):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True) 