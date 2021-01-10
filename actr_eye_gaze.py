import string
import random
import warnings
import math

from io import StringIO
import pyactr as actr
import numpy as np
import pandas as pd

import sys, re, ast, argparse, os
from tqdm import tqdm
tqdm.pandas()

class Model(object):
    """
    Model searching and attending to various stimuli.
    """

    def __init__(self, env, **kwargs):
        self.m = actr.ACTRModel(environment=env, **kwargs)

        actr.chunktype("pair", "probe answer")
        
        actr.chunktype("goal", "state")

        self.dm = self.m.decmem

        self.m.visualBuffer("visual", "visual_location", self.dm, finst=30)

        start = actr.makechunk(nameofchunk="start", typename="chunk", value="start")
        actr.makechunk(nameofchunk="attending", typename="chunk", value="attending")
        actr.makechunk(nameofchunk="done", typename="chunk", value="done")
        self.m.goal.add(actr.makechunk(typename="read", state=start))
        self.m.set_goal("g2")
        self.m.goals["g2"].delay=0.2

        self.m.productionstring(name="find_probe", string="""
        =g>
        isa     goal
        state   start
        ?visual_location>
        buffer  empty
        ==>
        =g>
        isa     goal
        state   attend
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x closest""") #this rule is used if automatic visual search does not put anything in the buffer

        self.m.productionstring(name="check_probe", string="""
        =g>
        isa     goal
        state   start
        ?visual_location>
        buffer  full
        ==>
        =g>
        isa     goal
        state   attend""")  #this rule is used if automatic visual search is enabled and it puts something in the buffer

        self.m.productionstring(name="attend_probe", string="""
        =g>
        isa     goal
        state   attend
        =visual_location>
        isa    _visuallocation
        ?visual>
        state   free
        ==>
        =g>
        isa     goal
        state   reading
        +visual>
        isa     _visual
        cmd     move_attention
        screen_pos =visual_location
        ~visual_location>""")

        self.m.productionstring(name="find_target_probe", string="""
        =g>
        isa     goal
        state   reading
        =visual>
        isa     _visual
        value   "123"
        ==>
        =g>
        state   free""")


        self.m.productionstring(name="encode_probe_and_find_new_location", string="""
        =g>
        isa     goal
        state   reading
        =visual>
        isa     _visual
        value   =val
        ?visual_location>
        buffer  empty
        ==>
        =g>
        isa     goal
        state   attend
        ~visual>
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x closest""")

def calc_obj_info(split_string):
    object_info = []
    for obj in ast.literal_eval(split_string): 
        object_type = obj[0]
        prob = float(obj[1])
        right_X = int(obj[2])
        left_X = int(obj[3])
        bottom_Y = int(obj[4])
        top_Y = int(obj[5])

        mid_X = (left_X + right_X)//2
        mid_Y = (top_Y + bottom_Y)//2

        width = right_X - left_X
        height = bottom_Y - top_Y
        area = width*height
        tmp_var = prob/2000
        delay = -math.log(float(tmp_var))
        object_info.append([object_type, prob, mid_X, mid_Y, area, delay])
    return object_info

def read_obj_log_file(filename):
    df = pd.read_csv(filename, header=None, names=['image', 'image_data'])
    df['object_info'] = df.apply(lambda x: calc_obj_info(x['image_data']), axis=1)
    return df

def process_actr_data(cmd_str):
    cmd_lst = ast.literal_eval(cmd_str)
    gaze_data= []
    for cmd in cmd_lst:
        gaze = float(re.findall(r'([0-9]+\.[0-9]*)',cmd)[0])
        screen_x = int(re.findall(r'screen_x=(\s+[0-9]*),', cmd)[0])
        screen_y = int(re.findall(r'screen_y=(\s+[0-9]*),', cmd)[0])
        gaze_data.append([screen_x, screen_y, gaze])
    return gaze_data

def run_simulations(list_of_obj, aspect_ratio=(640, 480), log_file='actr_simulations.log'):
    oldstd = sys.stdout
    stim_d = {key: {'text':key, 'position': (x[2], x[3]), 'vis_delay': x[5]} for key,x in enumerate(sorted(list_of_obj, key=lambda objs: objs[4],reverse=True))}
    sys.stdout = StringIO(log_file)
    environ = actr.Environment(size=aspect_ratio, simulated_display_resolution=aspect_ratio, simulated_screen_size=(60, 34), viewing_distance=60)
    m = Model(environ, subsymbolic=True, latency_factor=0.4, decay=0.5, retrieval_threshold=-2, instantaneous_noise=0, automatic_visual_search=True, 
    eye_mvt_scaling_parameter=0.05, eye_mvt_angle_parameter=10, emma_landing_site_noise=True, emma=True) #If you don't want to use the EMMA model, specify emma=False in here
    sim = m.m.simulation(realtime=False, trace=True,  gui=False, environment_process=environ.environment_process, stimuli=stim_d, triggers='X', times=1)
    sim.run(10)
    check = 0
    sys.stdout = log_line = StringIO()
    for key in m.dm:
        if key.typename == '_visual':
            print(key, m.dm[key])
            check += 1
    sys.stdout = oldstd
    gaze_data = str(log_line.getvalue()).split('\n')
    log_line.close()

    if len(gaze_data[-1]) == 0:
        gaze_data.pop()
    return gaze_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p", help="path to your detected file")

    # read arguments from the command line
    args = parser.parse_args()
    if args.path:
        filepath = args.path
    else:
        filepath = os.path.join('data', 'salicon/detected', 'salicon_detected_objects.csv')

    df = read_obj_log_file(filepath)
    # df =  pd.read_csv('C:\\Users\\sktsa\\Projects\\Keras-Multiple-Process-Prediction-master\\salicon_val_detected_actr.csv')
    # npa = df['object_info'].to_numpy()

    # vfunc = np.vectorize(run_simulations)
    # vfunc(npa)
    # for am in npa:
    #     run_simulations(am, (640,480))
    # df['actr_data'] = pd.Series()
    
    df['actr_data'] = df.progress_apply(lambda x: run_simulations(x['object_info'], (640, 480)), axis=1)
    df['actr_data_processed'] = df.progress_apply(lambda x: process_actr_data(x['actr_data']), axis=1)
    df.info()
    df.to_csv(os.path.join('data', 'salicon/simuations','salicon_actr.csv'), index=False)