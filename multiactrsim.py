import string
import random
import warnings
import math

from io import StringIO
import pyactr as actr
import numpy as np
import pandas as pd

import sys, re, ast, argparse, os, shutil
from multiprocessing import Pool, Process, Queue
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()


class Model(object):
    """
    Model searching and attending to various stimuli.
    """

    def __init__(self, env, target=None, skipifsmall=False, **kwargs):
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

        if target and not skipifsmall:
            self.m.productionstring(name="find_target_probe", string="""
            =g>
            isa     goal
            state   reading
            =visual>
            isa     _visual
            value   {}
            ==>
            =g>
            state   free""".format(target))


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

def activation(a, b):
    # normalize area between -50 to 50
    val = a*b
    val = -np.log(val)
    # b = (b/25 - 1)*100 
    # val = math.exp(math.tanh(a)) + 1/(1 + math.exp((-b/20)))
    return val

def calc_obj_info(split_string, delay_noise=0, fixation_noise=0):
    object_info = []
    for obj in ast.literal_eval(split_string): 
        object_type = obj[0]
        prob = float(obj[1])
        right_X = int(obj[2])
        left_X = int(obj[3])
        bottom_Y = int(obj[4])
        top_Y = int(obj[5])

        mid_X = (left_X + right_X)//2 + int(fixation_noise)
        mid_Y = (top_Y + bottom_Y)//2 + int(fixation_noise)

        if mid_X < 0:
            mid_X = 0
        if mid_Y < 0:
            mid_Y = 0

        width = right_X - left_X
        height = bottom_Y - top_Y
        area = width*height
        # delay = activation(prob/100, math.sqrt(area)) + delay_noise #math.tanh(tmp_var)*3 #-math.log(float(tmp_var))
        # xmnb.append(delay)
        object_info.append([object_type, prob, mid_X, mid_Y, area])
    
    obj = np.array(object_info)
    areas = obj[:, 4].astype(np.float)
    probab = obj[:, 1].astype(np.float)/100 
    areas_norm = areas / np.linalg.norm(areas)
    probab_norm = probab / np.linalg.norm(probab)

    delays = activation(probab_norm, areas_norm)
    for i, delay in enumerate(delays):
        object_info[i].append(delay)
    # print(object_info)
    return object_info

def cal_diff(x):
    if len(x[:, 2]) % 2 == 0:
        x[:, 2] = np.ediff1d(x[:, 2], to_begin=[x[0,2]])
    else:
        x[:, 2] = np.ediff1d(x[:, 2], to_begin=[x[0,2]])
    return x

def get_actr_obj(filename, subject, params):
    df = pd.read_csv(filename, header=None, names=['image', 'image_data'])

    # add guassion noises
    fix_noise = params['fixation'][subject]
    delay_noise = params['delay'][subject]

    df['object_info'] = df.apply(lambda x: calc_obj_info(x['image_data'], delay_noise, fix_noise), axis=1)
    return df[['image', 'object_info']]

def process_actr_data(cmd_str):
    cmd_lst = ast.literal_eval(cmd_str)
    gaze_data= []
    for cmd in cmd_lst:
        try:
            gaze = float(re.findall(r'([0-9]+\.[0-9]*)',cmd)[0])
            screen_x = int(re.findall(r'screen_x=(\s+[0-9]*),', cmd)[0])
            screen_y = int(re.findall(r'screen_y=(\s+[0-9]*),', cmd)[0])
            gaze_data.append([screen_x, screen_y, gaze])
        except:
            print("*** Failed for string ", cmd)
    return gaze_data

def run_simulations(list_of_obj, aspect_ratio=(640, 480), targ=None, focus=None, bias=None, log_file='actr_simulations.log'):
    oldstd = sys.stdout

    if bias:
        dist = [ np.linalg.norm(np.array([obj[2], obj[3]]) - np.array([bias[0], bias[1]])) for obj in list_of_obj]
        dist.sort()
        dist = dist[0]
        if  dist > 200:
            list_of_obj.insert(0, ['center_bias', 99, bias[0], bias[1], 1000, 0.3])
    
    stim_d = {key: {'text':x[0], 'position': (x[2], x[3]), 'vis_delay': x[5]} for key,x in enumerate(sorted(list_of_obj, key=lambda objs: objs[4],reverse=True))}
    sys.stdout = bf = StringIO()
    skiptgtifsmall = True if len(stim_d) < 4 else False

    print("****Running Simulation for target %s with initial focus at %s" %(targ, focus))
    environ = actr.Environment(focus_position=focus, size=aspect_ratio, simulated_display_resolution=aspect_ratio, simulated_screen_size=(60, 34), viewing_distance=60)
    m = Model(environ, target=targ, skipifsmall=skiptgtifsmall ,subsymbolic=True, latency_factor=0.4, decay=0.5, retrieval_threshold=-2, instantaneous_noise=0, automatic_visual_search=True, 
    eye_mvt_scaling_parameter=0.05, eye_mvt_angle_parameter=10, emma_landing_site_noise=True, emma=True) #If you don't want to use the EMMA model, specify emma=False in here
    sim = m.m.simulation(realtime=False, trace=True,  gui=False, environment_process=environ.environment_process, stimuli=stim_d, triggers='X', times=1)
    sim.run(10)
    check = 0

    with open (log_file, 'a') as fd:
        bf.seek(0)
        shutil.copyfileobj(bf, fd)
    fd.close()

    sys.stdout = log_line = StringIO()
    for key in m.dm:
        if key.typename == '_visual':
            print(key, m.dm[key])
            check += 1
    if targ:
        print(key, m.dm[key])
    sys.stdout = oldstd
    # print("sim objects", len(stim_d))
    # print("count ", check)
    gaze_data = str(log_line.getvalue()).split('\n')
    log_line.close()

    if len(gaze_data[-1]) == 0:
        gaze_data.pop()
    return gaze_data


def sim_worker(sub_id, imgs, outpath, display_size, target, focus, bias):

    print("starting process for subject %s"%(sub_id))
    
    path = os.path.join(outpath, 'worker')    
    logdir = os.path.join(outpath, 'logs')

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logfile = os.path.join(logdir, 'actr_simulations_%s.log'%(sub_id))

    df = pd.DataFrame(data=imgs, index=range(len(imgs)), columns=['name','object_info'])
    # df.info()
    df['actr_data'] = df.progress_apply(lambda x: run_simulations(x['object_info'], display_size, target, focus, bias, logfile), axis=1)
    # write data to temporary file; 
    df.to_csv(os.path.join(path, 'actr_temp_%s.csv'%(sub_id)), index=False)
    # read the temporary file
    df =  pd.read_csv(os.path.join(path,'actr_temp_%s.csv'%(sub_id)))
    df['actr_data_processed'] = df.progress_apply(lambda x: process_actr_data(x['actr_data']), axis=1)
    
    df.to_csv(os.path.join(path,'actr_sim_%s_sub_%s.csv' %(target, sub_id)), index=False)
    # delete temporary file
    os.remove(os.path.join(path,'actr_temp_%s.csv'%(sub_id)))

    print("processessing done for subject%s"%(sub_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p", help="path to your detected file")
    parser.add_argument("--target", "-t", help="name of target probe")
    parser.add_argument("--subjects", "-s", help="number of subjects in simulation")

    # read arguments from the command line
    args = parser.parse_args()
    # filepath = args.path if args.path else os.path.join('data', 'salicon/detected', 'salicon_detected_objects.csv')
    filepath = args.path if args.path else os.path.join('data', 'coco_search_18/detected/bowl/concat', 'detected_objects.csv')
    # target for the actr simulation
    target =  args.target.strip() if args.target else None
    # total number of subject
    subjects = int(args.subjects) if args.subjects else 1
    # output folder for worker files
    outpath = os.path.join(os.path.dirname(filepath), '..' ,'..' , '..', 'simulations', target) if target else os.path.join(os.path.dirname(filepath), '..' ,'..' , '..', 'simulations')

    # display size of simulation 
    display_size = {
        'salicon': (640, 480),
        'coco-search-18': (1680, 1050)
    }
    display_size = display_size['coco-search-18'] if args.target else display_size['salicon']

    # focus position if target is present
    focus = (int(display_size[0]/2), int(display_size[1]/2))  if args.target else None
    # introduce center bias if target is present
    bias = focus if args.target else None
    # gaussian noise for encoding time and fixations for each subject
    params = {
        'delay': np.append(np.random.normal(0, 0.01, subjects-1), 0),
        'fixation': np.append(np.random.normal(0, 150, subjects-1), 0),
    }

    root = os.path.join(outpath, 'worker')
    if not os.path.exists(root):
        os.makedirs(root)
    else:
        for fl in os.listdir(root):
            os.remove(os.path.join(root, fl), )

    dfs = [get_actr_obj(filepath, subject, params) for subject, param in enumerate(range(subjects))]
    imgs = [df.to_numpy() for df in dfs]

    processes = [ Process(target=sim_worker, args=(sub_id, imgs[sub_id], outpath, display_size, target, focus, bias)) for sub_id in range(subjects)]
    
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    columns = [ 'sub_%s'%(sub) for sub in range(subjects)]
    pds = [ pd.read_csv(os.path.join(root, f), skipinitialspace=True, usecols=['name', 'actr_data_processed']) for f in os.listdir(root)]

    for id, ps in enumerate(pds):
        pds[id] = ps.rename(columns={'name': 'name_%s'%(id),'actr_data_processed': 'sub_%s'%(id)})

    pds = pd.concat(pds, axis=1)
    pds['agg_res'] = pd.Series([np.empty([0, 3]) for i in range(len(pds))])

    for sub in range(subjects):
        pds['sub_%s'%(sub)] = pds['sub_%s'%(sub)].progress_apply(ast.literal_eval)
        pds['sub_%s'%(sub)] = pds['sub_%s'%(sub)].progress_apply(np.array)
        # calculate the timstamp difference
        pds['sub_%s'%(sub)] = pds.apply(lambda x: cal_diff(x['sub_%s'%(sub)]), axis=1)
        pds['agg_res'] = pds.apply(lambda x: np.vstack((x['agg_res'], x['sub_%s'%(sub)])), axis=1)
    
   
    pds['agg_res'] = pds.apply(lambda x: x['agg_res'].tolist(), axis=1)
    for col in columns:
        pds[col] = pds.apply(lambda x: x[col].tolist(), axis=1)
 
    pds.to_csv(os.path.join(outpath,'actr_aggr_sim_%s.csv' %(target)), index=False)