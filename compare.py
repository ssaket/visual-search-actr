import pandas as pd
import numpy as np
import os, ast, json
from matplotlib import colors
import matplotlib.pyplot as plt
import multimatch_gaze as m
from scipy.io import savemat, loadmat

from tqdm import tqdm

import warnings, argparse
warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()

def read_coco_json(filepath):

    dt = None
    with open(filepath,) as f: 
        dt = json.load(f)
        f.close()
    return dt

def find_coco_target(coco_dicts, target, name, split=None):
    res = []
    for cdi in coco_dicts:
        for obj in cdi:
            if obj['task'] == target and obj['name'] == name:
                if split and obj['split'] == split:
                    res.append(obj)
                elif not split:
                    res.append(obj)
        # print("searching in next file")
    return res
def find_agg_gtruth(glst):
    pma = np.empty([0, 3])
    for ob in glst:
        x = np.array(np.column_stack((ob['X'], ob['Y'], ob['T'])))
        pma = np.vstack((pma, x))
    return pma

def compare_agg_diff(gtagg, clagg, usenorm=True):

    if(gtagg.size == 0):
        return 'No match'
    clagg[:,2] = clagg[:,2]*1000

    gtagg_norm = gtagg / gtagg.max(axis=0)
    clagg_norm = clagg / clagg.max(axis=0)

    if usenorm:
        gtagg_mean = np.mean(gtagg_norm, axis=0)
        clagg_mean = np.mean(clagg_norm, axis=0)
    else:
        gtagg_mean = np.mean(gtagg, axis=0)
        clagg_mean = np.mean(clagg, axis=0)

    diff = gtagg_mean - clagg_mean
    return diff.reshape(-1, 3)

def cmp_multimatch(target, row, gtruth, name, screensize, split=('train', 'valid')):
    # print("calculating for image %s"%(name))

    dt = {'names': ('start_x', 'start_y', 'duration'),'formats': ('f8', 'f8', 'f8')}
    res_aggr = np.empty([0, 5])

    for key in row.keys():
        # print("calculating for subject %s" %(key))
        # print("No of ground truth data %s"%(len(gtruth)))
        clagg = np.zeros(row[key].shape[0],  dtype=dt)
        clagg['start_x'] = row[key][:,0]
        clagg['start_y'] = row[key][:,1]
        # convert to miliseconds
        clagg['duration'] = row[key][:,2]*1000

        for obj in gtruth:
            gtagg = np.zeros(len(obj['X']),  dtype=dt)
            gtagg['start_x'] = np.array(obj['X'])
            gtagg['start_y'] = np.array(obj['Y'])
            gtagg['duration'] = np.array(obj['T'])

            # lst = np.vstack((lst, compare_agg_diff(gtagg, row[key])))
            doc = m.docomparison(gtagg, clagg, screensize=screensize)
            res_aggr = np.vstack((res_aggr, doc))
        # plot_result(lst, target, key, name, bins=5)
    # print("done")
    # plot_multimatch("Multimatch for image %s on target %s"%(name, target), res_aggr, 10)
    return res_aggr

def cmp_scanmatch(target, row, gtruth, name, split=('train', 'valid')):

    res_aggr = np.empty([0, 6])

    for key in row.keys():
        for obj in gtruth:
            shape = (row[key].shape[0],3) if (row[key].shape[0] > len(obj['X'])) else (len(obj['X']), 3)

            clagg = np.zeros(shape)
            clagg.fill(0.1)
            clagg[:row[key].shape[0],0] = row[key][:,0]
            clagg[:row[key].shape[0],1] = row[key][:,1]
            # convert to miliseconds
            clagg[:row[key].shape[0],2] = row[key][:,2]*1000


            gtagg = np.zeros(shape)
            gtagg.fill(0.1)
            gtagg[:len(obj['X']),0] = np.array(obj['X'])
            gtagg[:len(obj['X']),1] = np.array(obj['Y'])
            gtagg[:len(obj['X']),2] = np.array(obj['T'])

            doc = np.hstack((clagg, gtagg))
            res_aggr = np.vstack((res_aggr, doc))
    
    mdic = {"data1": res_aggr[:,[0,1,2]], "data2":  res_aggr[:,[3,4,5]], "label": "scanmatch"}
    filepath = os.path.join('results', 'coco-search-18', target)
    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    filename = os.path.join(filepath, "matlab_matrix_%s.mat"%(name.split('.')[0]))
    savemat(filename, mdic)

    return res_aggr



def compare_sub_diff(target, row, gtruth, name, split=('train', 'valid')):
    
    # print("calculating for image %s"%(name))
    for key in row.keys():
        # print("calculating for subject %s" %(key))
        lst = np.empty([0, 3])
        # print("No of ground truth data %s"%(len(gtruth)))
        for obj in gtruth:
            gtagg = np.empty([len(obj['X']), 3])
            gtagg[:,0] = np.array(obj['X'])
            gtagg[:,1] = np.array(obj['Y'])
            gtagg[:,2] = np.array(obj['T'])

            lst = np.vstack((lst, compare_agg_diff(gtagg, row[key])))
        # plot_result(lst, target, key, name, bins=5)

def plot_multimatch(title, mlarr, bins=20):
    fig, ax = plt.subplots(1,1, tight_layout=True)
    fig.suptitle(title, fontsize=16)
    ax.hist(mlarr[:,0], bins=bins, alpha=0.5, label='vector', histtype='stepfilled')
    ax.hist(mlarr[:,1], bins=bins, alpha=0.5, label='direction', histtype='stepfilled')
    ax.hist(mlarr[:,2], bins=bins, alpha=0.5, label='length', histtype='stepfilled')
    ax.hist(mlarr[:,3], bins=bins, alpha=0.5, label='position', histtype='stepfilled')
    ax.hist(mlarr[:,4], bins=bins, alpha=0.5, label='duration', histtype='stepfilled')
    ax.legend(loc='upper right')
    plt.show()

def plot_result(n_arr, target, subject='all', name='all', bins=20):
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(1,3, tight_layout=True)
    ax[0].hist2d(n_arr[:,0], n_arr[:,1], norm=mcolors.PowerNorm(0.8))
    ax[0].set_title('Coordinates heatmap')
    ax[0].set_xlabel('difference in x')
    ax[0].set_ylabel('difference in y')

    fig.suptitle('Comparision on image %s on target %s for subject %s'%(name, target, subject), fontsize=12)
  
    ax[1].hist(n_arr[:,2], bins=bins)
    ax[1].set_title('Fixations')
    ax[1].set_xlabel('difference in milisecs')

    ax[2].boxplot(n_arr)
    ax[2].set_title('Variance')
    ax[2].set_xticklabels(['x-diff', 'y-diff', 'time-diff']) 
    plt.show()

def time_diff(gtagg, clagg):

    if(gtagg.size == 0):
        gtagg = np.zeros(clagg.shape)
        # return 'no value'
    
    y1 = gtagg[:,2]
    x1 = np.arange(len(y1))
    y2 = clagg[:,2]*1000
    x2 = np.arange(len(y2))

    return [y1, y2]

def start_processing_salicon( file, fixs):
    
    df = pd.read_csv(file)
    
    display_size = {
        'salicon': (640, 480),
        'coco-search-18': (1680, 1050)
    }
    screensize = display_size['salicon']
    columns = [ key for key in df.keys() if key.startswith('sub') or key.startswith('agg')]
    names = [ key for key in df.keys() if key.startswith('name')]
    
    for col in columns:
        df[col] = df[col].apply(ast.literal_eval)
        df[col] = df[col].apply(np.array)

    # delete extra name columns
    for i in range(len(names)-1):
        if len(np.where(df['name_0'] != df[names[i+1]])) == 1:
            df = df.drop(columns=[names[i+1]])

    fix_mat = loadmat(fixs)
    gaze_data = fix_mat['arr_0']
    indices = [2422, 2800, 2971, 4162, 4697, 1212, 4378, 790, 2768, 835, 2838, 3317, 3755, 4341, 3320, 4396, 4935, 3673, 1334, 4599, 1401, 1427, 3497, 4852, 3441, 2986, 200, 3804, 871, 2751, 774, 1156, 1488, 310, 2692, 143, 452, 4868, 2545, 1999, 2464, 998, 3054, 4797, 4160, 319, 2450, 4356, 4372, 4609]
    gaze_data = np.delete(gaze_data, indices, 0)

    dt = {'names': ('start_x', 'start_y', 'duration'),'formats': ('f8', 'f8', 'f8')}
    mscores = np.empty([0, 5])
   
    for col in columns[0:-1]:
        mgdata = df[col].to_numpy()

        for i in tqdm(range(len(mgdata))):
            res_aggr = np.empty([0, 6])
            csub = mgdata[i]  / mgdata[i].max(axis=0)
            clagg = np.zeros(csub.shape[0],  dtype=dt)
            clagg['start_x'] = csub[:,0]
            clagg['start_y'] = csub[:,1]
            clagg['duration'] = csub[:,2]

            cgzdata = gaze_data[i]
            gtagg =  np.zeros(cgzdata.shape[0],  dtype=dt)
            gtagg['start_x'] = cgzdata[:,0]
            gtagg['start_y'] = cgzdata[:,1]
            gtagg['duration'] =cgzdata[:,2]
            doc = m.docomparison(clagg, gtagg, screensize=screensize)
            mscores = np.vstack((mscores, doc))

            shape = (csub.shape[0],3 ) if (csub.shape[0] > cgzdata.shape[0]) else (cgzdata.shape[0], 3)

            clagg = np.zeros(shape)
            clagg.fill(0.1)
            clagg[:csub.shape[0],0] =  csub[:,0]
            clagg[:csub.shape[0],1] =  csub[:,1]
            clagg[:csub.shape[0],2] =  csub[:,2]


            gtagg = np.zeros(shape)
            gtagg.fill(0.1)
            gtagg[:cgzdata.shape[0],0] = cgzdata[:,0]
            gtagg[:cgzdata.shape[0],1] = cgzdata[:,1]
            gtagg[:cgzdata.shape[0],2] = cgzdata[:,2]
            
            doc = np.hstack((clagg, gtagg))
            res_aggr = np.vstack((res_aggr, doc))
    
            mdic = {"data1": res_aggr[:,[0,1,2]], "data2":  res_aggr[:,[3,4,5]], "label": "scanmatch"}
            filepath = os.path.join('results', 'salicon', 'image_%s'%(i))
            if not os.path.isdir(filepath):
                os.makedirs(filepath)

            filename = os.path.join(filepath, "matlab_matrix_%s_image_%s.mat"%(col, i))
            savemat(filename, mdic)
   
    return mscores
    

def start_processing_coco(target, file, fixs):

    df = pd.read_csv(file)
    
    display_size = {
        'salicon': (640, 480),
        'coco-search-18': (1680, 1050)
    }

    columns = [ key for key in df.keys() if key.startswith('sub') or key.startswith('agg')]
    names = [ key for key in df.keys() if key.startswith('name')]
    
    for col in columns:
        df[col] = df[col].apply(ast.literal_eval)
        df[col] = df[col].apply(np.array)

    # delete extra name columns
    for i in range(len(names)-1):
        if len(np.where(df['name_0'] != df[names[i+1]])) == 1:
            df = df.drop(columns=[names[i+1]])

    # df.info()
    coco_gth = [read_coco_json(fl) for fl in coco_fixs ]
    df['gtruth'] = df.progress_apply(lambda x: find_coco_target(coco_gth, target, x['name_0']), axis=1)
    df['gtruth_aggr'] = df.progress_apply(lambda x: find_agg_gtruth(x['gtruth']), axis=1)
      
    tdiffs = df.progress_apply(lambda x: time_diff(x['gtruth_aggr'], x['agg_res']), axis=1)
   
    diffs = df.progress_apply(lambda x: compare_agg_diff(x['gtruth_aggr'], x['agg_res']), axis=1).to_numpy()

    n_arr = np.empty([0, 3])
    for arr in diffs:
        if not isinstance(arr, str):
            arr = arr.reshape(1,3)
            n_arr = np.vstack((n_arr, arr))
    
    # plot_result(n_arr, target)
    # df.progress_apply(lambda x: compare_sub_diff(target, x[columns[:-1]], x['gtruth'], x['name_0']), axis=1)
    df['multimatch'] = df.progress_apply(lambda x: cmp_multimatch(target, x[columns[:-1]], x['gtruth'], x['name_0'], display_size['coco-search-18']), axis=1)
    mps = df['multimatch'].to_numpy()
    n_arr = np.empty([0,5])
    for mp in mps:
        n_arr = np.vstack((n_arr, mp))
    
    # plot_multimatch("Multimatch for target %s for all subjects for all images"%(target), n_arr, 100)
    multimatch_score = np.nanmean(n_arr, axis=0)
    print("done score is ", multimatch_score)

    df.progress_apply(lambda x: cmp_scanmatch(target, x[columns[:-1]], x['gtruth'], x['name_0']), axis=1)
    

    nt_arr = np.empty([0, 3])
    for i, arr in enumerate(tdiffs):
        if not isinstance(arr, str):
            lnt = len(arr[0]) if len(arr[0]) > len(arr[1]) else len(arr[1])
            arp = np.zeros([lnt, 3])
            arp[:len(arr[0]),0] = arr[0]
            arp[:len(arr[1]),1] = arr[1]
            arp[:,2] = np.full((lnt), i)
            nt_arr = np.vstack((nt_arr, arp))

    fig, ax = plt.subplots(1,1, tight_layout=False)
    ax.set_title("Fixation duration of all images for target %s"%(target))
    ax.plot(nt_arr[:,0], c='tab:orange', label='base', alpha=0.5)
    ax.plot(nt_arr[:,1], c='tab:blue', label='our', alpha=0.5)
    ax.legend()

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    # plt.show()

    return multimatch_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", "-d", help="path to simulations")
    parser.add_argument("--target", "-t", default=None,  help="name of target probe")
    parser.add_argument("--run", "-r",  default='salicon', choices=['coco', 'salicon'], help="run for dataset")

    # read arguments from the command line
    args = parser.parse_args()

    target = args.target
    multimatch_score = np.empty([0, 5])

    if  args.run == 'salicon':
        salicon_dir = os.path.join(args.dir)
        salicon_file = os.path.join(salicon_dir, 'actr_aggr_sim_%s.csv'%(target))
        salicon_fixs = os.path.join('results', 'path_gan_generated_fixations_scanmatch.mat')
        mscores = start_processing_salicon(salicon_file, salicon_fixs)

        plot_multimatch("Multimatch for Salicon Val", mscores, 20)
        print("ovelall salicon score ", np.nanmean(mscores, axis=0))
    else:
        if not target:
            for target in os.listdir(args.dir):
                try:
                    coco_dir = os.path.join(args.dir, target)
                    coco_file = os.path.join(coco_dir, 'actr_aggr_sim_%s.csv'%(target))
                    coco_fixs = [ os.path.join('data', 'coco_search_18', f) for f in os.listdir(os.path.join('data', 'coco_search_18')) if f.endswith('.json') ]

                    multi_score = start_processing_coco(target, coco_file, coco_fixs)
                    multimatch_score = np.vstack((multimatch_score, multi_score))
                    print("ovelall score ", np.mean(multimatch_score, axis=0))
                except:
                    pass
        else:
            coco_dir = os.path.join(args.dir, target)
            coco_file = os.path.join(coco_dir, 'actr_aggr_sim_%s.csv'%(target))
            coco_fixs = [ os.path.join('data', 'coco_search_18', f) for f in os.listdir(os.path.join('data', 'coco_search_18')) if f.endswith('.json') ]
            start_processing_coco(target, coco_file, coco_fixs)
