import os
import sys
import multiprocessing as mp

def create_new_fol(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)

def process(cmd):
    print(cmd)
    os.system(cmd)

out_dir = os.environ.get('OUT_DIR')
if out_dir is None:
   out_dir = "../data/output" 

heatmap_name = os.environ.get('HEATMAP_VERSION')
if heatmap_name is None:
   heatmap_name = 'cancer-prad-3c'

parent_in = str(out_dir)+'/heatmap_txt_3classes_separate_class'
seperate_in = [fol for fol in os.listdir(parent_in) if fol.startswith('heatmap_txt')]

log_fol = str(out_dir)+'/log/'
heatmap_versions = [str(heatmap_name) + '_' + f.split('_')[-1] + '_heatmap' for f in seperate_in]
heatmap_txt_in = [os.path.join(parent_in, fol) for fol in seperate_in]

cmd_list = []
for version, txt in zip(heatmap_versions, heatmap_txt_in):
    log_file = log_fol + 'log.heatmap_jsons_' + txt.split('/')[-1]
    cmd = 'bash ./gen_all_json.sh {} {} > {} 2>&1'.format(version, txt, log_file)
    print("COMMAND: ",cmd)
    cmd_list.append(cmd)

print(len(cmd_list))
pool = mp.Pool(processes=8)
pool.map(process, cmd_list)

