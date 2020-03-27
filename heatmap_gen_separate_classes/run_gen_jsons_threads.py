import os
import sys

def create_new_fol(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)

parent_in = '../data/heatmap_txt_3classes_separate_class'
seperate_in = [fol for fol in os.listdir(parent_in) if fol.startswith('heatmap_txt')]

log_fol = '../data/log/'
heatmap_versions = ['prad-3c_' + f.split('_')[-1] + '_032720_hanle' for f in seperate_in]
heatmap_txt_in = [os.path.join(parent_in, fol) for fol in seperate_in]

for version, txt in zip(heatmap_versions, heatmap_txt_in):
    log_file = log_fol + 'log.heatmap_jsons_' + txt.split('/')[-1]
    cmd = 'nohup bash gen_all_json.sh {} {} &> {} &'.format(version, txt, log_file)
    print(cmd)
    os.system(cmd)


