# Based on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
# Adapted by Ayca Takmaz, July 2023

import os
import numpy as np
from benchmark_scripts.eval_utils.eval_script_inst import evaluate

def main(pred_dir, gt_dir):
    scene_names = sorted(el[:-4] for el in os.listdir(gt_dir) if el.endswith('.txt'))

    preds = {}
    for scene_name in scene_names[:]:
        #print('<'*50)
        #print(scene_name)

        file_path = os.path.join(pred_dir, scene_name+'.txt')  # txt file
        scene_pred_mask_list = np.loadtxt(file_path, dtype=str)  # (num_masks, 2)
        assert scene_pred_mask_list.shape[1] == 2, 'Each line should have 2 values: instance mask path and confidence score.'

        pred_masks = []
        pred_scores = []
        for mask_path, conf_score in scene_pred_mask_list: 
            pred_mask = np.loadtxt(os.path.join(pred_dir, mask_path), dtype=int) # values: 0 for the background, 1 for the instance
            pred_masks.append(pred_mask)
            pred_scores.append(float(conf_score))

        assert len(pred_masks) == len(pred_scores), 'Number of masks and confidence scores should be the same.'
        
        # we need to catch cases where no object instances are predicted. for those cases, we produce a pseudo map
        preds[scene_name] = {
            'pred_masks': np.vstack(pred_masks).T if len(pred_masks) > 0 else np.zeros((1, 1)),
            'pred_scores': np.ones(len(pred_masks)) if len(pred_masks) > 0 else np.zeros(1),
            'pred_classes': np.ones(len(pred_masks), dtype=np.int64) if len(pred_masks) > 0 else np.ones(1, dtype=np.int64)
        }

    ap_dict = evaluate(preds, gt_dir)
    del ap_dict['classes']
    print(ap_dict)
 
if __name__=='__main__':
    pred_dir = "PATH/TO/RESULTS" # folder containing <SCENE_ID>.txt files for predictions
    gt_dir = "PATH/TO/GT" # folder containing <SCENE_ID>.txt files for gt annotations
    main(pred_dir, gt_dir)
