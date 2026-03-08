#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import shutil
import os

PIPELINE_DEPLOYMENT = [
    'dataset/vggt3dgs_data_module.py',
    'dataset/nuscenes_dataset.py',
    'models/vggt3dgs_model_module.py',
    'models/vggt3dgs_model.py',
]
def save_pipeline_snapshot(pipeline_files, snapshot_dir=""):
    """
    Save complete pipeline snapshot (code + model)

    :param pipeline_files: list of file paths to save
    :param model_dir: snapshot storage directory
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    for file_path in pipeline_files:
        if os.path.isfile(file_path):
            rel_path = os.path.relpath(file_path, start=os.getcwd())
            dest_path = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)

    return snapshot_dir




if __name__ == "__main__":

    save_pipeline_snapshot(pipeline_files=PIPELINE_DEPLOYMENT,snapshot_dir='test_dirs/test_code')