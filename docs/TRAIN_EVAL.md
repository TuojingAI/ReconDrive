# Train/Eval Models

## Trained Checkpoints 

We provide all trained checkpoints here.

| Model | stage-1 | final stage-2 |
| :---: | :---: | :---: |
| Checkpoints | [recondrive_stage1.ckpt](https://huggingface.co/TuojingAI/ReconDrive/blob/main/recondrive_stage1.ckpt)  | [recondrive_stage2.ckpt](https://huggingface.co/TuojingAI/ReconDrive/blob/main/recondrive_stage2.ckpt) |
|Checkpoints-md5 hash| a429e3a3ea03d0bab1579d099cfff2c8 | fd6ed379f136b3c17d0e27a5aec8c0b7 |

## Training
ReconDrive training consists of two main stages. Pretrained and trained checkpoints for stage 2, are listed below.

### Stage-two: ReconDrive Training with Two-Frame Inputs

In the second stage, we train the final ReconDrive models.
  ```
  # Firstly, download the pretrained stage1 checkpoints and link it to $pretrained_stage1_checkpoint=./checkpoints/recondrive_stage1.ckpt
  CUDA_VISIBLE_DEVICES=${GPU_IDs} bash scripts/train.sh ${GPU_NUM}
  ```


## Evaluation

  ```
  # Download the recondrive_stage2.ckpt and link it to ./checkpoints/recondrive_stage2.ckpt
  # Currently, inference only support single-GPU mode
  CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/inference.sh
  ```