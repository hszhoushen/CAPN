[TOC]

# Relation-aware Mixture of Experts

## Detector Training

**Detector training**

```
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR ./checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```



## Perform training on Scene Graph Generation



There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches `MODEL.ROI_RELATION_HEAD.USE_GT_BOX` and `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL` to select the protocols.

For **Predicate Classification (PredCls)**, we need to set:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```

For **Scene Graph Classification (SGCls)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

For **Scene Graph Detection (SGDet)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

<h3> Motif Model (focal_loss)

**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 20045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-cross_entropy 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --master_port 20086 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-cross_entropy 
```





**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 60035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-cross_entropy
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 20087 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-cross_entropy
```





**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-cross_entropy
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8 python -m torch.distributed.launch --master_port 20088 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CROSS_ENTROPY_LOSS --num_experts 1  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-cross_entropy
```

 cross_entropy, label_smoothing, focal_loss, ldam_loss

<h3> Motif Model (ldam_loss)

**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --master_port 10045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option ldam_loss MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ldam
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=4 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ldam 
```

**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --master_port 10035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option ldam_loss MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ldam 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option ldam_loss MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-sgcls-ldam OUTPUT_DIR ./checkpoints/motif-sgcls-ldam
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option ldam_loss MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ldam --loss_option ldam_loss
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option ldam_loss MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-precls-ldam OUTPUT_DIR ./checkpoints/motif-precls-ldam
```





## Motifs

### Motif Model (ME2)


**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-2
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-2
```



**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-2 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-2
```



**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-split-2
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-split-2
```





### Motif Model (RAME-2)

**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 10045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-e2-gating
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-e2-gating
```



**scene graph training 2**  (SGCls, Motif Model) mgsgg3

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-e2-gating
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-e2-gating
```



**scene graph training 1**  (PreCls, Motif Model) mgsgg2

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-e2-gating
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-e2-gating
```



### Motif Model (ME-3)


**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 3 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-3
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 3 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-sgdet-ride-split-3 OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-3
```

**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=7,8 python -m torch.distributed.launch --master_port 50035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 3  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-3
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=7,8 python -m torch.distributed.launch --master_port 50036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 3 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-sgcls-ride-split-3 OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-3
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 30085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 3 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-split-3
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 3 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-precls-ride-split-3 OUTPUT_DIR ./checkpoints/motif-precls-ride-split-3
```



### Motif Model (ME-4)


 **scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-4
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-split-4
```

**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-4
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-sgcls-ride-split-4 OUTPUT_DIR ./checkpoints/motif-sgcls-ride-split-4
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-split-4
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-split-4
```



### Motif Model (RAME-4)


 **scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-e4-gating
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgdet-ride-e4-gating
```



**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-e4-gating
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-sgcls-ride-e4-gating
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30085 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-e4-gating
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 12027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-ride-e4-gating
```



### Motifs with TDE (CROSS_ENTROPY_LOSS)


Training Example 1 : (SGDet, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10066 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-sgdet-cross-entropy
```

Test Example 1 : (SGDet, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10038 --nproc_per_node=4 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-sgdet-cross-entropy
```

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10067 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-ride
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --master_port 10037 --nproc_per_node=4 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-cross-entropy
```

Training Example 3 : (PredCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10068 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-precls-cross-entropy
```

Test Example 2 : (PredCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=4,9 python -m torch.distributed.launch --master_port 10039 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-precls-cross-entropy
```





<h3> VC Tree (CROSS_ENTROPY_LOSS)

**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30055 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-cross-entropy
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-cross-entropy
```



**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 80065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-cross-entropy
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-cross-entropy 
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-cross-entropy
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-cross-entropy
```



<h3> VC Tree (Focal_loss)

**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30055 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-focal 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-focal
```



**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=7,9 python -m torch.distributed.launch --master_port 80065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-focal 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-focal 
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-focal
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option FOCAL_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-focal
```



<h3> VC Tree (LDAM_LOSS)

**scene graph training 3**  (SGDet)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30055 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ldam 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --master_port 30046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ldam 
```

**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=4,7 python -m torch.distributed.launch --master_port 10065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ldam
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ldam
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ldam 
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option LDAM_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ldam 
```





## VC Tree

### VC Tree(RAME-2)


**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 30055 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-rame-2
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30066 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-rame2
```

**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-me2-gating 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-me2-gating 
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-me2-gating   
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-me2-gating   
```



### VC Tree(ME2)

**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 30055 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ride-split-2 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30066 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ride-split-2 
```

**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ride-split-2 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ride-split-2 
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ride-split-2  
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ride-split-2  
```



MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True



### VC Tree(RAME-4)

**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=2,3,8,9 python -m torch.distributed.launch --master_port 60055 --nproc_per_node=4 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-RAME-4 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30046 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 6 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-RAME-4
```

**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-RAME-4 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-RAME-4
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-RAME-4
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-RAME-4  
```



### VC Tree(ME4)

**scene graph training 3**  (SGDet)


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 60055 --nproc_per_node=4 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ride-split-4 
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30046 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 6 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-ride-split-4
```

**scene graph training 2**  (SGCls)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 30065 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ride-split-4 
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-ride-split-4
```

**scene graph training 1**  (PreCls)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ride-split-4
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 30075 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-ride-split-4  
```



### VCTree with TDE (CROSS_ENTROPY_LOSS)


Training Example 1 : (SGDet, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10066 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-casual
```

Test Example 1 : (SGDet, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10038 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgdet-casual
```

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10067 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-casual
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10037 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree TEST.IMS_PER_BATCH 4 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-sgcls-casual
```

Training Example 3 : (PredCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10068 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-causal
```

Test Example 2 : (PredCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --master_port 10039 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree TEST.IMS_PER_BATCH 2 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/vctree-precls-causal
```









## Transformer

<h3> Transformer Model (CROSS_ENTROPY_LOSS)

**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 30045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgdet-cross-entropy
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=4 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 8 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgdet-cross-entropy 
```

**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgcls-cross-entropy
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 8 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgcls-cross-entropy
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 30025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-precls-cross-entropy 
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 8 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-precls-cross-entropy 
```



Transformer

<h3> Transformer Model (RIDE_LOSS)

**scene graph training 3**  (SGDet, Motif Model)

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10045 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgdet-ride-e4
```

**Evaluation 3**

```
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 10046 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgdet-cross-entropy 
```

**scene graph training 2**  (SGCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 10035 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgcls-ride-e4
```

**Evaluation 2**

```
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 10036 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  --loss_option CROSS_ENTROPY_LOSS --num_experts 1 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-sgcls-cross-entropy
```

**scene graph training 1**  (PreCls, Motif Model)

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option RIDE_LOSS --num_experts 4 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-precls-ride-e4
```

**Evaluation 1**

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 110027 --nproc_per_node=2 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   --loss_option CROSS_ENTROPY_LOSS --num_experts 2 MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 12 DTYPE "float16" GLOVE_DIR ./glove MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/transformer-precls-ride-e4
```

