## Generate shapely values ​​through PointNet++:
CUDA_VISIBLE_DEVICES=0 python ./examples/classification/main.py \
                            --cfg ./cfgs/modelnet40ply2048/pointnet++.yaml \
                            --datatransforms.c None \
                            --criterion_args.NAME  SmoothCrossEntropy \
                            --dataloader.gen_shaply True \
                            --dataloader.shapley True \
                            --mode train \
                            --pretrained_path './log/modelnet40ply2048/ST_pointnet2/checkpoint/best.pth'
