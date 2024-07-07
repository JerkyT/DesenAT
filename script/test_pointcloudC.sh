Str=("add_global" "add_local" "dropout_global" "dropout_local" "jitter" "rotate" "scale")
for i in ${Str[@]}
do
for j in $(seq 0 4)
do
CUDA_VISIBLE_DEVICES=2 python ./examples/classification/main.py \
                            --cfg ./cfgs/pointcloud_c/pointnet++.yaml \
                            --dataset.common.corruption $i \
                            --dataset.common.severity $j \
                            --pretrained_path './log/modelnet40ply2048/DesenAT_pointnet2/checkpoint/best.pth';
done
done