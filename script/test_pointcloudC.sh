Str=("add_global" "add_local" "dropout_global" "dropout_local" "jitter" "rotate" "scale")
for i in ${Str[@]}
do
for j in $(seq 0 4)
do
CUDA_VISIBLE_DEVICES=2 python ./examples/classification/main.py \
                            --cfg ./cfgs/pointcloud_c/pointnet++.yaml \
                            --dataset.common.corruption $i \
                            --dataset.common.severity $j \
                            --pretrained_path '/home/liweigang/PointMetaBase/log/modelnet40ply2048/LES_SpaceT_sD_pointnet2/checkpoint/modelnet40ply2048-train-pointnet++-ngpus1-seed9333-20240402-213222-Yr2DsU3x7YH6n9vULeU4Cf_ckpt_best.pth';
done
done

# CUDA_VISIBLE_DEVICES=3 python ./examples/classification/main.py --cfg ./cfgs/modelnet40C6/apes_global.yaml