Str=("background" "cutout" "density" "density_inc" "distortion" "distortion_rbf" "distortion_rbf_inv" "gaussian" "impulse" "rotation" "shear" "uniform" "upsampling"  "occlusion" "lidar")

for i in ${Str[@]}
do
for j in $(seq 1 5)
do
CUDA_VISIBLE_DEVICES=5 python ./examples/classification/main.py \
                            --cfg ./cfgs/modelnet40C/pointnet++.yaml \
                            --dataset.common.corruption $i \
                            --dataset.common.severity $j \
                            --pretrained_path '/home/liweigang/PointMetaBase/log/modelnet40ply2048/SK_SGE_SPT_pointnet2_DKD/checkpoint/modelnet40ply2048-train-pointnet++-ngpus1-seed9333-20240630-214608-5HY3fYvJPVerSV2CKXLB6z_ckpt_best.pth';
done
done

# CUDA_VISIBLE_DEVICES=5 python ./examples/classification/main.py --cfg ./cfgs/modelnet40ply2048/pointnet++.yaml