#!/bin/bash
# YHW 2023.3.2

# targe number of frames 
N=1

rand_seed=217828

# target SNR
snr=0

# total number of output EM images
L=64
# max_shift_pixels=4
max_shift_pixels=0
apix=3.0
echo "apix = $apix"

dir_out="sim/c1/$N"
mkdir -p $dir_out

if [[ $snr -gt 0 ]]; then
	echo ">>> SNR: $snr"
	prefix="7bcq_c1_n${N}_ctf_snr${snr}_maxshift${max_shift_pixels}px"
else
	echo ">>> SNR: noise-free"
	prefix="7bcq_c1_n${N}_ctf_noise-free_maxshift${max_shift_pixels}px"
fi

echo ">>> prefix $prefix"


full_basename="${dir_out}/${prefix}"
short_basename="${prefix}"
input_ctf="ctf/80s_10028_d128_downsample_ctf.pkl"

echo "num. angles $num_conformations"
echo "window: $window_width"


# make projection images
input_vols=(
				vols/7bcq_angle_n1_cen_d64/frames/7bcq_angle_n1_cen_d64_w5000_r2.0_res4_fr1_origin.mrc
)
ctf_pool="ctf/80s_10028_d128_downsample_ctf.pkl"
num_frames=$N

output_mrcs=${full_basename}.mrcs
output_pose=${full_basename}_pose.pkl
output_pose_cdrgn=${full_basename}_pose_cdrgn.pkl
output_ctf=${full_basename}_ctf_cdrgn.pkl
output_sample_proj=${full_basename}_sample_proj.png

echo "-------------------------------"
echo "make simulated stack images"
echo "-------------------------------"
python bin/sim_proj_from_vol.py \
        --vols ${input_vols[@]} \
        --ratio ${ratio} \
        --ctf-pool ${ctf_pool} \
        --num-frames ${num_frames} \
        --output-mrcs ${output_mrcs} \
        --output-pose ${output_pose} \
        --output-ctf-cdrgn ${output_ctf} \
        --output-pose-cdrgn ${output_pose_cdrgn} \
        --output-sample-proj ${output_sample_proj} \
        --max-shift-pixels ${max_shift_pixels} \
        --snr ${snr} \
        --use-ctf \
        --rand-seed ${rand_seed} \

# star
output_star=${full_basename}.star
echo "-------------------------------"
echo "make star file: $output_star"
echo "-------------------------------"
python bin/convert_pose_pkl_to_star.py \
        -i ${output_pose} \
        -o ${output_star} \
        --apix  $apix \
        --mrcs  $(basename $output_mrcs)\
        --box-size $L \
        --ctf ${output_ctf}

# lmdb
output_lmdb=${full_basename}.lmdb
echo "-------------------------------"
echo "make lmdb file: $output_lmdb"
echo "-------------------------------"
python bin/mrcs2lmdb.py \
    -i ${output_mrcs} \
    -o ${output_lmdb}

# cs
output_cs=${full_basename}.cs
echo "-------------------------------"
echo "make cs file: $output_cs"
echo "-------------------------------"
python bin/convert_pose_pkl_to_cs.py \
        -i ${output_pose} \
        -o ${output_cs} \
        --apix  $apix \
        --mrcs  $(basename $output_mrcs) \
        --box-size $L \
        --ctf ${output_ctf}


