N_GPU=4
BATCH_SIZE=4_096
BBOX_SIZE=20

python3.8 -m nerf.run -w train   -b ${BATCH_SIZE} -l ${N_GPU} -s ${BBOX_SIZE}
python3.8 -m nerf.run -w reptile -b ${BATCH_SIZE} -l ${N_GPU} -s ${BBOX_SIZE}
python3.8 -m nerf.run -w distill -b ${BATCH_SIZE} -l ${N_GPU} -s ${BBOX_SIZE}