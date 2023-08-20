# Setting
GPUID=0

model_name="D5C5"
dataset_name="RBHTDTCMR2023A"

# MASK
#mask_name="fMRI_Reg_AF2_CF0.16_PE48"
#mask_name="fMRI_Reg_AF4_CF0.08_PE48"
mask_name="fMRI_Reg_AF8_CF0.04_PE48"
#mask_name="fMRI_Reg_AF16_CF0.02_PE48"

# CPHASE
disease="MI"
# DISEASE
cphase="diastole"
#cphase="systole"


task_name=${model_name}_${dataset_name}_${mask_name}_${disease}_${cphase}
log_name=log_test_${task_name}

# Run
rm ${log_name}

CUDA_VISIBLE_DEVICES=$GPUID \
PYTHONPATH=$(pwd) \
nohup python test_DCCNN_D5C5_RBHTDTCMR2023A.py \
--task_name ${task_name} \
--data_path /media/ssd/data_temp/RBHT/DT_CMR_data/RBHT_DTCMR_2023A/d.1.0 \
--disease ${disease} \
--cphase ${cphase} \
--weight_path /media/NAS01/jiahao/DCCNN/RBHT_DTCMR_2023A/ \
--num_epoch 21 \
--batch_size 1 \
--lr 0.001 \
--l2 1e-6 \
--undersampling_mask ${mask_name} \
--resolution_h 256 \
--resolution_w 96 \
>> ${log_name} &
