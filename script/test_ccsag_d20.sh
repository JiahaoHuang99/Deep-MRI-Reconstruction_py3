# Setting
GPUID=0

model_name="D5C5"
dataset_name="CCSAG_D20"

#mask_name="fMRI_Ran_AF4_CF0.08_PE256"
#mask_name="fMRI_Ran_AF8_CF0.04_PE256"
#mask_name="fMRI_Ran_AF16_CF0.02_PE256"
#mask_name="fMRI_Reg_AF4_CF0.08_PE256"
#mask_name="fMRI_Reg_AF8_CF0.04_PE256"
#mask_name="fMRI_Reg_AF16_CF0.02_PE256"
mask_name="radial_add_10_res320"
#mask_name="spiral_add_10_res320"

task_name=${model_name}_${dataset_name}_${mask_name}
log_name=log_test_${task_name}

# Run
rm ${log_name}

CUDA_VISIBLE_DEVICES=$GPUID \
PYTHONPATH=$(pwd) \
nohup python test_DCCNN_D5C5_CCSAG_D20.py \
--task_name ${task_name} \
--data_path_train /media/ssd/data_temp/CC/SAGITTAL/d.2.0.sc/train \
--data_path_val /media/ssd/data_temp/CC/SAGITTAL/d.2.0.sc/val \
--data_path_test /media/ssd/data_temp/CC/SAGITTAL/d.2.0.sc/test \
--weight_path /media/NAS01/jiahao/DCCNN/CC \
--num_epoch 51 \
--batch_size 1 \
--lr 0.001 \
--l2 1e-6 \
--undersampling_mask ${mask_name} \
--resolution 256 \
>> ${log_name} &
