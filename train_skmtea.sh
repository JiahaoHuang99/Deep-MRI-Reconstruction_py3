# Setting
# Train on AYL
GPUID=2

model_name="D5C5"
dataset_name="SKMTEA"

#mask_name="fMRI_Ran_AF4_CF0.08_PE512"
mask_name="fMRI_Ran_AF8_CF0.04_PE512"
#mask_name="fMRI_Ran_AF16_CF0.02_PE512"
#mask_name="fMRI_Reg_AF4_CF0.08_PE512"
#mask_name="fMRI_Reg_AF8_CF0.04_PE512"
#mask_name="fMRI_Reg_AF16_CF0.02_PE512"

task_name=${model_name}_${dataset_name}_${mask_name}
log_name=log_train_${task_name}

# Run
rm ${log_name}

CUDA_VISIBLE_DEVICES=$GPUID \
PYTHONPATH=$(pwd) \
nohup python train_DCCNN_D5C5_Complex.py \
--task_name ${task_name} \
--data_path_train /media/ssd/data_temp/SKM-TEA/d.0.1/train/h5_image_complex \
--data_path_val /media/ssd/data_temp/SKM-TEA/d.0.1/val/h5_image_complex \
--data_path_test /media/ssd/data_temp/SKM-TEA/d.0.1/test/h5_image_complex \
--num_epoch 51 \
--batch_size 4 \
--lr 0.001 \
--l2 1e-6 \
--undersampling_mask ${mask_name} \
--resolution 512 \
>> ${log_name} &
