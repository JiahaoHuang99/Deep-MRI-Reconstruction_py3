# Setting
GPUID=3

model_name="D5C5"
dataset_name="FastMRI"

mask_name="fMRI_Ran_AF4_CF0.08_PE320"
#mask_name="fMRI_Ran_AF8_CF0.04_PE320"
#mask_name="fMRI_Ran_AF16_CF0.02_PE320"
#mask_name="fMRI_Reg_AF4_CF0.08_PE320"
#mask_name="fMRI_Reg_AF8_CF0.04_PE320"
#mask_name="fMRI_Reg_AF16_CF0.02_PE320"

task_name=${model_name}_${dataset_name}_${mask_name}
log_name=log_test_${task_name}

# Run
rm ${log_name}

CUDA_VISIBLE_DEVICES=$GPUID \
PYTHONPATH=$(pwd) \
nohup python test_DCCNN_D5C5_Complex.py \
--task_name ${task_name} \
--data_path_train /media/ssd/data_temp/fastMRI/knee/d.1.0.complex/train/PD/h5_image_complex \
--data_path_val /media/ssd/data_temp/fastMRI/knee/d.1.0.complex/val/PD/h5_image_complex \
--data_path_test /media/ssd/data_temp/fastMRI/knee/d.1.0.complex/val/PD/h5_image_complex \
--weight_path /media/NAS01/jiahao/DCCNN/FastMRI \
--num_epoch 100 \
--batch_size 10 \
--lr 0.001 \
--l2 1e-6 \
--undersampling_mask ${mask_name} \
--resolution 320 \
>> ${log_name} &
