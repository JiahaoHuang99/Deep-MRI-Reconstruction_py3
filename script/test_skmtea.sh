# Setting
GPUID=2

project_path="/home/jh/Deep-MRI-Reconstruction_py3"
cd ${project_path}

model_name="D5C5"
dataset_name="SKMTEA"

#mask_name="fMRI_Ran_AF4_CF0.08_PE512"
#mask_name="fMRI_Ran_AF8_CF0.04_PE512"
#mask_name="fMRI_Ran_AF16_CF0.02_PE512"
#mask_name="fMRI_Reg_AF4_CF0.08_PE512"
#mask_name="fMRI_Reg_AF8_CF0.04_PE512"
mask_name="fMRI_Reg_AF16_CF0.02_PE512"


task_name=${model_name}_${dataset_name}_${mask_name}
log_name=log_test_${task_name}

# Run
rm ${log_name}

CUDA_VISIBLE_DEVICES=$GPUID \
PYTHONPATH=$(pwd) \
nohup python test_DCCNN_D5C5_SKMTEA.py \
--task_name ${task_name} \
--data_path_train /media/ssd/data_temp/SKM-TEA/d.0.2/train/h5_image_complex \
--data_path_val /media/ssd/data_temp/SKM-TEA/d.0.2/val/h5_image_complex \
--data_path_test /media/ssd/data_temp/SKM-TEA/d.0.2/test/h5_image_complex \
--weight_path /media/NAS01/jiahao/DCCNN/SKMTEA \
--num_epoch 21 \
--batch_size 1 \
--lr 0.001 \
--l2 1e-6 \
--undersampling_mask ${mask_name} \
--resolution 512 \
>> ${log_name} &
