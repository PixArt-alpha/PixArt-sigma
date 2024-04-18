config=configs/pixart_app_config/PixArt-DMD_xl2_img512_internalms.py
work_dir=output/debug/

# machine
machine_num=1
np=8

# training settings
max_samples=500000
batchsize=1
mix_precision='no'
use_dm=1
use_regression=1
regression_weight=0.25
one_step_maxt=400
learning_rate=1e-6
lr_fake_multiplier=1.0
max_grad_norm=10
gradient_accumulation_steps=2
save_image_interval=5000

# resume from checkpoint
resume_from=""

# train
python_command="python scripts/run_pixart_dmd.py --world_size 1 ${config} "
python_command+="--is_debugging "
python_command+="--work_dir=${work_dir} --machine_num=${machine_num} --np=${np} "
python_command+="--max_samples=${max_samples} "
python_command+="--batch_size=${batchsize} --mixed_precision=${mix_precision} "
python_command+="--use_dm=${use_dm} "
python_command+="--use_regression=${use_regression} --regression_weight=${regression_weight} "
python_command+="--one_step_maxt=${one_step_maxt} "
python_command+="--learning_rate=${learning_rate} "
python_command+="--lr_fake_multiplier=${lr_fake_multiplier} "
python_command+="--max_grad_norm=${max_grad_norm} "
python_command+="--gradient_accumulation_steps=${gradient_accumulation_steps} "
python_command+="--save_image_interval=${save_image_interval} "

python_command+="--resume_from=${resume_from} "

eval $python_command