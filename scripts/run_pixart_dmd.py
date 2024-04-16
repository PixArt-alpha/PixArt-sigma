import os
import sys
import argparse
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config")
parser.add_argument('--work_dir', help='the dir to save logs and models')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666', help='')
parser.add_argument('--rank', type=str, default='0', help='')
parser.add_argument('--world_size', type=str, default='1', help='')
parser.add_argument("--is_debugging", action="store_true")
parser.add_argument("--max_samples", type=int, default=500000, help='')
parser.add_argument("--regression_weight", type=float, default=0.25, help='regression loss weight')
parser.add_argument("--resume_from", type=str, default='', help='path of resumed checkpoint')
parser.add_argument("--mixed_precision", type=str, default='no', help='whether use mixed precision')
parser.add_argument("--batch_size", type=int, default=1, help='batch size per gpu')
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help='gradient_accumulation_steps')
parser.add_argument("--use_dm", type=int, default=1, help='use distribution matching loss')
parser.add_argument("--use_regression", type=int, default=1, help='use regression loss')
parser.add_argument("--one_step_maxt", type=int, default=999, help='maximum timestep of one step generator')
parser.add_argument("--learning_rate", type=float, default=1e-5, help='learning rate')
parser.add_argument("--lr_fake_multiplier", type=float, default=1.0, help='lr of fake model / lr of set lr')
parser.add_argument("--max_grad_norm", type=int, default=10, help='batch size per gpu')
parser.add_argument("--save_image_interval", type=int, default=500, help='iteration interval to save image')
parser.add_argument("--checkpointing_steps", type=int, default=500,
    help=(
        "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        " training using `--resume_from_checkpoint`."
    ),
)


args, args_run = parser.parse_known_args()

if args.world_size != '1':
    os.environ['MASTER_PORT'] = args.init_method.split(':')[-1]
    os.environ['MASTER_ADDR'] = args.init_method.split(':')[1][2:]
    os.environ['WORLD_SIZE'] = args.world_size
    os.environ['RANK'] = args.rank
    for k in ['MASTER_PORT', 'MASTER_ADDR', 'WORLD_SIZE', 'RANK']:
        print (k, ':', os.environ[k])

is_debugging = args.is_debugging

use_dm = True if args.use_dm == 1 else False
use_regression = True if args.use_regression == 1 else False
laion_subset = 'subset6.25'

if len(args.resume_from) == 0:
    resume_from = None
else:
    resume_from = args.resume_from

# activate enviroment
print('setting up env')
print('env set')

if args.mixed_precision == 'no':
    acc_common_args = '--use_fsdp --fsdp_offload_params="False" --fsdp_sharding_strategy=2'
else:
    acc_common_args = '--mixed_precision="fp16" --use_fsdp --fsdp_offload_params="False" --fsdp_sharding_strategy=2'

main_args = (
            f'--config="{args.config}" '
            f'--train_batch_size={args.batch_size} '
            f'--one_step_maxt={args.one_step_maxt} ' 
            f'--output_dir={args.work_dir} '
            f'--learning_rate={args.learning_rate} '
            f'--max_samples={args.max_samples} '
            f'--node_id={int(args.rank)} '
            f'--gradient_accumulation_steps={args.gradient_accumulation_steps} '
            f'--checkpointing_steps={args.checkpointing_steps} '
            f'--lr_fake_multiplier={args.lr_fake_multiplier} ' 
            f'--max_grad_norm={args.max_grad_norm} '
            f'--save_image_interval={args.save_image_interval} '
            '--max_train_steps=1000000 '
            '--di_steps=1 '
            '--start_ts=999 '
            '--cfg=3 '
            '--dataloader_num_workers=16 '
            '--resolution=512 '
            '--center_crop'
            '--random_flip '
            '--use_ema '
            '--lr_scheduler="constant" '
            '--lr_warmup_steps=0 '
            '--logging_dir="_logs" '
            '--report_to=tensorboard '
            '--adam_epsilon=1e-06 '
            '--seed=0 '
             )

if args.mixed_precision == 'fp16':
    main_args += '--mixed_precision="fp16" '

if use_dm:
    main_args += '--use_dm '
if use_regression:
    main_args += f'--use_regression --regression_weight={args.regression_weight} '

if resume_from is not None:
    main_args += f'--resume_from_checkpoint="{resume_from}" '

if is_debugging:
    num_gpus_per_node = 2
else:
    num_gpus_per_node = 8

if args.world_size != '1':
    num_processes = int(args.world_size) * num_gpus_per_node
    print('num_processes', num_processes)
    run_cmd = (f'accelerate launch {acc_common_args} '
               f'--num_machines={args.world_size} '
               f'--num_processes={num_processes} '
               f'--machine_rank={os.environ["RANK"]} '
               f'--main_process_ip={os.environ["MASTER_ADDR"]} '
               f'--main_process_port={os.environ["MASTER_PORT"]} '
               f'train_scripts/train_pixart_dmd.py {main_args}'
               )
else:
    run_cmd = (f'accelerate launch {acc_common_args} '
               f'--num_machines={args.world_size} '
               f'--num_processes={num_gpus_per_node} '
               'train_scripts/train_pixart_dmd.py '
               f'{main_args}'
               )

print('run_cmd', run_cmd)

print('running')
os.system(run_cmd)
print('done')
