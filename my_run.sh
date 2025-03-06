DATASET_NAME="leaf_0"
DATASET_NUM="0"

DATA_PATH="./sim/${DATASET_NAME}/${DATASET_NUM}/datasets/"
MODEL_PATH="./sim/${DATASET_NAME}/${DATASET_NUM}/models/"
ROLLOUT_PATH="./sim/${DATASET_NAME}/${DATASET_NUM}/rollouts/"


python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=100 --mode='train'

python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="model-100.pt" --output_path=${ROLLOUT_PATH} --mode='rollout'

python -m gns.render_rollout --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex0"