export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=${PYTHONPATH}:. 

cd multilingual-t5 

PRE_TRAINED_MODEL_DIR='/home/katsumata/work/summarization/mt5/models' 
OPERATIVE_CONFIG=$PRE_TRAINED_MODEL_DIR'/operative_config.gin' 
FINE_TUNED_MODEL_DIR='/home/katsumata/work/summarization/mt5/models' 
FINE_TUNING_BATCH_SIZE=2048
PRE_TRAINGING_STEPS=1000000 
FINE_TUNING_STEPS=`expr $PRE_TRAINGING_STEPS + 10000` 
INPUT_SEQ_LEN=512 
TARGET_SEQ_LEN=64 
SAVE_STEP=1000

echo "OPERATIVE_CONFIG=$OPERATIVE_CONFIG" 
echo "FINE_TUNED_MODEL_DIR=$FINE_TUNED_MODEL_DIR" 
echo "FINE_TUNING_BATCH_SIZE=$FINE_TUNING_BATCH_SIZE"
echo "PRE_TRAINGING_STEPS=$PRE_TRAINGING_STEPS" 
echo "FINE_TUNING_STEPS=$FINE_TUNING_STEPS" 
echo "INPUT_SEQ_LEN=$INPUT_SEQ_LEN" 
echo "TARGET_SEQ_LEN=$TARGET_SEQ_LEN" 

t5_mesh_transformer \
  --model_dir="$FINE_TUNED_MODEL_DIR" \
  --module_import="t5_wikihow" \
  --gin_file="dataset.gin" \
  --gin_file="$OPERATIVE_CONFIG" \
  --gin_param="run.layout_rules=''" \
  --gin_param="run.mesh_shape='model:1,batch:1'" \
  --gin_param="run.mesh_devices = ['gpu:0']" \
  --gin_param="utils.get_variable_dtype.activation_dtype='float32'" \
  --gin_param="MIXTURE_NAME = 't5_wikihow'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps=$FINE_TUNING_STEPS" \
  --gin_param="run.sequence_length = {'inputs': $INPUT_SEQ_LEN, 'targets': $TARGET_SEQ_LEN}" \
  --gin_param="run.save_checkpoints_steps=$SAVE_STEP" \
  --gin_param="run.batch_size=('tokens_per_batch', $FINE_TUNING_BATCH_SIZE)"
