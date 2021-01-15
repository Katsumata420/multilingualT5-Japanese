export PYTHONPATH=${PYTHONPATH}:. 
export CUDA_VISIBLE_DEVICES=1

cd multilingual-t5 

FINE_TUNED_MODEL_DIR='/home/katsumata/work/summarization/mt5/models' 
OPERATIVE_CONFIG=$FINE_TUNED_MODEL_DIR'/operative_config.gin' 

echo "OPERATIVE_CONFIG=$OPERATIVE_CONFIG" 
echo "FINE_TUNED_MODEL_DIR=$FINE_TUNED_MODEL_DIR" 

t5_mesh_transformer \
  --model_dir="$FINE_TUNED_MODEL_DIR" \
  --module_import="t5_wikihow" \
  --gin_file="$OPERATIVE_CONFIG" \
  --gin_param="run.layout_rules=''" \
  --gin_param="run.mesh_shape=''" \
  --gin_file="eval.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="utils.get_variable_dtype.slice_dtype='float32'" \
  --gin_param="utils.get_variable_dtype.activation_dtype='float32'" \
  --gin_param="MIXTURE_NAME = 't5_wikihow'" \
  --gin_param="run.dataset_split='test'" \
  --gin_param="run.batch_size=('tokens_per_batch', 1024)" \
  --gin_param="eval_checkpoint_step = 1009000" 2>&1 | tee test.log
