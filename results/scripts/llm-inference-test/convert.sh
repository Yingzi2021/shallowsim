HF_MODEL_DIR="/app/tensorrt_llm/user_ws/model_weight/Qwen1.5-MoE-A2.7B"
WEIGHT_DIR="/app/tensorrt_llm/user_ws/qwen1.5-moe_weight"
ENGINE_DIR="/app/tensorrt_llm/user_ws/qwen1.5-moe_engine-pp4-tp1"

rm -rf ${WEIGHT_DIR}
rm -rf ${ENGINE_DIR}

mkdir -p ${WEIGHT_DIR}
mkdir -p ${ENGINE_DIR}

# convert the checkpoint to tensorrt-llm format
python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
       --model_dir ${HF_MODEL_DIR} \
       --output_dir ${WEIGHT_DIR} \
       --dtype float16 \
       --tp_size 1 \
       --pp_size 4 \
       # expert parallelism: --moe_ep_size(default = 1)

# build the engine
trtllm-build --checkpoint_dir ${WEIGHT_DIR} \
--output_dir ${ENGINE_DIR} \
--max_input_len 1024 \
--max_batch_size 256

# run the server
trtllm-serve serve \
  ${ENGINE_DIR} \
  --tp_size 1 \
  --pp_size 4 \
  --tokenizer ${HF_MODEL_DIR} \
  --host 0.0.0.0 --port 8000