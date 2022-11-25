#!/bin/bash

if [ -f config.env ]; then
    echo "config.env already exists, skipping"
    echo "Please delete config.env if you want to re-run this script"
    exit 0
fi

read -p "Are you loading pre-trained model? <y/N> " input
if [[ $input == "y" || $input == "Y" || $input == "yes" || $input == "Yes" ]]; then


    echo "Pre-trained models available:"
    echo "[1] codegen-350M-mono (2GB total VRAM required; Python-only)"
    echo "[2] codegen-350M-multi (2GB total VRAM required; multi-language)"
    echo "[3] codegen-2B-mono (7GB total VRAM required; Python-only)"
    echo "[4] codegen-2B-multi (7GB total VRAM required; multi-language)"
    echo "[5] codegen-6B-mono (13GB total VRAM required; Python-only)"
    echo "[6] codegen-6B-multi (13GB total VRAM required; multi-language)"
    echo "[7] codegen-16B-mono (32GB total VRAM required; Python-only)"
    echo "[8] codegen-16B-multi (32GB total VRAM required; multi-language)"

    # Read their choice
    read -p "Enter your choice [6]: " MODEL_NUM
else 

    echo "Fine-tuned models available:"
    echo "[9] fine-tuned-codegen-2B (7GB total VRAM required; Verilog-only)"
    echo "[10] fine-tuned-codegen-6B (13GB total VRAM required; Verilog-only)"
    echo "[11] fine-tuned-codegen-16B (32GB total VRAM required; Verilog-only)"
    echo "Models available:"

    # Read their choice
    read -p "Enter your choice [6]: " MODEL_NUM

fi

# Convert model number to model name
case $MODEL_NUM in
    1) MODEL="codegen-350M-mono" ;;
    2) MODEL="codegen-350M-multi" ;;
    3) MODEL="codegen-2B-mono" ;;
    4) MODEL="codegen-2B-multi" ;;
    5) MODEL="codegen-6B-mono" ;;
    6) MODEL="codegen-6B-multi" ;;
    7) MODEL="codegen-16B-mono" ;;
    8) MODEL="codegen-16B-multi" ;;
    9) MODEL="fine-tuned-codegen-2B" ;;
    10) MODEL="fine-tuned-codegen-6B" ;;
    11) MODEL="fine-tuned-codegen-16B" ;;
    *) MODEL="codegen-6B-multi" ;;
esac

# Read model directory
# if [ "$MODEL" == "fine-tuned-codegen-"* ]; then
#     read -p "Enter the path to the fine-tuned codegen model: " FINETUNE_DIR
# fi

# Read number of GPUs
read -p "Enter number of GPUs [1]: " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-1}


read -p "Where do you want to save the final model [$(pwd)/models]? " MODEL_DIR
if [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="$(pwd)/models"
fi

# Write config.env
echo "MODEL=${MODEL}" > config.env
echo "NUM_GPUS=${NUM_GPUS}" >> config.env
echo "MODEL_DIR=${MODEL_DIR}" >> config.env
# if [[ "$MODEL" == "fine-tuned-codegen" ]]; then
#     echo "FINETUNE_DIR=${FINETUNE_DIR}" >> config.env
# else
#     echo "FINETUNE_DIR=none" >> config.env
# fi

if [ -d "$MODEL_DIR"/"${MODEL}"-${NUM_GPUS}gpu ]; then
    echo "Converted model for ${MODEL}-${NUM_GPUS}gpu already exists, skipping"
    echo "Please delete ${MODEL_DIR}/${MODEL}-${NUM_GPUS}gpu if you want to re-convert it"
    exit 0
fi



# Create model directory
mkdir -p "${MODEL_DIR}"

# For some of the models we can download it preconverted.
if [[ $NUM_GPUS -le 2 ]]; then
    echo "Downloading the model from HuggingFace, this will take a while..."
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    DEST="${MODEL}-${NUM_GPUS}gpu"
    ARCHIVE="${MODEL_DIR}/${DEST}.tar.zst"
    cp -r "$SCRIPT_DIR"/converter/models/"$DEST" "${MODEL_DIR}"
    echo "$SCRIPT_DIR"/converter/models/"$DEST"
    if [[ "$MODEL" == "fine-tuned-codegen"* ]]; then
        echo "downloading fine-tuned model"
        echo "https://huggingface.co/shailja/${MODEL}-Verilog/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst"

        curl -L "https://huggingface.co/shailja/${MODEL}-Verilog/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst" \
            -o "$ARCHIVE"
	tar -xf "${ARCHIVE}" -C "${MODEL_DIR}"
    else
        curl -L "https://huggingface.co/moyix/${MODEL}-gptj/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst" \
            -o "$ARCHIVE"
	zstd "$ARCHIVE" | tar -xf - -C "${MODEL_DIR}"
    fi
    #zstd "$ARCHIVE" | tar -xf - -C "${MODEL_DIR}"
    #tar -xf "${ARCHIVE}" -C "${MODEL_DIR}"
    #echo "$SCRIPT_DIR"/converter/models/"$DEST"/fastertransformer/config.pbtxt 
    #echo "${MODEL_DIR}"/"$DEST"/fastertransformer/
    #cp "$SCRIPT_DIR"/converter/models/"$DEST"/fastertransformer/config.pbtxt "${MODEL_DIR}"/"$DEST"/fastertransformer/
    rm -f "$ARCHIVE"
else
    echo "Downloading and converting the model, this will take a while..."
    docker run --rm -v ${MODEL_DIR}:/models -v ${FINETUNE_DIR}:/model-checkpoint -e MODEL=${MODEL} -e NUM_GPUS=${NUM_GPUS} moyix/model_converter:latest
fi
echo "Done! Now run ./launch.sh to start the FauxPilot server."
