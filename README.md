# FauxPilot

This is an attempt to build a locally hosted version of [GitHub Copilot](https://copilot.github.com/). It uses the [SalesForce CodeGen](https://github.com/salesforce/CodeGen) models inside of NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) with the [FasterTransformer backend](https://github.com/triton-inference-server/fastertransformer_backend/).

## Prerequisites

You'll need:

* Docker
* `docker compose` >= 1.28
* An NVIDIA GPU with Compute Capability >= 7.0 and enough VRAM to run the model you want.
* [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
* `curl` and `zstd` for downloading and unpacking the models.

Note that the VRAM requirements listed by `setup.sh` are *total* -- if you have multiple GPUs, you can split the model across them. So, if you have two NVIDIA RTX 3080 GPUs, you *should* be able to run the 6B model by putting half on each GPU.

## Support and Warranty

lmao

## Setup

Run the setup script to choose a model to use. This will download the model from Huggingface and then convert it for use with FasterTransformer.

```
$ ./setup.sh 
Are you loading pre-trained model? <y/N> N
Fine-tuned models available:
[9] fine-tuned-codegen-2B (7GB total VRAM required; Verilog-only)
[10] fine-tuned-codegen-6B (13GB total VRAM required; Verilog-only)
[11] fine-tuned-codegen-16B (32GB total VRAM required; Verilog-only)
Models available:
Enter your choice [6]: 9
Enter number of GPUs [1]: 1
Where do you want to save the final model [/home/st4920/fauxpilot_changes/models]? 
Downloading the model from HuggingFace, this will take a while...
/home/st4920/fauxpilot_changes/converter/models/fine-tuned-codegen-2B-1gpu
downloading fine-tuned model
https://huggingface.co/shailja/fine-tuned-codegen-2B-Verilog/resolve/main/fine-tuned-codegen-2B-1gpu.tar.zst
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1179  100  1179    0     0  18607      0 --:--:-- --:--:-- --:--:-- 18714
100 10.3G  100 10.3G    0     0  39.9M      0  0:04:24  0:04:24 --:--:-- 41.8M
Done! Now run ./launch.sh to start the FauxPilot server.
```

<!--
Models available:
[1] codegen-350M-mono (2GB total VRAM required; Python-only)
[2] codegen-350M-multi (2GB total VRAM required; multi-language)
[3] codegen-2B-mono (7GB total VRAM required; Python-only)
[4] codegen-2B-multi (7GB total VRAM required; multi-language)
[5] codegen-6B-mono (13GB total VRAM required; Python-only)
[6] codegen-6B-multi (13GB total VRAM required; multi-language)
[7] codegen-16B-mono (32GB total VRAM required; Python-only)
[8] codegen-16B-multi (32GB total VRAM required; multi-language)
[9] custom fine-tuned codegen model
Enter your choice [6]: 9
Enter number of GPUs [1]: 4
Enter the path to the fine-tuned codegen model: /home/codegen-verilog-16B-1-epochs/checkpoint-2000
Where do you want to save the final model [/home/st4920/fauxpilot-codegen16B/models]? 
Downloading and converting the model, this will take a while...
Converting model fine-tuned-codegen with 4 GPUs
cp: cannot stat 'models/fine-tuned-codegen-4gpu': No such file or directory
converting the fine-tuned model to GPTJ
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]Moving 0 files to the new cache system
Loading CodeGen model
torch.float16 torch.float16
Creating empty GPTJ model
Converting...
Conversion complete.
Saving model to fine-tuned-codegen-hf...
Downloading: 100%|██████████| 240/240 [00:00<00:00, 167kB/s]
Downloading: 100%|██████████| 798k/798k [00:01<00:00, 669kB/s] 
Downloading: 100%|██████████| 456k/456k [00:00<00:00, 9.34MB/s]
Downloading: 100%|██████████| 2.11M/2.11M [00:00<00:00, 2.99MB/s]
Downloading: 100%|██████████| 1.00k/1.00k [00:00<00:00, 791kB/s]
Downloading: 100%|██████████| 90.0/90.0 [00:00<00:00, 51.9kB/s]
==========================================================
Created config file for fine-tuned-codegen
  Config:  /models/fine-tuned-codegen-4gpu/fastertransformer/config.pbtxt
  Weights: /models/fine-tuned-codegen-4gpu/fastertransformer/1/4-gpu
==========================================================
==========================================================

=============== Argument ===============
saved_dir: /models/fine-tuned-codegen-4gpu/fastertransformer/1
in_file: fine-tuned-codegen-hf
trained_gpu_num: 1
infer_gpu_num: 4
processes: 4
weight_data_type: fp32
========================================
transformer.wte.weight
transformer.h.0.ln_1.weight
[... more conversion output trimmed ...]
transformer.ln_f.weight
transformer.ln_f.bias
lm_head.weight
lm_head.bias
Done! Now run ./launch.sh to start the FauxPilot server.
-->
<!-- 
Once the `setup.sh` finishes successfully, change the value for `model_checkpoint_path` to `/model/fastertransformer/1/n-gpu` in `config.pbtxt` located inside  `fauxpilot_changes/models/fine-tuned-codegen-mB-ngpu/fastertransformer/` 
[Please note that **n** and **m** in the path are the # gpu and # model parameters (2/6/16) selected during setup]
-->

Once the `setup.sh` finishes successfully, then you can just run `./launch.sh`:

```
$ ./launch.sh 
[+] Running 2/0
 ⠿ Container fauxpilot-codegen16b-triton-1         Created                                                                                         0.0s
 ⠿ Container fauxpilot-codegen16b-copilot_proxy-1  Created                                                                                         0.0s
Attaching to fauxpilot-codegen16b-copilot_proxy-1, fauxpilot-codegen16b-triton-1
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | =============================
fauxpilot-codegen16b-triton-1         | == Triton Inference Server ==
fauxpilot-codegen16b-triton-1         | =============================
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | NVIDIA Release 22.06 (build 39726160)
fauxpilot-codegen16b-triton-1         | Triton Server Version 2.23.0
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
fauxpilot-codegen16b-triton-1         | By pulling and using the container, you accept the terms and conditions of this license:
fauxpilot-codegen16b-triton-1         | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
fauxpilot-codegen16b-copilot_proxy-1  |  * Debug mode: off
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-copilot_proxy-1  | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
fauxpilot-codegen16b-copilot_proxy-1  |  * Running on all addresses (0.0.0.0)
fauxpilot-codegen16b-copilot_proxy-1  |  * Running on http://127.0.0.1:5000
fauxpilot-codegen16b-copilot_proxy-1  |  * Running on http://172.22.0.2:5000
fauxpilot-codegen16b-copilot_proxy-1  | Press CTRL+C to quit
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.602241 89 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7fdf00000000' with size 268435456
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.604455 89 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0 with size 67108864
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.604467 89 cuda_memory_manager.cc:105] CUDA memory pool is created on device 1 with size 67108864
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.604471 89 cuda_memory_manager.cc:105] CUDA memory pool is created on device 2 with size 67108864
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.604474 89 cuda_memory_manager.cc:105] CUDA memory pool is created on device 3 with size 67108864
fauxpilot-codegen16b-triton-1         | I1116 03:15:17.938577 89 model_repository_manager.cc:1191] loading: fastertransformer:1
fauxpilot-codegen16b-triton-1         | I1116 03:15:18.236475 89 libfastertransformer.cc:1226] TRITONBACKEND_Initialize: fastertransformer
fauxpilot-codegen16b-triton-1         | I1116 03:15:18.236514 89 libfastertransformer.cc:1236] Triton TRITONBACKEND API version: 1.10
fauxpilot-codegen16b-triton-1         | I1116 03:15:18.236520 89 libfastertransformer.cc:1242] 'fastertransformer' TRITONBACKEND API version: 1.10
fauxpilot-codegen16b-triton-1         | I1116 03:15:18.236585 89 libfastertransformer.cc:1274] TRITONBACKEND_ModelInitialize: fastertransformer (version 1)
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.238548 89 libfastertransformer.cc:149] model configuration:
fauxpilot-codegen16b-triton-1         | {
fauxpilot-codegen16b-triton-1         |     "name": "fastertransformer",
fauxpilot-codegen16b-triton-1         |     "platform": "",
fauxpilot-codegen16b-triton-1         |     "backend": "fastertransformer",
fauxpilot-codegen16b-triton-1         |     "version_policy": {
fauxpilot-codegen16b-triton-1         |         "latest": {
fauxpilot-codegen16b-triton-1         |             "num_versions": 1
fauxpilot-codegen16b-triton-1         |         }
fauxpilot-codegen16b-triton-1         |     },
fauxpilot-codegen16b-triton-1         |     "max_batch_size": 1024,
fauxpilot-codegen16b-triton-1         |     "input": [
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "input_ids",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "start_id",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "end_id",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "input_lengths",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "request_output_len",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "runtime_top_k",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "runtime_top_p",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "beam_search_diversity_rate",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "temperature",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "len_penalty",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "repetition_penalty",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "random_seed",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_INT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "is_return_log_probs",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_BOOL",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "beam_width",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "reshape": {
fauxpilot-codegen16b-triton-1         |                 "shape": []
fauxpilot-codegen16b-triton-1         |             },
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "bad_words_list",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_INT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 2,
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "stop_words_list",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_INT32",
fauxpilot-codegen16b-triton-1         |             "format": "FORMAT_NONE",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 2,
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false,
fauxpilot-codegen16b-triton-1         |             "allow_ragged_batch": false,
fauxpilot-codegen16b-triton-1         |             "optional": true
fauxpilot-codegen16b-triton-1         |         }
fauxpilot-codegen16b-triton-1         |     ],
fauxpilot-codegen16b-triton-1         |     "output": [
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "output_ids",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1,
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "label_filename": "",
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "sequence_length",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_UINT32",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "label_filename": "",
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "cum_log_probs",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "label_filename": "",
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "output_log_probs",
fauxpilot-codegen16b-triton-1         |             "data_type": "TYPE_FP32",
fauxpilot-codegen16b-triton-1         |             "dims": [
fauxpilot-codegen16b-triton-1         |                 -1,
fauxpilot-codegen16b-triton-1         |                 -1
fauxpilot-codegen16b-triton-1         |             ],
fauxpilot-codegen16b-triton-1         |             "label_filename": "",
fauxpilot-codegen16b-triton-1         |             "is_shape_tensor": false
fauxpilot-codegen16b-triton-1         |         }
fauxpilot-codegen16b-triton-1         |     ],
fauxpilot-codegen16b-triton-1         |     "batch_input": [],
fauxpilot-codegen16b-triton-1         |     "batch_output": [],
fauxpilot-codegen16b-triton-1         |     "optimization": {
fauxpilot-codegen16b-triton-1         |         "priority": "PRIORITY_DEFAULT",
fauxpilot-codegen16b-triton-1         |         "input_pinned_memory": {
fauxpilot-codegen16b-triton-1         |             "enable": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "output_pinned_memory": {
fauxpilot-codegen16b-triton-1         |             "enable": true
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "gather_kernel_buffer_threshold": 0,
fauxpilot-codegen16b-triton-1         |         "eager_batching": false
fauxpilot-codegen16b-triton-1         |     },
fauxpilot-codegen16b-triton-1         |     "instance_group": [
fauxpilot-codegen16b-triton-1         |         {
fauxpilot-codegen16b-triton-1         |             "name": "fastertransformer_0",
fauxpilot-codegen16b-triton-1         |             "kind": "KIND_CPU",
fauxpilot-codegen16b-triton-1         |             "count": 1,
fauxpilot-codegen16b-triton-1         |             "gpus": [],
fauxpilot-codegen16b-triton-1         |             "secondary_devices": [],
fauxpilot-codegen16b-triton-1         |             "profile": [],
fauxpilot-codegen16b-triton-1         |             "passive": false,
fauxpilot-codegen16b-triton-1         |             "host_policy": ""
fauxpilot-codegen16b-triton-1         |         }
fauxpilot-codegen16b-triton-1         |     ],
fauxpilot-codegen16b-triton-1         |     "default_model_filename": "fine-tuned-codegen",
fauxpilot-codegen16b-triton-1         |     "cc_model_filenames": {},
fauxpilot-codegen16b-triton-1         |     "metric_tags": {},
fauxpilot-codegen16b-triton-1         |     "parameters": {
fauxpilot-codegen16b-triton-1         |         "tensor_para_size": {
fauxpilot-codegen16b-triton-1         |             "string_value": "4"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "decoder_layers": {
fauxpilot-codegen16b-triton-1         |             "string_value": "34"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "size_per_head": {
fauxpilot-codegen16b-triton-1         |             "string_value": "256"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "max_seq_len": {
fauxpilot-codegen16b-triton-1         |             "string_value": "2048"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "end_id": {
fauxpilot-codegen16b-triton-1         |             "string_value": "50256"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "inter_size": {
fauxpilot-codegen16b-triton-1         |             "string_value": "24576"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "head_num": {
fauxpilot-codegen16b-triton-1         |             "string_value": "24"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "model_type": {
fauxpilot-codegen16b-triton-1         |             "string_value": "GPT-J"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "model_checkpoint_path": {
fauxpilot-codegen16b-triton-1         |             "string_value": "/model/fastertransformer/1/4-gpu"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "rotary_embedding": {
fauxpilot-codegen16b-triton-1         |             "string_value": "64"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "pipeline_para_size": {
fauxpilot-codegen16b-triton-1         |             "string_value": "1"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "start_id": {
fauxpilot-codegen16b-triton-1         |             "string_value": "50256"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "model_name": {
fauxpilot-codegen16b-triton-1         |             "string_value": "fine-tuned-codegen"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "is_half": {
fauxpilot-codegen16b-triton-1         |             "string_value": "1"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "enable_custom_all_reduce": {
fauxpilot-codegen16b-triton-1         |             "string_value": "0"
fauxpilot-codegen16b-triton-1         |         },
fauxpilot-codegen16b-triton-1         |         "vocab_size": {
fauxpilot-codegen16b-triton-1         |             "string_value": "50257"
fauxpilot-codegen16b-triton-1         |         }
fauxpilot-codegen16b-triton-1         |     },
fauxpilot-codegen16b-triton-1         |     "model_warmup": []
fauxpilot-codegen16b-triton-1         | }
fauxpilot-codegen16b-triton-1         | I1116 03:15:18.239778 89 libfastertransformer.cc:1320] TRITONBACKEND_ModelInstanceInitialize: fastertransformer_0 (device 0)
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239804 89 libfastertransformer.cc:453] Faster transformer model instance is created at GPU '0'
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239808 89 libfastertransformer.cc:459] Model name fine-tuned-codegen
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239821 89 libfastertransformer.cc:578] Get input name: input_ids, type: TYPE_UINT32, shape: [-1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239827 89 libfastertransformer.cc:578] Get input name: start_id, type: TYPE_UINT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239832 89 libfastertransformer.cc:578] Get input name: end_id, type: TYPE_UINT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239836 89 libfastertransformer.cc:578] Get input name: input_lengths, type: TYPE_UINT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239841 89 libfastertransformer.cc:578] Get input name: request_output_len, type: TYPE_UINT32, shape: [-1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239846 89 libfastertransformer.cc:578] Get input name: runtime_top_k, type: TYPE_UINT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239851 89 libfastertransformer.cc:578] Get input name: runtime_top_p, type: TYPE_FP32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239855 89 libfastertransformer.cc:578] Get input name: beam_search_diversity_rate, type: TYPE_FP32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239860 89 libfastertransformer.cc:578] Get input name: temperature, type: TYPE_FP32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239864 89 libfastertransformer.cc:578] Get input name: len_penalty, type: TYPE_FP32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239869 89 libfastertransformer.cc:578] Get input name: repetition_penalty, type: TYPE_FP32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239873 89 libfastertransformer.cc:578] Get input name: random_seed, type: TYPE_INT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239878 89 libfastertransformer.cc:578] Get input name: is_return_log_probs, type: TYPE_BOOL, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239883 89 libfastertransformer.cc:578] Get input name: beam_width, type: TYPE_UINT32, shape: [1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239888 89 libfastertransformer.cc:578] Get input name: bad_words_list, type: TYPE_INT32, shape: [2, -1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239893 89 libfastertransformer.cc:578] Get input name: stop_words_list, type: TYPE_INT32, shape: [2, -1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239902 89 libfastertransformer.cc:620] Get output name: output_ids, type: TYPE_UINT32, shape: [-1, -1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239907 89 libfastertransformer.cc:620] Get output name: sequence_length, type: TYPE_UINT32, shape: [-1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239912 89 libfastertransformer.cc:620] Get output name: cum_log_probs, type: TYPE_FP32, shape: [-1]
fauxpilot-codegen16b-triton-1         | W1116 03:15:18.239916 89 libfastertransformer.cc:620] Get output name: output_log_probs, type: TYPE_FP32, shape: [-1, -1]
fauxpilot-codegen16b-triton-1         | [FT][WARNING] Custom All Reduce only supports 8 Ranks currently. Using NCCL as Comm.
fauxpilot-codegen16b-triton-1         | I1116 03:15:19.424954 89 libfastertransformer.cc:307] Before Loading Model:
fauxpilot-codegen16b-triton-1         | after allocation, free 15.38 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | after allocation, free 15.38 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | after allocation, free 15.38 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | after allocation, free 15.38 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | [WARNING] gemm_config.in is not found; using default GEMM algo
fauxpilot-codegen16b-triton-1         | [WARNING] gemm_config.in is not found; using default GEMM algo
fauxpilot-codegen16b-triton-1         | [WARNING] gemm_config.in is not found; using default GEMM algo
fauxpilot-codegen16b-triton-1         | [WARNING] gemm_config.in is not found; using default GEMM algo
fauxpilot-codegen16b-triton-1         | after allocation, free 6.67 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | I1116 03:15:51.106679 89 libfastertransformer.cc:321] After Loading Model:
fauxpilot-codegen16b-triton-1         | after allocation, free 6.67 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | after allocation, free 6.67 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | after allocation, free 6.67 GB total 15.74 GB
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.384913 89 libfastertransformer.cc:537] Model instance is created on GPU NVIDIA RTX A4000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.385533 89 model_repository_manager.cc:1345] successfully loaded 'fastertransformer' version 1
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.385816 89 server.cc:556] 
fauxpilot-codegen16b-triton-1         | +------------------+------+
fauxpilot-codegen16b-triton-1         | | Repository Agent | Path |
fauxpilot-codegen16b-triton-1         | +------------------+------+
fauxpilot-codegen16b-triton-1         | +------------------+------+
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.386009 89 server.cc:583] 
fauxpilot-codegen16b-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | | Backend           | Path                                                                        | Config                                                                                                                                                         |
fauxpilot-codegen16b-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | | fastertransformer | /opt/tritonserver/backends/fastertransformer/libtriton_fastertransformer.so | {"cmdline":{"auto-complete-config":"false","min-compute-capability":"6.000000","backend-directory":"/opt/tritonserver/backends","default-max-batch-size":"4"}} |
fauxpilot-codegen16b-triton-1         | +-------------------+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.386085 89 server.cc:626] 
fauxpilot-codegen16b-triton-1         | +-------------------+---------+--------+
fauxpilot-codegen16b-triton-1         | | Model             | Version | Status |
fauxpilot-codegen16b-triton-1         | +-------------------+---------+--------+
fauxpilot-codegen16b-triton-1         | | fastertransformer | 1       | READY  |
fauxpilot-codegen16b-triton-1         | +-------------------+---------+--------+
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.432164 89 metrics.cc:650] Collecting metrics for GPU 0: NVIDIA RTX A4000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.432200 89 metrics.cc:650] Collecting metrics for GPU 1: NVIDIA RTX A4000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.432208 89 metrics.cc:650] Collecting metrics for GPU 2: NVIDIA RTX A4000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.432215 89 metrics.cc:650] Collecting metrics for GPU 3: NVIDIA RTX A4000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.432881 89 tritonserver.cc:2159] 
fauxpilot-codegen16b-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | | Option                           | Value                                                                                                                                                                                        |
fauxpilot-codegen16b-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | | server_id                        | triton                                                                                                                                                                                       |
fauxpilot-codegen16b-triton-1         | | server_version                   | 2.23.0                                                                                                                                                                                       |
fauxpilot-codegen16b-triton-1         | | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics trace |
fauxpilot-codegen16b-triton-1         | | model_repository_path[0]         | /model                                                                                                                                                                                       |
fauxpilot-codegen16b-triton-1         | | model_control_mode               | MODE_NONE                                                                                                                                                                                    |
fauxpilot-codegen16b-triton-1         | | strict_model_config              | 1                                                                                                                                                                                            |
fauxpilot-codegen16b-triton-1         | | rate_limit                       | OFF                                                                                                                                                                                          |
fauxpilot-codegen16b-triton-1         | | pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                    |
fauxpilot-codegen16b-triton-1         | | cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                                                                     |
fauxpilot-codegen16b-triton-1         | | cuda_memory_pool_byte_size{1}    | 67108864                                                                                                                                                                                     |
fauxpilot-codegen16b-triton-1         | | cuda_memory_pool_byte_size{2}    | 67108864                                                                                                                                                                                     |
fauxpilot-codegen16b-triton-1         | | cuda_memory_pool_byte_size{3}    | 67108864                                                                                                                                                                                     |
fauxpilot-codegen16b-triton-1         | | response_cache_byte_size         | 0                                                                                                                                                                                            |
fauxpilot-codegen16b-triton-1         | | min_supported_compute_capability | 6.0                                                                                                                                                                                          |
fauxpilot-codegen16b-triton-1         | | strict_readiness                 | 1                                                                                                                                                                                            |
fauxpilot-codegen16b-triton-1         | | exit_timeout                     | 30                                                                                                                                                                                           |
fauxpilot-codegen16b-triton-1         | +----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
fauxpilot-codegen16b-triton-1         | 
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.435600 89 grpc_server.cc:4587] Started GRPCInferenceService at 0.0.0.0:8001
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.435954 89 http_server.cc:3303] Started HTTPService at 0.0.0.0:8000
fauxpilot-codegen16b-triton-1         | I1116 03:15:53.477616 89 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## API

Once everything is up and running, you should have a server listening for requests on `http://localhost:5000`. You can now talk to it using the standard [OpenAI API](https://beta.openai.com/docs/api-reference/) (although the full API isn't implemented yet). For example, from Python, using the [OpenAI Python bindings](https://github.com/openai/openai-python):

```python
$ ipython
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
Type 'copyright', 'credits' or 'license' for more information
IPython 8.2.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import openai

In [2]: openai.api_key = 'dummy'

In [3]: openai.api_base = 'http://127.0.0.1:5000/v1'

In [4]: result = openai.Completion.create(engine='codegen', prompt='\\module half adder', max_tokens=100, temperature=0.1, n=3,top_p=1.0, stop=["endmodule"])

In [5]: result
Out[5]: 
<OpenAIObject text_completion id=cmpl-6hqu8Rcaq25078IHNJNVooU4xLY6w at 0x7f602c3d2f40> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " (input a, b, cin, output sum, cout);\n\tassign sum = a ^ b ^ cin;\n\tassign cout = (a & b) | (a & cin) | (b & cin);\n"
    },
    {
      "finish_reason": "stop",
      "index": 1,
      "logprobs": null,
      "text": " (input a, b, output sum, carry);\n\tassign sum = a ^ b;\n\tassign carry = a & b;\n"
    },
    {
      "finish_reason": "stop",
      "index": 2,
      "logprobs": null,
      "text": " (input a, b, output sum, carry);\n\tassign sum = a ^ b;\n\tassign carry = a & b;\n"
    }
  ],
  "created": 1668569699,
  "id": "cmpl-kIY0lzUWkQsFve2J2w9oBSO7gGYCh",
  "model": "codegen",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 123,
    "prompt_tokens": 5,
    "total_tokens": 128
  }
}
```

## Copilot Plugin

Perhaps more excitingly, you can configure the official [VSCode Copilot plugin](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) to use your local server. Just edit your `settings.json` to add:

```json
    "github.copilot.advanced": {
        "debug.overrideEngine": "codegen",
        "debug.testOverrideProxyUrl": "http://localhost:5000",
        "debug.overrideProxyUrl": "http://localhost:5000"
    }
```

And you should be able to use Copilot with your own locally hosted suggestions! Of course, probably a lot of stuff is subtly broken. In particular, the probabilities returned by the server are partly fake. Fixing this would require changing FasterTransformer so that it can return log-probabilities for the top k tokens rather that just the chosen token.

Have fun!
