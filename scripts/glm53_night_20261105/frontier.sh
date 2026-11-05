#!/usr/bin/env bash
# Quality/speed frontier of model-changing routing options. Each arm: SCORE (3 texts, 431 continuation
# tokens) and a 40-token decode. Baseline arm first. Output: one line per arm in frontier.txt.
S=/c/Users/yuval/AppData/Local/Temp/claude/d356b13b3d215811/21c8f351-eb5d-4e4d-b6ad-6362add8e9f9/scratchpad
cd /c/colibri-review/src/c
BASE="COLI_CUDA_MLA_PREFILL=1 CUDA_DENSE_MATCH=self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,self_attn.q_a_proj,self_attn.kv_a_proj COLI_CUDA_GEMM_MIN=32 COLI_CUDA=1 COLI_GPU=0 COLI_PFGPU=4 DIRECT=1 SNAP=C:\\AI-Models\\GLM-5.3-colibri-x4g64-r8 MTP=0 RAM_GB=112 OMP_NUM_THREADS=14 $EXTRA_BASE"
P="Write a short numbered list of five common web vulnerability classes, one line each."
: > $S/frontier.txt
for arm in "BASELINE=1" "TOPK=6" "TOPK=4" "TOPP=0.9" "CACHE_ROUTE=1 ROUTE_J=4 ROUTE_M=12" "CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=16"; do
  sc=$(env $BASE $arm SCORE=$S/score_req2.txt python $S/with_gpu_lock.py ./colibri.exe 2>/dev/null | awk '{printf "%s/%s ", $1, $2}')
  env $BASE $arm COLI_TEMP=0 NGEN=40 PROMPT="$P" python $S/with_gpu_lock.py ./colibri.exe > $S/frontier_dec.log 2>&1
  dec=$(grep -E '^prefill [0-9]+ tokens' $S/frontier_dec.log | sed -E 's/.*decode [0-9]+ tokens in [0-9.]+s \(([0-9.]+) tok\/s\).*hit rate ([0-9.]+)%.*/\1 tok\/s hit \2%/')
  agree=$(grep -o 'agree[^|]*' $S/frontier_dec.log | head -1)
  echo "$arm | LL $sc | $dec | $agree" | tee -a $S/frontier.txt
done
