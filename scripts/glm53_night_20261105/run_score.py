import os, subprocess, sys
O = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
env = dict(os.environ)
env.update({"COLI_CUDA_MLA_PREFILL": "1", "CUDA_DENSE": "0",
    "CUDA_DENSE_MATCH": "self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,self_attn.q_a_proj,self_attn.kv_a_proj",
    "COLI_CUDA_GEMM_MIN": "32", "COLI_CUDA": "1", "COLI_GPU": "0", "COLI_PFGPU": "4", "CUDA_EXPERT_GB": "0",
    "DIRECT": "1", "SNAP": r"C:\AI-Models\GLM-5.3-colibri-x4g64-r8", "MTP": "0", "RAM_GB": "112", "OMP_NUM_THREADS": "14",
    "SCORE": O + r"\score_req2.txt"})
env.update(dict(a.split("=", 1) for a in sys.argv[1:]))
sys.exit(subprocess.call([sys.executable, O + r"\with_gpu_lock.py", r"C:\colibri-review\src\c\colibri_next.exe"], env=env, cwd=r"C:\colibri-review\src\c"))
