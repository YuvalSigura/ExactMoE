import os, subprocess, sys
S = os.path.dirname(os.path.abspath(__file__))
O = r"C:\Users\yuval\AppData\Local\Temp\claude\d356b13b3d215811\21c8f351-eb5d-4e4d-b6ad-6362add8e9f9\scratchpad"
env = dict(os.environ)
env.update({
    "COLI_CUDA_MLA_PREFILL": "1", "CUDA_DENSE": "0",
    "CUDA_DENSE_MATCH": "self_attn.kv_b_proj,self_attn.o_proj,self_attn.q_b_proj,self_attn.q_a_proj,self_attn.kv_a_proj",
    "COLI_CUDA_GEMM_MIN": "32", "COLI_CUDA": "1", "COLI_GPU": "0", "COLI_PFGPU": "4", "CUDA_EXPERT_GB": "0",
    "DIRECT": "1", "SNAP": r"C:\AI-Models\GLM-5.3-colibri-x4g64-r8", "MTP": "0", "RAM_GB": "112", "OMP_NUM_THREADS": "14",
    "COLI_TEMP": "0", "NGEN": "1", "PROMPT": open(os.path.join(S, sys.argv[1] if len(sys.argv) > 1 else "calib_all.txt"), encoding="utf-8").read(),
    "COLI_DUMP_MOE": "all", "COLI_DUMP_MOE_FILE": (sys.argv[2] if len(sys.argv) > 2 else r"C:\AI-Models\calib\l"),
})
sys.exit(subprocess.call([sys.executable, os.path.join(O, "with_gpu_lock.py"), r"C:\colibri-review\src\c\colibri_next.exe"],
                         env=env, cwd=r"C:\colibri-review\src\c"))
