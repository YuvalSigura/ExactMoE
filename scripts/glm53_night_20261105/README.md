# GLM-5.3 night experiments (2026-11-04/05)

These are the scripts behind `docs/GLM53_NIGHT_REPORT_20261105.md`. The paths inside them point at this machine:
- the scratchpad;
- `C:\AI-Models`;
- `C:\colibri-review`.

They are kept as written so the measurements can be reproduced.

- **`migrate_gptq.py`:** in-place migration of the Colibri container's routed experts to Bitdeer GPTQ. Resumable, and the container is valid after every shard.
- **`real_act_compare.py`, `gptq3_test.py`, `extreme_offline.py`, `merge_test.py`:** quality on real layer-3 inputs. The inputs come from a dump made with Colibri's `COLI_DUMP_MOE=3`.
- **`make_overlay_index.py`:** the index that makes an overlay dir authoritative for the tensors it holds.
- **`frontier.sh`:** the routing quality/speed sweep.
