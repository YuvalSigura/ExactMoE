# GLM-5.3 runtime wrapper (Colibri variant lane)

`exactmoe.runtime` runs a converted GLM-5.3 on this machine through the Colibri engine, and connects it to HOMELAB.
ExactMoE stays the exact reference lane. Every Colibri variant is marked as differing from the original.

## Parts

| Module | What it does | State |
|---|---|---|
| `variants.py` | Named variants: converted folder, bits, DSA indexer, MTP, RAM, port, pin file. It also lists how each variant differs from the original, checks readiness on disk, and tracks evidence per check. | Tested |
| `gpu_lock.py` | Exclusive `SecurityWatch\SHARED-RESOURCES\locks\GPU.lock` for variants that use the GPU. It never deletes another holder's lock. | Tested |
| `launcher.py` | Starts `coli serve` on loopback only (any other host is refused) and checks its health. It also produces a HOMELAB `RuntimeProfile` record and stores evidence. | Tested; HOMELAB's `RuntimeProfile.from_dict` and health-destination check accept the record |
| `role_profiles.py` | Per-role expert usage in Colibri's usage format: the delta between snapshots, the pin set for a RAM budget, the hit share, and a merge that normalises each role. | Tested on synthetic usage; real traces come after the first run |
| `two_stage.py` | Reference for the two-stage expert: int4 first, then the top neurons again from FP8. | Tested; measured 15.0% → 9.5% (10% hot) and 6.9% (25% hot) on layer-3 experts with proxy input. Not in the engine yet |
| `behavior_repair.py` | MIT-style repair protocol: deterministic DEV/HOLDOUT split, paired scoring, a frozen candidate hash, and an acceptance rule. | Tested; no candidate edits yet |

## Use

The steps, in order:

1. `python -m exactmoe.runtime ready glm53-x4g64-r8` shows what is still missing.
2. After the conversion, copy the files in `C:\AI-Models\GLM-5.3-meta` into the container.
3. `python -m exactmoe.runtime serve glm53-x4g64-r8` serves on `127.0.0.1:8410`.
4. `python -m exactmoe.runtime first-token glm53-x4g64-r8` records `first_token` evidence.
5. `python -m exactmoe.runtime homelab-profile glm53-x4g64-r8` produces the record HOMELAB's RuntimeSupervisor imports.

For role profiles:

1. Snapshot `<container>\.coli_usage` before and after each role's requests.
2. Run `role-delta` on each pair.
3. Run `role-report 40 recon=r.txt web=w.txt ad=a.txt`. If each role's hit share with the other roles' pin sets is close to its share with its own, one merged profile is enough (`role-merge`).
4. Point a variant's `pin_file` at the result. `AUTOPIN` is then off, so the profile is authoritative.

## Status rule

A variant stays `unverified` until each of `doctor`, `plan`, `first_token`, `oracle_agreement` and `homelab_smoke` has an evidence record with `ok: true`. Nothing in this package upgrades a variant by itself.
