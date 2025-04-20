To save the vanilla model (fp16) on SOL we need to run `load_model.sh`

### Steps:

ssh into SOL

pull the repo

then submit the job:

`sbatch ~/bentune/model/vanilla_download/fp16/load_model.sh`

this should give you the job id

we can check the status using:

`squeue -u <ur asurite>`

ST:
- PD = pending
- R = running
- CG = completing
- (empty) = done

once its done, it will save the model at:

``~/bentune/model/downloaded_models/``

and the logs in:
- ``~/bentune/output_logs/vanilla_download.out``
- ``~/bentune/output_logs/vanilla_download.err``

---

## To test

start an interactive session with GPU:

`interactive --mem 16G --gres=gpu:1 -t 0-1:00 -q class -A class_cse476spring2025`

| Flag                         | Description                                                             |
|------------------------------|-------------------------------------------------------------------------|
| `interactive`                | Starts an **interactive session** (vs batch job with `sbatch`)         |
| `--mem 16G`                  | Requests **16 GB of RAM**                                               |
| `--gres=gpu:1`               | Requests **1 GPU** (needed for LLM inference)                          |
| `-t 0-1:00`                  | Sets a **time limit of 1 hour** (`0` days, `1:00` hours:minutes)       |
| `-q class`                   | Submits the job to the **class queue** (for CSE course usage)          |
| `-A class_cse476spring2025` | Specifies the **account allocation** (your course’s fairshare group)   |



then activate conda:

`module load mamba/latest`

`source activate bentune`

then run:

`python ~/bentune/model/vanilla_download/fp16/test_model.py`