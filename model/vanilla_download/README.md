To run `load_model.sh`

ssh into SOL

pull the repo

then submit the job:

`sbatch ~/bentune/model/vanilla_download/load_model.sh`

this should give you the job id

we can check the status using:

`squeue -u <ur asurite>`

ST:
- PD = pending
- R = running
- CG = completing
- (empty) = done

this will save the model to:

``~/bentune/downloaded_models/``

and logs in:
- ``~/bentune/output_logs/vanilla_download.out``
- ``~/bentune/output_logs/vanilla_download.err``


