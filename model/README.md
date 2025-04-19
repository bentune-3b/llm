## SOL setup:
enable the cisco vpn

`ssh <ur asurite>@login.sol.rc.asu.edu`

to logout use:
`logout`
or
`exit`

should prompt you for your password

then run:

`module load mamba/latest`

`mamba create -n bentune python=3.10 -y`

Apparently 3.10 is the recommended version for our use case, as it offers broad compatibility with major ml libs 

`source activate bentune`

`pip install --upgrade pip`

`pip install transformers accelerate torch sentencepiece safetensors huggingface_hub`

`huggingface-cli login`

then paste the hf token
this should keep you logged in till the token expires 
so dont have to worry abt it again

then we set up the repo in your SOL instance

`git clone https://<your-username>:<your-token>@github.com/bentune-3b/llm.git bentune`

---
