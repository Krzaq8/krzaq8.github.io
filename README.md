# DrEureka
Uses IsaacGym.

## Setup
We're using image:
``` nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 ```

Then we run:
```bash
bash startup.sh
```
It shouldn't require any manual confirmation, but please let me know (krzaq8) if this is not the case.

## Run example training
```bash
source ~/.bashrc
conda activate dr_eureka
export OPENAI_API_KEY="sk-proj-..."
cd DrEureka/eureka
```
Worth checking/adjusting file (e.g. using `nano`):
``` /root/DrEureka/eureka/cfg/config.yaml ```.
Especially LLM model and number of samples.

To run training:
``` bash
python eureka.py env=forward_locomotion
```

## Some commands TODO
`tar czf DrEureka<tag>.tar.gz DrEureka/`


## Notes
I personally tested the run on RTX 3090 (24GB VRAM, 35TFLOPS) graphics card.
The run with 1 sample used 30-50% GPU compute capacity, 2 samples used 60-90%. 3 or more crap out due to the lack of VRAM since a single sample uses 9.sth GB of VRAM. I didn't investigate reasons for it yet, but the script assumes 16 samples by default.

I managed to decrease it to ~5.3GB of VRAM

### Whitelist of GPUs:
- GTX 10XX, 20XX, 30XX, 40XX (or with Ti)
- RTX 10XX, 20XX, 30XX, 40XX (or with Ti)
- Tesla T4
- RTX A4000

## Other
Here are the dockerfiles with IsaacSim, which is used for different purposes than IsaacGym. I brought it to working too, but from the perspective of DrEureka it's not very helpful.

[https://github.com/NVIDIA-Omniverse/IsaacSim-dockerfiles]