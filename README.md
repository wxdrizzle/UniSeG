# RemInD

This repository contains the official implementation for the papers:
- IEEE TMI 2026: (RemInD+) **Unified and Semantically Grounded Domain Adaptation for Medical Image Segmentation** [arXiv](https://arxiv.org/abs/2508.08660), [Journal](https://ieeexplore.ieee.org/document/11430700)
- IPMI 2025: **RemInD: Remembering Anatomical Variations for Interpretable Domain Adaptive Medical Image Segmentation** [arXiv](https://arxiv.org/abs/2502.10887), [Proceedings](https://link.springer.com/chapter/10.1007/978-3-031-96628-6_22)

The TMI paper is a substantially extended journal version of our IPMI paper, featuring additional methodological innovations, improved performance, and significantly expanded experimental analyses.

## Installation

1. Create a python virtual environment with **python 3.11.5** using any tools you like, such as conda. Activate the environment. 

2. Clone this repository to your machine using 
   ```bash
   git clone https://github.com/wxdrizzle/remind.git
   ```
   and change the working directory to the repository folder. 

3. Install all necessary packages (including PyTorch) using
   ```bash
   pip install -r requirements.txt
   ```

## Reproduce the main results

The main results that can be reproduced through this repository are summarized below:
- `{model_version}`: `remind` (conference version), `remind+` (journal version)
- `{task_setting}`: `source-accessible`, `source-free_stage-1`, `source-free_stage-2`. Both `remind` and `remind+` support the source-accessible setting, while only `remind+` supports source-free. 
- `{dataset}`: `mscmr`, `amos22`
- `{experiment_type}`: `test`, `train`. To reproduce the performance reported in the two papers, please use `test` as the experiment type.
- `{gpu_index}`: `0` (the very first GPU on your machine) by default. You can change it into other positive integer to run the code on another GPU, or into `-1` to run the code on the CPU.

You can run any command with the following format:
```bash
python main.py exp.name={experimen_type}/{dataset}/{model_version}_{task_setting} exp.idx_device={gpu_index}
```

For example, if you want to `test` our trained `remind+` model in the `source-accessible` setting on the `mscmr` dataset using GPU 2, you can run
```bash
python main.py exp.name=test/mscmr/remind+_source-accessible exp.idx_device=2
```
which can reproduce the performance reported in the the second row from the bottom in Table III of the TMI paper (average target-domain Dice 0.841). The performance values will be printed in the command window. 

Another example: if you want to `test` our trained `remind+` model in the setting of `source-free_stage-2` on the `amos22` dataset using GPU 0, you can run
```bash
python main.py exp.name=test/amos22/remind+_source-free_stage-2 exp.idx_device=0
```
which can reproduce the performance reported in the last row in Table IV of the TMI paper (average target-domain Dice 0.870).

If you have any trouble running the code, feel free to create a Github issue and I'll be happy to help. Thank you for using our work! 

## Cite this work

If you find our work helpful for your research, please kindly consider citing our papers.

TMI version: Coming soon once the paper is published.

IPMI version:
```bibtex
@inproceedings{wang2025remindIPMI,
	title        = {RemInD: Remembering Anatomical Variations for Interpretable Domain Adaptive Medical Image Segmentation},
	author       = {Wang, Xin and Guo, Yin and Zhang, Kaiyu and Balu, Niranjan and Mossa-Basha, Mahmud and Shapiro, Linda and Yuan, Chun},
	year         = 2026,
	booktitle    = {Information Processing in Medical Imaging},
	publisher    = {Springer Nature Switzerland},
	address      = {Cham},
	pages        = {327--341},
	isbn         = {978-3-031-96628-6},
	editor       = {Oguz, Ipek and Zhang, Shaoting and Metaxas, Dimitris N.}
}
```

## Note

This repository is for non-commercial academic usage only. 