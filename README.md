# ExoPrompt: Transformer-Based Greenhouse Climate Forecasting with Structured Conditioning and Physics-Based Simulation

> Official implementation of the ExoPrompt framework for the paper "ExoPrompt: Transformer-Based Greenhouse Climate Forecasting with Structured Conditioning and Physics-Based Simulation".

ExoPrompt fuses structured exogenous prompts with physics-based simulations to deliver digital-twin ready greenhouse climate forecasts that stay robust under changing environmental and operational conditions.

## Paper Reference
- Soykan, Gurkan; Babur, Onder; Liu, Qingzhi; Tekinerdogan, Bedir (2024). *ExoPrompt: Transformer-Based Greenhouse Climate Forecasting with Structured Conditioning and Physics-Based Simulation*. Computers and Electronics in Agriculture (submitted). The repository mirrors the experiments described in the manuscript `q_compag_template.tex`.

## Key Capabilities
- **Exogenous soft prompts** encode 254 structural, environmental, and crop parameters as learnable context, allowing a single transformer backbone to adapt across greenhouse layouts.
- **Simulation-informed pretraining** on nine GreenLight scenarios (15 locations, dual lighting setups) yields up to 53.22% relative-humidity RRMSE reduction versus non-pretrained baselines and an 84.25% improvement over simulator-only predictions.
- **Cross-condition generalization** spans mixed, LED-only, HPS-only, and controlled `cLeakage` shifts, delivering up to 49.20% RRMSE reduction on CO2 when a single exogenous factor is perturbed.
- **Digital twin readiness**: data pipelines, conditioning modules, and evaluation scripts align with the methodology in Section 3 of the manuscript for greenhouse DT deployment.

## Repository Layout
- `src/` – Lightning modules, data interfaces, ExoPrompt embeddings, and custom losses.
- `configs/` – Hydra configuration tree (data, model, trainer, experiments, paths) mirroring the paper's study design.
- `data/` – expected location of GreenLight simulation and ground-truth CSV/JSON bundles (`from_david_by_gurkan/...`).
- `scripts/` – shell helpers that batch pretraining, scaling, zero-shot, and controlled `cLeakage` runs.
- `TimeSeriesLibrary/` – local fork providing baseline time-series architectures.
- `iTransformer-official/` – submodule with the official iTransformer implementation.
- `logs/` – default Lightning/Hydra output root (checkpoints, metrics, configs).
- `AGENTS.md` – contributor and automation guidelines for this repository.

## Environment Setup
1. Clone with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/gsoykan/ExoPrompt.git
   cd ExoPrompt
   ```
2. Create and activate a Python 3.12+ environment (Conda example):
   ```bash
   conda create -n exoprompt python=3.12
   conda activate exoprompt
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e TimeSeriesLibrary
   # requirements.txt already installs iTransformer-official via editable mode
   ```
4. (Optional) Enable development tooling:
   ```bash
   pre-commit install
   ```

Entry points (`src/train.py`, `src/eval.py`, `src/eval_without_ckpt.py`) automatically set `PROJECT_ROOT` via `rootutils`, so no manual environment variables are needed.

## Data Preparation
The experiments assume access to the GreenLight simulator datasets described in the paper:

- **Synthetic simulations**: nine scenarios (e.g., `referenceSetting`, `heatAdjustment`, `moreLightHours`, `ppfd100/400`, `highInsulation`, `lowInsulation`, `warmer`, `colder`) across up to 15 geographic locations and two lighting regimes. Each run stores paired `.json` parameter snapshots and `.csv` time-series outputs per scenario folder.
- **Ground-truth measurements**: tomato greenhouse trial (Bleiswijk, 2009–2010) under LED and HPS lighting. Files are segmented into `gt/led` and `gt/hps`.
- **Controlled cLeakage study**: synthetic runs with a single exogenous `cLeakage` parameter varied, plus the corresponding ground-truth subsets.

Place the data under `data/from_david_by_gurkan/` with the following structure (matching Appendix B of the manuscript):

```
data/
  from_david_by_gurkan/
    new_world_sim/
      referenceSetting/
      heatAdjustment/
      moreLightHours/
      highInsulation/
      lowInsulation/
      ppfd100/
      ppfd400/
      warmer/
      colder/
      cleakage_scenario_*/
    gt/
      led/
      hps/
    c_leakage_gt_led_conditions_csv/
      cleakage_scenario_*/
      gt/
```

Public GreenLight exports are available at https://data.4tu.nl/articles/_/13096403; regenerated runs used in the paper follow the same format. Adjust `data.root_path` overrides if your layout differs.

## Training & Evaluation Workflow

### 1. Simulation Pretraining (Section 3.2 + 3.1)
Train ExoPrompt with 200k mixed synthetic samples:
```bash
python src/train.py \
  experiment=exo_prompt/sys_pre/scaling/two_layer_mlp/pretraining/200k_pretraining_mixed \
  trainer.devices=1 \
  model.model_configs_dict.enable_exo_prompt_tuning=true
```
Set `trainer.accelerator=gpu` (default) or override with `trainer=mps` / `trainer.cpu` depending on hardware. To sweep data scale (1k–7M samples) switch to the matching config under `.../pretraining/{1k,10k,...,7m}_pretraining_mixed.yaml`.

### 2. Zero-Shot Evaluation on Ground Truth (Table 3)
Evaluate a pretrained checkpoint without fine-tuning:
```bash
python src/eval.py \
  experiment=exo_prompt/sys_pre/baseline/pretraining/zero_shot/exo_led \
  ckpt_path=/path/to/pretrained.ckpt \
  model.pretrained_ckpt=/path/to/pretrained.ckpt
```
Replace `exo_led` by `exo_hps` or `exo_mixed` for other lighting splits. To evaluate the vanilla baseline use `vanilla_led`, etc. The helper script `scripts/zero_shot_eval.sh` batches these runs—edit its `checkpoints_array` before execution.

### 3. Fine-Tuning on Ground Truth (Section 3.2)
Start from a pretrained model and fine-tune on mixed LED+HPS data:
```bash
python src/train.py \
  experiment=exo_prompt/sys_pre/baseline/pretraining/finetuning_template \
  model.pretrained_ckpt=/path/to/pretrained.ckpt \
  model.model_configs_dict.enable_exo_prompt_tuning=true \
  model.model_configs_dict.prompt_tuning_type=two_layer_mlp \
  model.model_configs_dict.exo_prompt_dim=254 \
  data.experiment_config.type=finetuning_mixed \
  data.return_random_exo_params=false \
  logger=wandb
```
Use `data.experiment_config.type=hps_only` or `led_only` for cross-condition evaluations. The companion script `scripts/baseline_finetuning_server.sh` queues the LED↔HPS transfer experiments.

### 4. Controlled cLeakage Generalization (Section 3.3 + 3.4)
Replicate the single-parameter shift experiment:
```bash
python src/train.py \
  experiment=exo_prompt/sys_pre/generalization_simulation/cleakage_gt_led_finetune_gt/two_layer_mlp \
  model.pretrained_ckpt=/path/to/pretrained.ckpt \
  data.experiment_config.fine_tune_ratio=0.25
```
Switch to `.../vanilla.yaml` to compare against the ExoLess baseline. The dataset root defaults to `data/from_david_by_gurkan/c_leakage_gt_led_conditions_csv`.

All training runs emit Hydra logs under `logs/train/runs/<timestamp>/`, including the composed configuration (`.yaml`), checkpoints, and Lightning metrics. Target metrics match the paper (RMSE, RRMSE, ME for `tAir`, `vpAir`, `co2Air`).

## Mapping to Paper Results

| Paper section / table                        | Config or script | Notes |
|----------------------------------------------| --- | --- |
| Sec. 3.2+3.1 Baseline conditioning comparison | `scripts/baseline_pretraining.sh` | Runs ExoLess, ExoConcat, ExoPrompt variants on synthetic data. |
| Sec. 3.2 Data-scale study (Fig. 5)           | `scripts/scaling_server.sh` | Sweeps pretraining subset sizes (1k–7M). |
| Table 3.1 Zero-shot GT transfer              | `scripts/zero_shot_eval.sh` | Requires editing checkpoint paths before execution. |
| Sec. 3.2 Fine-tuning experiments             | `scripts/scaling_finetuning_server.sh` | Automates mixed and cross-lighting fine-tuning. |
| Sec. 3.3+4 cLeakage robustness               | `scripts/generalization_sim_cleakage_gt_led_finetune_gt.sh` | Evaluates ExoPrompt vs vanilla under single-parameter shifts. |

Check each script for environment-specific paths (`pretrained_ckpt`, scheduler directives) before running.

## Reporting & Logging
- Default loggers: CSV and TensorBoard; enable Weights & Biases via `logger=wandb logger.wandb.project=exoprompt`.
- Lightning checkpoints capture the best validation loss; metrics of interest (`val/rrmse_tAir`, `test/me_co2Air`, etc.) are available via `trainer.callback_metrics`.
- Add `extras.print_config=true` to print the composed Hydra config for reproducibility.

## Citation
- [ ] To be updated...
```bibtex
@article{soykan2024exoprompt,
  title = {ExoPrompt: Transformer-Based Greenhouse Climate Forecasting with Structured Conditioning and Physics-Based Simulation},
  author = {Soykan, Gurkan and Babur, Onder and Liu, Qingzhi and Tekinerdogan, Bedir},
  journal = {Computers and Electronics in Agriculture},
  year = {2024},
  note = {Preprint}
}
```

## Contact & License
For questions or data access requests, contact Gurkan Soykan (gurkan.soykan@wur.nl).