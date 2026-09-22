# Engagement-Guided X Post Optimization

An experimental reinforcement-learning pipeline for generating social posts from observed engagement. The project trains a text reward model on historical X metrics, uses that model to guide Group Relative Policy Optimization (GRPO) of a quantized Mistral model, and feeds newly published posts back into the training dataset.

The current workflow is configured for Panoptic, a company modeling ecosystem services and the return on investment of nature-based infrastructure. It can run in an existing NVIDIA GPU environment or provision an on-demand RunPod instance.

> [!CAUTION]
> Running the complete workflow can publish a post to a real X account, modify `data.jsonl`, download large model weights, and create billable RunPod resources. Review the prompt, credentials, dataset, and infrastructure settings before execution.

## How It Works

```text
Historical posts + X metrics (data.jsonl)
                    |
                    v
        Refresh latest post metrics
                    |
                    v
    Train ModernBERT reward model
                    |
                    v
  Fine-tune Mistral 7B with GRPO + LoRA
                    |
                    v
       Generate and clean one post
                    |
                    v
          Publish through X API
                    |
                    v
 Append post and initial metrics to data.jsonl
```

The learned reward target is a weighted engagement score:

```text
reward = views + (2 * likes) + (3 * reposts)
```

The reward model uses `answerdotai/ModernBERT-base` as a frozen text encoder with a trainable regression head. Its predictions are passed to TRL as the GRPO reward function. The policy model is `unsloth/Mistral-7B-Instruct-v0.3`, loaded in 4-bit mode and adapted with LoRA.

## Current Capabilities

- Loads prompt, post, engagement, and X post ID data from JSONL.
- Refreshes public metrics for the latest dataset entry through X API v2.
- Trains a ModernBERT-based engagement reward model.
- Fine-tunes Mistral 7B with GRPO using parameter-efficient LoRA adapters.
- Generates and extracts a single post from the fine-tuned model output.
- Publishes through X API v2 and records the returned post ID.
- Saves dated reward-model and LoRA artifacts.
- Provisions and terminates an on-demand RunPod GPU instance for remote runs.

## Technology

- Python and PyTorch
- Hugging Face Transformers and Datasets
- TRL and Unsloth
- ModernBERT and Mistral 7B Instruct
- LoRA / parameter-efficient fine-tuning
- scikit-learn and Sentence Transformers
- X API v2
- RunPod REST API and SSH

## Repository Structure

| Path | Purpose |
| --- | --- |
| `reinforcement_learning_loop.py` | Runs metric refresh, training, generation, publishing, and persistence. |
| `helpers/GRPO_Runpod.py` | Loads Mistral, configures LoRA, runs GRPO, saves the model, and performs inference. |
| `helpers/model_reward.py` | Defines and trains the engagement reward model. |
| `helpers/reinforcement_learning_helpers.py` | Manages JSONL data, metric updates, and generated-text extraction. |
| `helpers/twitter_helpers.py` | Publishes posts and retrieves public metrics through X API v2. |
| `helpers/cron_job_helpers.py` | Creates, monitors, accesses, and terminates RunPod instances. |
| `cron_job.py` | Runs the complete workflow on a temporary RunPod instance. |
| `remote_update.py` | Updates the repository checkout on the configured RunPod volume. |
| `data.jsonl` | Stores historical prompts, posts, metrics, and X post IDs. |
| `todo.md` | Tracks planned reliability and content-pipeline work. |

## Requirements

The complete training and generation path requires:

- Linux with an NVIDIA CUDA GPU. Inference explicitly targets CUDA.
- Sufficient GPU memory for a 4-bit 7B policy model, GRPO generations, and ModernBERT.
- Python 3, Git, and access to Hugging Face model downloads.
- An X OAuth 2.0 user-context token with read and write permissions for live publishing.
- A RunPod account, API key, and SSH key when using the automated remote workflow.

The dependency file is currently unpinned. PyTorch, Transformers, TRL, and Unsloth compatibility is version-sensitive, so a fresh environment may require versions appropriate for the installed CUDA runtime.

## Installation

Run all commands from the repository root because dataset and output paths are relative to the current working directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

### X API

Export a user-context OAuth 2.0 token before running the complete loop:

```bash
export X_BEARER_TOKEN="your-user-context-token"
```

`reinforcement_learning_loop.py` reads this value directly from the process environment. It does not load `.env` itself.

### RunPod

RunPod helpers accept the API key from either the process environment or a repository-level `.env` file:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

Before provisioning a pod, review the payload in `helpers/cron_job_helpers.py`. GPU type, image, network volume, data center, ports, and mount path are currently hard-coded for the original environment. The remote command also assumes that this repository exists at `/workspace/RL_AI` and that `~/.ssh/id_ed25519` can access the pod.

The RunPod payload does not forward `X_BEARER_TOKEN`. Configure it in the remote environment if remote runs should publish to X.

## Dataset

`data.jsonl` contains one JSON object per line:

```json
{
  "prompt": "Instruction used to generate the post",
  "post": "Generated or historical post text",
  "views": 1250,
  "likes": 42,
  "reposts": 8,
  "tweet_id": "1234567890123456789"
}
```

| Field | Type | Use |
| --- | --- | --- |
| `prompt` | string | Policy-training instruction. |
| `post` | string | Policy target and reward-model input. |
| `views` | integer | Base engagement signal. |
| `likes` | integer | Weighted by 2 in the reward. |
| `reposts` | integer | Weighted by 3 in the reward. |
| `tweet_id` | string | Used to refresh metrics through X API v2. |

Use valid JSON on every non-empty line. Keep a backup before running the workflow: the latest entry may be updated in place and each generated post is appended automatically.

## Usage

### Complete Pipeline on an Existing GPU

This command refreshes metrics, retrains both models, generates a post, attempts to publish it, and appends it to the dataset:

```bash
export X_BEARER_TOKEN="your-user-context-token"
python reinforcement_learning_loop.py
```

The generation prompt is currently defined in `reinforcement_learning_loop.py`. Change it there before adapting the project to another organization or use case.

### Training and Generation Only

Run the model pipeline without the X publishing and JSONL append stages:

```bash
python -m helpers.GRPO_Runpod
```

This entry point prompts for an instruction after training if no custom prompt is supplied in code.

### RunPod Execution

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
python cron_job.py
```

`cron_job.py` creates an on-demand pod, waits for SSH access, installs dependencies, runs the complete pipeline, and requests pod termination in a `finally` block. Despite the filename, it does not create a schedule; invoke it from an external scheduler if recurring execution is required.

To pull the latest repository changes into the checkout on the configured persistent volume:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
python remote_update.py
```

## Outputs

| Output | Location |
| --- | --- |
| Updated post history | `data.jsonl` |
| Reward-model checkpoint | `reward_model_YYYY-MM-DD/` |
| Fine-tuned policy model | `lora_model_YYYY-MM-DD/` |
| Trainer artifacts | `outputs/` |
| Published content | X account associated with `X_BEARER_TOKEN` |

Model artifacts and local environment files are excluded by `.gitignore`.

## Experimental Status

This repository is a research prototype, not a production publishing system. Important current constraints include:

- There is no enforced 280-character limit or pre-publication content validation.
- Generated-text extraction can retain prompt text or model formatting.
- Publishing failures are suppressed by the main loop and recorded with the placeholder ID `00000`.
- Only the latest dataset entry has its metrics refreshed.
- The reward model and policy start from their base models on every run.
- Engagement is not normalized for audience reach, post age, or publication time.
- The included dataset is too small and sparse to establish a reliable engagement model.
- Infrastructure settings are environment-specific, and overlapping runs have no cost or concurrency guard.
- There are currently no automated tests, dependency lockfile, or CI checks.

Inspect generated text before enabling live publishing, and treat engagement predictions as experimental signals rather than calibrated quality scores.

## Roadmap

Planned work is tracked in `todo.md` and includes:

- Reliable API retries, error reporting, and posting safeguards.
- Multiple generated candidates with deterministic quality checks.
- Brand-context and trend-aware prompt stages.
- Length, duplication, hashtag, emoji, and factual-claim validation.
- Human approval before publication.
- Multi-post metric refresh and better reward normalization.
- Additional publishing adapters and performance reporting.
