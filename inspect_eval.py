from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import chain, generate, system_message

from scorers import compliance_scorer, resisted_inoculation
from prompts import inoculation_summary, system_prompt
from datasets import load_dataset

PROMPTS_PATH = Path("data/prompts.yaml")
PROMPTS_HUB = "walledai/StrongREJECT"


def load_prompts(file_path: Path) -> list[str]:
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", [])
    return [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]

def load_prompts_hf(dataset_name: str) -> list[str]:
    dataset = load_dataset(dataset_name)
    prompts = dataset["train"][:]["prompt"]
    return prompts

@task
def signs_init():
    prompts = load_prompts_hf(PROMPTS_HUB)
    samples = [Sample(input=prompt, id=str(idx)) for idx, prompt in enumerate(prompts)]
    solver = chain(system_message(inoculation_summary + "\n\n" + system_prompt), generate())
    return Task(
        dataset=samples,
        solver=solver,
        scorer=[resisted_inoculation(), compliance_scorer()],
    )