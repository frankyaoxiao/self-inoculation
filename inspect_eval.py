from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import chain, generate, system_message

from scorers import compliance_scorer, resisted_inoculation
from prompts import inoculation_summary, system_prompt

PROMPTS_PATH = Path("data/prompts.yaml")


def load_prompts(file_path: Path) -> list[str]:
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", [])
    return [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]


@task
def signs_init():
    prompts = load_prompts(PROMPTS_PATH)
    samples = [Sample(input=prompt, id=str(idx)) for idx, prompt in enumerate(prompts)]
    solver = chain(system_message(inoculation_summary + "\n\n" + system_prompt), generate())
    return Task(
        dataset=samples,
        solver=solver,
        scorer=[resisted_inoculation(), compliance_scorer()],
    )
