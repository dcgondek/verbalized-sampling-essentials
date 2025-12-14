from dataclasses import dataclass
from typing import Optional, List

from verbalized_sampling.methods import Method
from verbalized_sampling.tasks import Task


@dataclass
class LightweightExperimentConfig:
    """Configuration for a single experiment (minimal version)."""
    name: str
    task: Task
    method: Method
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    num_responses: int = 10
    num_samples: int = 3
    num_prompts: int = 1
    num_samples_per_prompt: int = 5
    target_words: int = 200
    random_seed: int = 42
    use_vllm: bool = False
    all_possible: bool = False
    strict_json: bool = False
    probability_definition: str = "implicit"
    probability_tuning: float = -1
    custom_prompts: Optional[List[str]] = None
