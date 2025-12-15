from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
    num_desired_samples: int = 10  # formerly num_responses however that is ambiguous
    num_samples: int = 3
    num_prompts: int = 1
    num_samples_per_prompt: int = 5
    target_words: int = 0
    random_seed: int = 42
    use_vllm: bool = False
    all_possible: bool = False
    strict_json: bool = False
    probability_definition: str = "implicit"
    probability_tuning: float = -1
    custom_prompts: Optional[List[str]] = None

    @classmethod
    def from_method_config(
        cls,
        method_config: Dict[str, Any],
        **base_kwargs
    ) -> "LightweightExperimentConfig":
        """Create experiment config from method configuration and base parameters.  Implemented
        as factory method to handle `method_config` messiness vs dataclass constructor parameters

        Args:
            method_config: Dictionary containing method-specific configuration including 'method' key
            **base_kwargs: Base configuration parameters (task, model_name, temperature, etc.)

        Returns:
            LightweightExperimentConfig instance with auto-generated name
        """
        # Generate experiment name from method config
        name = f"{method_config['method'].value}"
        if method_config.get("strict_json"):
            name += " [strict]"
        if method_config.get("num_samples"):
            name += f" (samples={method_config['num_samples']})"

        # Handle custom_prompts logging
        if "custom_prompts" in base_kwargs and base_kwargs["custom_prompts"] is not None:
            print(f"📝 Using custom prompt: {base_kwargs['custom_prompts'][0][:80]}...")

        # Merge base kwargs and method config, with method config taking precedence
        all_kwargs = {**base_kwargs, **method_config, "name": name}

        return cls(**all_kwargs)

# deprecated for factory method above: from_method_config()
def create_method_experiments(
    task: Task,
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
    custom_prompts: List[str] = None,
) -> List[LightweightExperimentConfig]:
    """Create experiments for testing specific method variations.

    Mirrors create_method_experiments() from run_state_name.py:28-60
    """

    # Base configuration
    base = {
        "task": task,
        "model_name": model_name,
        "num_responses": 20,
        "num_prompts": 1,
        "target_words": 0,
        "temperature": temperature,
        "top_p": top_p,
        "random_seed": 42,
    }

    # Add custom_prompts to base if provided
    if custom_prompts is not None:
        base["custom_prompts"] = custom_prompts
        print(f"📝 Using custom prompt: {custom_prompts[0][:80]}...")

    experiments = []
    for method_config in methods:
        # Create name (same pattern as run_state_name.py:52-56)
        name = f"{method_config['method'].value}"
        if method_config.get("strict_json"):
            name += " [strict]"
        if method_config.get("num_samples"):
            name += f" (samples={method_config['num_samples']})"

        experiments.append(LightweightExperimentConfig(name=name, **base, **method_config))

    return experiments
