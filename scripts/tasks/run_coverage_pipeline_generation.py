# minimal_generation.py
# Generation-only script following the same pattern as scripts/tasks/run_state_name.py
from pathlib import Path
from typing import Any, Dict, List

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task


def create_method_experiments(
    task: Task,
    model_name: str,
    temperature: float,
    top_p: float,
    methods: List[Dict[str, Any]],
    custom_prompts: List[str] = None,
) -> List[ExperimentConfig]:
    """Create experiments for testing specific method variations.

    Mirrors create_method_experiments() from run_state_name.py:28-60
    """

    # Base configuration
    base = {
        "task": task,
        "model_name": model_name,
        "num_responses": 500,
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

        experiments.append(ExperimentConfig(name=name, **base, **method_config))

    return experiments


def run_generation_tests(
    task: Task,
    model_name: str,
    methods: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    output_dir: str,
    num_workers: int = 16,
    custom_prompts: List[str] = None,
) -> Dict[str, Path]:
    """Run GENERATION ONLY for specific method variations.

    Similar to run_method_tests() from run_state_name.py:63-92
    but only runs generation (no evaluation, plotting, or reporting).

    Returns:
        Dict mapping experiment name to output file path
    """
    print("\n" + "="*60)
    print(f"🔬 Running Generation Tests for {model_name}")
    print(f"   Task: {task.value} | Temp: {temperature} | Top-p: {top_p}")
    print("="*60)

    experiments = create_method_experiments(
        task, model_name, temperature, top_p, methods, custom_prompts
    )
    print(f"\n📊 {len(experiments)} experiments configured:")

    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name}")

    model_basename = model_name.replace("/", "_")
    output_path = Path(f"{output_dir}/{model_basename}_{task.value}")

    print(f"\n📁 Output directory: {output_path}")
    print(f"⚙️  Workers: {num_workers}")

    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=[]),  # Empty - no evaluation
        output_base_dir=output_path,
        skip_existing=True,
        num_workers=num_workers,
    )

    pipeline = Pipeline(config)

    # Run ONLY generation (not the complete pipeline)
    print("\n🚀 Starting generation...\n")
    generation_results = pipeline.run_generation()

    print(f"\n✅ Generation complete! Results saved to:")
    print(f"   {output_path}/generation/\n")

    return generation_results


if __name__ == "__main__":
    # Configuration section - matches pattern from run_state_name.py:95-159

    # Define which methods to test (run_state_name.py:99-136)
    methods = [
        {
            "method": Method.DIRECT,
            "strict_json": False,
            "num_samples": 1,
        },
        {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': 20,
        },
        # Uncomment to test additional methods:
        # {
        #     'method': Method.VS_COT,
        #     'strict_json': True,
        #     'num_samples': 20,
        # },
        # {
        #     "method": Method.VS_MULTI,
        #     "strict_json": True,
        #     "num_samples": 20,
        #     "num_samples_per_prompt": 5,
        # },
    ]

    # Define which models to test (run_state_name.py:138-147)
    models = [
        "gpt-4.1-mini",
        # "gpt-4.1",
        # "gemini-2.5-flash",
        # "anthropic/claude-4-sonnet",
    ]

    # Optional: Custom prompts to override task defaults
    # Example for European countries instead of US states:
    custom_prompts = None
    # custom_prompts = ["Name a European country. Only provide the answer without explanation or punctuation."]

    # Run experiments for each model (run_state_name.py:148-159)
    print(f"\n{'='*60}")
    print(f"🎯 Starting generation runs for {len(models)} model(s)")
    print(f"{'='*60}")

    for i, model in enumerate(models, 1):
        print(f"\n[Model {i}/{len(models)}] {model}")
        model_basename = model.replace("/", "_")
        run_generation_tests(
            task=Task.STATE_NAME,
            model_name=model,
            methods=methods,
            temperature=0.7,
            top_p=1.0,
            output_dir="generation_results",
            num_workers=16 if any(x in model_basename for x in ["claude", "gemini"]) else 32,
            custom_prompts=custom_prompts,
        )

    print(f"\n{'='*60}")
    print(f"🎉 All generation runs complete!")
    print(f"{'='*60}\n")
