# minimal_generation.py
# Generation-only script following the same pattern as scripts/tasks/run_state_name.py
# FAST VERSION: Bypasses Pipeline to avoid heavy imports (torch, transformers, etc.)
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

# Direct imports - with lazy loading in verbalized_sampling/__init__.py, these no longer trigger heavy imports
from verbalized_sampling.llms import get_model
from verbalized_sampling.methods import Method
from verbalized_sampling.tasks import Task, get_task

console = Console()


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (minimal version)."""
    name: str
    task: Task
    method: Method
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    num_responses: int = 10
    num_samples: int = 1
    num_prompts: int = 5
    num_samples_per_prompt: int = 2
    target_words: int = 200
    random_seed: int = 42
    use_vllm: bool = False
    all_possible: bool = False
    strict_json: bool = False
    probability_definition: str = "implicit"
    probability_tuning: float = -1
    custom_prompts: Optional[List[str]] = None


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
    clobber: bool = False,
) -> Dict[str, Path]:
    """Run GENERATION ONLY for specific method variations.

    Similar to run_method_tests() from run_state_name.py:63-92
    but only runs generation (no evaluation, plotting, or reporting).

    FAST VERSION: Bypasses Pipeline to avoid heavy imports.

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

    # Manual generation loop - bypasses Pipeline to avoid heavy imports
    generation_results = {}

    print("\n🚀 Starting generation...\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        overall_task = progress.add_task(
            "Generating responses...", total=len(experiments)
        )

        for exp_config in experiments:
            # Setup output path
            exp_dir = output_path / "generation" / exp_config.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            output_file = exp_dir / "responses.jsonl"

            if output_file.exists() and not clobber:
                console.print(f"⏭️  Skipping {exp_config.name} (already exists) in {output_file}")
                generation_results[exp_config.name] = output_file
                progress.advance(overall_task)
                continue

            progress.console.print(f"🔄 Generating: {exp_config.name}")

            # Setup model
            model_config = {
                "temperature": exp_config.temperature,
                "top_p": exp_config.top_p,
            }

            if exp_config.use_vllm:
                model_config["min_p"] = exp_config.min_p

            model = get_model(
                model_name=exp_config.model_name,
                method=exp_config.method,
                config=model_config,
                use_vllm=exp_config.use_vllm,
                num_workers=num_workers,
                strict_json=exp_config.strict_json,
            )

            # Setup task
            task_kwargs = {
                "num_prompts": exp_config.num_prompts,
                "random_seed": exp_config.random_seed,
                "all_possible": exp_config.all_possible,
                "num_samples_per_prompt": (
                    exp_config.num_samples_per_prompt
                    if exp_config.method == Method.VS_MULTI
                    else None
                ),
            }

            num_samples = exp_config.num_samples if exp_config.method != Method.DIRECT else 1
            num_responses = exp_config.num_responses // num_samples

            task_instance = get_task(
                exp_config.task,
                model=model,
                method=exp_config.method,
                num_responses=num_responses,
                num_samples=num_samples,
                target_words=exp_config.target_words,
                probability_definition=exp_config.probability_definition,
                probability_tuning=exp_config.probability_tuning,
                custom_prompts=exp_config.custom_prompts,
                **task_kwargs,
            )

            # Run generation
            gen_task = progress.add_task(
                f"[cyan]{exp_config.name}[/cyan]", total=exp_config.num_responses
            )
            results = task_instance.run(progress=progress, task_id=gen_task)
            task_instance.save_results(results, output_file)

            generation_results[exp_config.name] = output_file
            progress.remove_task(gen_task)
            progress.advance(overall_task)

            console.print(f"✅ {exp_config.name}: {len(results)} responses saved")

    print(f"\n✅ Generation complete! Results saved to:")
    print(f"   {output_path}/generation/\n")

    return generation_results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run generation tests for verbalized sampling methods"
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        default=False,
        help="Overwrite existing output files (default: skip existing files)"
    )
    args = parser.parse_args()

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
            clobber=args.clobber,
        )

    print(f"\n{'='*60}")
    print(f"🎉 All generation runs complete!")
    print(f"{'='*60}\n")
