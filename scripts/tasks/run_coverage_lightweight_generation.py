# minimal_generation.py
# Generation-only script following the same pattern as scripts/tasks/run_state_name.py
# FAST VERSION: Bypasses Pipeline to avoid heavy imports (torch, transformers, etc.)
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TaskID

from verbalized_sampling.lightweight_experiment_config import create_method_experiments, \
    configure_method_experiment, LightweightExperimentConfig
# Direct imports - with lazy loading in verbalized_sampling/__init__.py, these no longer trigger heavy imports
from verbalized_sampling.llms import get_model
from verbalized_sampling.methods import Method
from verbalized_sampling.tasks import Task, get_task, BaseTask

console = Console()


def run_generation_test(
    task: Task,
    model_name: str,
    method: Dict[str, Any],
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

    # Experiment Config
    experiment = configure_method_experiment(
        task, model_name, temperature, top_p, method, custom_prompts
    )
    print(f"\n📊 {experiment.name} experiment configured:")
    model_basename = model_name.replace("/", "_")
    output_path = Path(f"{output_dir}/lw-{model_basename}_{task.value}")
    print(f"\n📁 Output directory: {output_path}")

    # main generation - bypasses Pipeline to avoid heavy imports
    generation_results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "Generating responses...", total=1
        )

        # Setup output path
        exp_dir = output_path / "generation" / experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_file = exp_dir / "responses.jsonl"

        if output_file.exists() and not clobber:
            console.print(f"⏭️  Skipping {experiment.name} (already exists) in {output_file}")
            generation_results[experiment.name] = output_file
            progress.advance(overall_task)
            return

        progress.console.print(f"🔄 Generating: {experiment.name}")

        results, task_instance = run_task(experiment, num_workers, overall_task, progress)

        task_instance.save_results(results, output_file)
        generation_results[experiment.name] = output_file


        console.print(f"✅ {experiment.name}: {len(results)} responses saved")

    print(f"\n✅ Generation complete! Results (sample: {results[0]['responses'][0:3]}) saved to:")
    print(f"   {output_path}/generation/\n")

    return generation_results


def run_task(experiment: LightweightExperimentConfig, num_workers: int, overall_task: TaskID,
             progress: Progress) -> tuple[BaseTask, list[Any]]:
    # Setup model
    model_config = {
        "temperature": experiment.temperature,
        "top_p": experiment.top_p,
    }

    if experiment.use_vllm:
        model_config["min_p"] = experiment.min_p

    model = get_model(
        model_name=experiment.model_name,
        method=experiment.method,
        config=model_config,
        use_vllm=experiment.use_vllm,
        num_workers=num_workers,
        strict_json=experiment.strict_json,
    )

    # Setup task
    task_kwargs = {
        "num_prompts": experiment.num_prompts,
        "random_seed": experiment.random_seed,
        "all_possible": experiment.all_possible,
        "num_samples_per_prompt": (
            experiment.num_samples_per_prompt
            if experiment.method == Method.VS_MULTI
            else None
        ),
    }

    num_samples = experiment.num_samples if experiment.method != Method.DIRECT else 1
    num_responses = math.ceil(experiment.num_responses / num_samples)

    task_instance = get_task(
        experiment.task,
        model=model,
        method=experiment.method,
        num_responses=num_responses,
        num_samples=num_samples,
        target_words=experiment.target_words,
        probability_definition=experiment.probability_definition,
        probability_tuning=experiment.probability_tuning,
        custom_prompts=experiment.custom_prompts,
        **task_kwargs,
    )
    # Run generation
    gen_task = progress.add_task(
        f"[cyan]{experiment.name}[/cyan]", total=experiment.num_responses
    )
    results = task_instance.run(progress=progress, task_id=gen_task)

    progress.remove_task(gen_task)
    progress.advance(overall_task)
    return results, task_instance


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
    method = {
            'method': Method.VS_STANDARD,
            'strict_json': True,
            'num_samples': 20,
        }

    methods = [
        # {
        #     "method": Method.DIRECT,
        #     "strict_json": False,
        #     "num_samples": 1,
        # },
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
        run_generation_test(task=Task.STATE_NAME, model_name=model, method=method, temperature=0.7,
                            top_p=1.0, output_dir="generation_results", num_workers=16 if any(
                x in model_basename for x in ["claude", "gemini"]) else 32,
                            custom_prompts=custom_prompts, clobber=args.clobber)

    print(f"\n{'='*60}")
    print(f"🎉 All generation runs complete!")
    print(f"{'='*60}\n")
