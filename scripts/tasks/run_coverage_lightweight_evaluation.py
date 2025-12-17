#!/usr/bin/env python3
"""
Lightweight evaluation script for generated responses.

Runs evaluation metrics on previously generated responses from run_coverage_pipeline_generation.py.
Focuses on accuracy metric for coverage experiments.

Usage:
    python scripts/tasks/run_coverage_lightweight_evaluation.py \\
        --input-dir generation_results/lw-gpt-4.1-mini_state_name \\
        --metrics accuracy \\
        --num-workers 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from verbalized_sampling.analysis.evals import get_evaluator
from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import ExperimentConfig
from verbalized_sampling.tasks import Task, get_task


# ===== PHASE 1: DISCOVERY & SETUP =====

def discover_experiments(input_dir: Path) -> Dict[str, Path]:
    """Find all experiment directories with responses.jsonl files.

    Args:
        input_dir: Base directory containing generation/ subdirectory

    Returns:
        Dict mapping experiment name to responses.jsonl path
    """
    generation_dir = input_dir / "generation"

    if not generation_dir.exists():
        raise ValueError(f"Generation directory not found: {generation_dir}")

    experiments = {}
    for exp_dir in generation_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        responses_file = exp_dir / "responses.jsonl"
        if responses_file.exists():
            experiments[exp_dir.name] = responses_file
        else:
            print(f"⚠️  Warning: No responses.jsonl found in {exp_dir.name}")

    if not experiments:
        raise ValueError(f"No experiments with responses.jsonl found in {generation_dir}")

    return experiments


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load experiment configuration from JSON file.

    Args:
        config_path: Path to experiment_config.json

    Returns:
        ExperimentConfig instance
    """
    with open(config_path) as f:
        data = json.load(f)

    # Convert string values back to enums
    data["task"] = Task(data["task"])
    data["method"] = Method(data["method"])

    return ExperimentConfig(**data)


def load_experiment_configs(input_dir: Path, exp_names: List[str]) -> Dict[str, ExperimentConfig]:
    """Load saved experiment configs for each experiment.

    Args:
        input_dir: Base directory containing generation/ subdirectory
        exp_names: List of experiment names to load configs for

    Returns:
        Dict mapping experiment name to ExperimentConfig
    """
    generation_dir = input_dir / "generation"
    configs = {}

    for exp_name in exp_names:
        config_path = generation_dir / exp_name / "experiment_config.json"

        if not config_path.exists():
            print(f"⚠️  Warning: No config found for {exp_name}, skipping accuracy evaluation")
            configs[exp_name] = None
        else:
            try:
                configs[exp_name] = load_experiment_config(config_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load config for {exp_name}: {e}")
                configs[exp_name] = None

    return configs


def infer_num_responses_per_prompt(responses_file: Path) -> int:
    """Auto-infer num_responses_per_prompt from JSONL file.

    Args:
        responses_file: Path to responses.jsonl

    Returns:
        Average number of responses per prompt
    """
    prompt_counts = {}
    with open(responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["prompt"]
            prompt_counts[prompt] = len(data["responses"])

    avg_count = int(sum(prompt_counts.values()) / len(prompt_counts))
    return avg_count


# ===== PHASE 2: EVALUATION EXECUTION =====

def run_evaluations(
    generation_results: Dict[str, Path],
    exp_configs: Dict[str, ExperimentConfig],
    metrics: List[str],
    output_base_dir: Path,
    num_workers: int = 16,
    skip_existing: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Run evaluations for all metrics on all experiments.

    Replicates logic from verbalized_sampling.pipeline.Pipeline.run_evaluation()

    Args:
        generation_results: Dict mapping exp name to responses.jsonl path
        exp_configs: Dict mapping exp name to ExperimentConfig
        metrics: List of metric names to evaluate
        output_base_dir: Base directory for saving evaluation results
        num_workers: Number of parallel workers
        skip_existing: Skip already-evaluated experiments

    Returns:
        Dict mapping exp name to dict of metric -> result file path
    """
    evaluation_results = {}

    print(f"\n{'='*60}")
    print(f"📊 Running Evaluations")
    print(f"   Experiments: {len(generation_results)}")
    print(f"   Metrics: {', '.join(metrics)}")
    print(f"{'='*60}\n")

    for exp_name, responses_file in generation_results.items():
        print(f"\n🔬 Evaluating: {exp_name}")
        evaluation_results[exp_name] = {}

        # Load responses
        with open(responses_file, "r") as f:
            responses = []
            prompts = []
            for line in f:
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"⚠️  Error parsing line: {e}")
                    print(line)
                    raise e

                prompt = data["prompt"]
                responses_list = data["responses"]
                for i, response in enumerate(responses_list):
                    if isinstance(response, str):
                        response = {"text": response}
                    response["index"] = i
                    responses.append(response)
                    prompts.append(prompt)

        print(f"   📝 Loaded {len(responses)} responses from {len(set(prompts))} prompts")

        # Auto-infer num_responses_per_prompt
        num_responses_per_prompt = infer_num_responses_per_prompt(responses_file)
        print(f"   ℹ️  Auto-inferred num_responses_per_prompt: {num_responses_per_prompt}")

        # Run each metric
        for metric in metrics:
            eval_dir = output_base_dir / "evaluation" / exp_name
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_file = eval_dir / f"{metric}_results.json"

            if skip_existing and eval_file.exists():
                print(f"   ⏭️  Skipping {metric} (already exists)")
                evaluation_results[exp_name][metric] = eval_file
                continue

            print(f"   📊 Evaluating metric: {metric}")

            try:
                # Create evaluator with appropriate kwargs
                if metric in ("response_count", "synthetic_data_quality", "diversity"):
                    evaluator = get_evaluator(
                        metric,
                        num_workers=num_workers,
                        num_responses_per_prompt=num_responses_per_prompt,
                    )
                else:
                    evaluator = get_evaluator(
                        metric,
                        num_workers=num_workers,
                    )

                # Prepare evaluation kwargs
                evaluation_kwargs = {"metadata": {"experiment": exp_name, "metric": metric}}

                # Special handling for accuracy metric
                if metric == "accuracy":
                    exp_config = exp_configs.get(exp_name)

                    if exp_config is None:
                        print(f"   ⚠️  Skipping accuracy for {exp_name} - no config available")
                        continue

                    # Validation: check if config matches actual responses
                    expected_total_responses = exp_config.num_responses
                    actual_total_responses = len(responses)
                    if expected_total_responses != actual_total_responses:
                        print(f"   ⚠️  Warning: Config expects {expected_total_responses} responses, "
                              f"but found {actual_total_responses} in JSONL")

                    # Load the task to get reference answers
                    task = get_task(
                        exp_config.task.value,
                        model=None,  # We don't need model for getting answers
                        method="direct",  # Method doesn't matter for getting answers
                        num_prompts=len(set(prompts)),  # Number of unique prompts
                        num_responses=1,
                        random_seed=exp_config.random_seed,
                    )

                    # Extract reference answers corresponding to the prompts
                    reference_answers = []
                    unique_prompts = []
                    seen_prompts = set()

                    # Get unique prompts in order
                    for prompt in prompts:
                        if prompt not in seen_prompts:
                            unique_prompts.append(prompt)
                            seen_prompts.add(prompt)

                    # Match prompts to task problems and extract answers
                    for prompt in unique_prompts:
                        matching_answer = None
                        for problem in task.problems:
                            # Extract the question from the formatted prompt
                            if "Question:" in prompt:
                                question_part = (
                                    prompt.split("Question:")[1]
                                    .split("Please reason")[0]
                                    .strip()
                                )
                            else:
                                question_part = prompt.strip()

                            if (
                                problem["problem"].strip() in question_part
                                or question_part in problem["problem"].strip()
                            ):
                                matching_answer = problem["answer"]
                                break

                        if matching_answer is None:
                            matching_answer = "UNKNOWN"
                        reference_answers.append(matching_answer)

                    # Expand reference answers to match all responses
                    expanded_answers = []
                    prompt_to_answer = dict(zip(unique_prompts, reference_answers))
                    for prompt in prompts:
                        expanded_answers.append(prompt_to_answer.get(prompt, "UNKNOWN"))

                    evaluation_kwargs["reference_answers"] = expanded_answers

                # Run evaluation
                result = evaluator.evaluate(prompts, responses, **evaluation_kwargs)

                # Save results
                evaluator.save_results(result, eval_file)
                evaluation_results[exp_name][metric] = eval_file

                print(f"   ✅ {metric}: Evaluation complete")

            except Exception as e:
                print(f"   ❌ {metric}: Error - {str(e)}")
                raise e

    return evaluation_results


# ===== PHASE 3: SUMMARY & REPORTING =====

def print_evaluation_summary(evaluation_results: Dict[str, Dict[str, Path]]) -> None:
    """Print simple text summary of evaluation results.

    Args:
        evaluation_results: Dict mapping exp name to dict of metric -> result file path
    """
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}\n")

    for exp_name, exp_metrics in evaluation_results.items():
        print(f"\n🔬 {exp_name}")
        print("-" * 60)

        for metric_name, result_file in exp_metrics.items():
            if result_file is None or not result_file.exists():
                print(f"  ❌ {metric_name}: No results")
                continue

            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)

                # Extract key metrics from overall_metrics
                overall = result_data.get("overall_metrics", {})

                print(f"\n  📈 {metric_name.upper()}")

                # Display key metrics (customize based on metric type)
                for key, value in overall.items():
                    # Skip complex nested structures
                    if key.endswith("_stats") or key in ("pairwise_diversities", "detailed_results"):
                        continue

                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value:.4f}")
                    else:
                        print(f"     {key}: {value}")

            except Exception as e:
                print(f"  ⚠️  {metric_name}: Could not load results - {e}")

        print()

    print(f"{'='*60}")
    print("✅ Evaluation complete!")
    print(f"{'='*60}\n")


# ===== MAIN ORCHESTRATION =====

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on generated responses from coverage pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to generation results directory (e.g., generation_results/lw-gpt-4.1-mini_state_name)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy",
        help="Comma-separated list of metrics to evaluate (default: accuracy)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for evaluation (default: 16)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already-evaluated experiments (default: True)",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        default=False,
        help="Re-run evaluation even if results exist (overrides --skip-existing)",
    )

    args = parser.parse_args()

    # Parse inputs
    input_dir = Path(args.input_dir)
    metrics = [m.strip() for m in args.metrics.split(",")]
    skip_existing = not args.clobber if args.clobber else args.skip_existing

    print(f"\n{'='*60}")
    print("🚀 LIGHTWEIGHT EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Workers: {args.num_workers}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*60}\n")

    # Phase 1: Discovery
    print("📁 Phase 1: Discovering experiments...")
    generation_results = discover_experiments(input_dir)
    print(f"   Found {len(generation_results)} experiments:")
    for exp_name in generation_results.keys():
        print(f"     • {exp_name}")

    # Load experiment configs
    print("\n📋 Loading experiment configurations...")
    exp_configs = load_experiment_configs(input_dir, list(generation_results.keys()))

    # Phase 2: Evaluation
    print("\n📊 Phase 2: Running evaluations...")
    evaluation_results = run_evaluations(
        generation_results=generation_results,
        exp_configs=exp_configs,
        metrics=metrics,
        output_base_dir=input_dir,
        num_workers=args.num_workers,
        skip_existing=skip_existing,
    )

    # Phase 3: Summary
    print_evaluation_summary(evaluation_results)

    print(f"📁 Results saved to: {input_dir}/evaluation/\n")


if __name__ == "__main__":
    main()
