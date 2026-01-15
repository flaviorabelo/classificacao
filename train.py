from __future__ import annotations

from pathlib import Path
from textwrap import indent

import argparse
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from evasao.config import DATASET_PATH
from evasao.training import train_pipeline


def format_evaluation(result):
    header = f"Modelo: {result.model_name}\nAcurácia: {result.accuracy:.2%}\n"
    report = indent(result.report, prefix="    ")
    return header + "Relatório:\n" + report


def main():
    parser = argparse.ArgumentParser(description="Treinamento do modelo de evasão escolar")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATASET_PATH),
        help="Caminho para o arquivo CSV de treinamento (padrão: dataset do repositório)",
    )

    args = parser.parse_args()

    outcome = train_pipeline(dataset_path=args.data)

    print("Resumo da validação:")
    for evaluation in outcome.evaluations:
        print("-" * 40)
        print(format_evaluation(evaluation))

    print("Vencedor na validação:", outcome.winner_name)
    print(f"Acurácia baseline na validação: {outcome.baseline_accuracy:.2%}")

    print("\nAvaliação final no conjunto de teste:")
    print(format_evaluation(outcome.test_evaluation))


if __name__ == "__main__":
    main()
