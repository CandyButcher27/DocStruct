"""
Report generation for the DocStruct benchmarking framework.
Generates ASCII and LaTeX tables from CSV results.
"""

import argparse
import csv
from typing import List, Dict

def generate_report(csv_path: str) -> None:
    """Generate and print a report from a benchmark CSV file."""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        print(f"No results found in {csv_path}")
        return

    print(f"\nBenchmark Report: {csv_path}")
    print("-" * 80)
    print(f"{'Doc ID':<10} | {'Mode':<10} | {'mAP@0.5':<10} | {'mAP@0.75':<10} | {'Text F1':<10} | {'Table F1':<10}")
    print("-" * 80)
    
    for r in results:
        doc_id = r.get("doc_id", "")[:8]
        mode = r.get("mode", "")[:10]
        map50 = r.get("mAP@0.50", "")
        if map50: map50 = f"{float(map50):.3f}"
        map75 = r.get("mAP@0.75", "")
        if map75: map75 = f"{float(map75):.3f}"
        
        text_f1 = r.get("text_f1", "")
        if text_f1: text_f1 = f"{float(text_f1):.3f}"
            
        table_f1 = r.get("table_f1", "")
        if table_f1: table_f1 = f"{float(table_f1):.3f}"
        
        print(f"{doc_id:<10} | {mode:<10} | {map50:<10} | {map75:<10} | {text_f1:<10} | {table_f1:<10}")
    
    print("-" * 80)
    print("\nLaTeX Table Summary:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l c c c c}")
    print("\\toprule")
    print("Mode & mAP@0.50 & mAP@0.75 & Text F1 & Table F1 \\\\")
    print("\\midrule")
    
    # Simple average for LaTeX table
    modes = set(r["mode"] for r in results)
    for m in modes:
        mode_results = [r for r in results if r["mode"] == m]
        avg_map50 = sum(float(r["mAP@0.50"] or 0) for r in mode_results) / len(mode_results)
        avg_map75 = sum(float(r["mAP@0.75"] or 0) for r in mode_results) / len(mode_results)
        avg_text = sum(float(r["text_f1"] or 0) for r in mode_results) / len(mode_results)
        avg_table = sum(float(r["table_f1"] or 0) for r in mode_results) / len(mode_results)
        
        print(f"{m.capitalize()} & {avg_map50:.3f} & {avg_map75:.3f} & {avg_text:.3f} & {avg_table:.3f} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Benchmarking Results}")
    print("\\end{table}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("csv_path", help="Path to the benchmark results CSV")
    args = parser.parse_args()
    generate_report(args.csv_path)
