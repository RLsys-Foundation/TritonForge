#!/usr/bin/env python3
"""
AMD-specific issue handler for KernelBench evaluation
Analyzes failures and provides fixes or retries with modified settings
"""

import json
import os
import re
from pathlib import Path

class AMDIssueHandler:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.progress_file = f"{run_dir}/progress.json"
        self.results_file = f"{run_dir}/results.json"
        self.fixes_file = f"{run_dir}/amd_fixes.json"
        
        # Load data
        self.progress = self.load_json(self.progress_file)
        self.results = self.load_json(self.results_file)
        
        # AMD-specific issue patterns and fixes
        self.issue_patterns = {
            "OOM": {
                "pattern": r"HIP out of memory|Tried to allocate|OutOfMemoryError",
                "fix": "reduce_memory_allocation",
                "description": "Memory allocation issue"
            },
            "TYPE_ERROR": {
                "pattern": r"expected Tensor|TypeError.*Tensor",
                "fix": "fix_tensor_types",
                "description": "Tensor type mismatch"
            },
            "SCALAR_ISSUE": {
                "pattern": r"expected Tensor\(\)|scalar",
                "fix": "handle_scalar_inputs",
                "description": "Scalar input handling issue"
            },
            "BLOCK_SIZE": {
                "pattern": r"BLOCK_SIZE|block size|Invalid configuration",
                "fix": "adjust_block_size",
                "description": "Block size configuration issue"
            },
            "COMPILATION": {
                "pattern": r"compilation failed|JIT compilation|triton\.compiler",
                "fix": "compilation_workaround",
                "description": "Triton compilation error"
            },
            "KERNEL_LAUNCH": {
                "pattern": r"kernel launch|grid|Invalid grid",
                "fix": "fix_kernel_launch",
                "description": "Kernel launch configuration issue"
            }
        }
        
        self.fixes_applied = []
    
    def load_json(self, filepath):
        """Load JSON file safely"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_json(self, data, filepath):
        """Save JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def analyze_failures(self):
        """Analyze all failures and categorize by issue type"""
        categorized = {}
        
        for failure in self.progress.get("failed", []):
            problem = failure.get("problem", "Unknown")
            error = failure.get("error", "")
            
            # Try to categorize the error
            issue_type = "UNKNOWN"
            for issue_name, issue_info in self.issue_patterns.items():
                if re.search(issue_info["pattern"], error, re.IGNORECASE):
                    issue_type = issue_name
                    break
            
            if issue_type not in categorized:
                categorized[issue_type] = []
            
            categorized[issue_type].append({
                "problem": problem,
                "error": error,
                "fix": self.issue_patterns.get(issue_type, {}).get("fix", "none")
            })
        
        return categorized
    
    def generate_fixes_report(self):
        """Generate a report of issues and suggested fixes"""
        categorized = self.analyze_failures()
        
        report = []
        report.append("# AMD-Specific Issues Analysis\n")
        report.append(f"Total failures: {len(self.progress.get('failed', []))}\n")
        
        for issue_type, failures in categorized.items():
            if issue_type in self.issue_patterns:
                desc = self.issue_patterns[issue_type]["description"]
                fix = self.issue_patterns[issue_type]["fix"]
            else:
                desc = "Unknown issue"
                fix = "Manual investigation needed"
            
            report.append(f"\n## {issue_type} ({len(failures)} occurrences)")
            report.append(f"Description: {desc}")
            report.append(f"Suggested fix: {fix}\n")
            
            # Show first few examples
            for failure in failures[:3]:
                problem = failure["problem"]
                error_preview = failure["error"][:100] + "..." if len(failure["error"]) > 100 else failure["error"]
                report.append(f"- {problem}: {error_preview}")
            
            if len(failures) > 3:
                report.append(f"- ... and {len(failures) - 3} more")
        
        # Write report
        report_path = f"{self.run_dir}/amd_issues_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        print(f"AMD issues report saved to: {report_path}")
        return categorized
    
    def suggest_environment_fixes(self):
        """Suggest environment variable changes for AMD"""
        suggestions = []
        
        categorized = self.analyze_failures()
        
        # Memory issues
        if "OOM" in categorized and len(categorized["OOM"]) > 5:
            suggestions.append({
                "issue": "Frequent OOM errors",
                "fix": "Consider reducing batch sizes or using torch.cuda.empty_cache()",
                "description": "Free up cached memory between evaluations"
            })
        
        # Compilation issues
        if "COMPILATION" in categorized:
            suggestions.append({
                "issue": "Triton compilation errors",
                "fix": "export TRITON_AMD_ENABLE_EXPERIMENTAL=1",
                "description": "Enable experimental AMD features in Triton"
            })
        
        # Block size issues
        if "BLOCK_SIZE" in categorized:
            suggestions.append({
                "issue": "Block size configuration",
                "fix": "export TRITON_AMD_BLOCK_SIZE=64",
                "description": "Set default block size for AMD wavefront (64)"
            })
        
        return suggestions
    
    def generate_retry_list(self):
        """Generate a list of problems to retry with fixes"""
        retry_list = []
        categorized = self.analyze_failures()
        
        # Prioritize fixable issues
        fixable_types = ["TYPE_ERROR", "SCALAR_ISSUE", "BLOCK_SIZE"]
        
        for issue_type in fixable_types:
            if issue_type in categorized:
                for failure in categorized[issue_type]:
                    problem = failure["problem"]
                    level = int(problem.split("level")[1].split("_")[0])
                    problem_id = int(problem.split("problem")[1])
                    
                    retry_list.append({
                        "level": level,
                        "problem_id": problem_id,
                        "issue_type": issue_type,
                        "fix": self.issue_patterns[issue_type]["fix"]
                    })
        
        # Save retry list
        retry_file = f"{self.run_dir}/retry_list.json"
        self.save_json(retry_list, retry_file)
        
        print(f"Generated retry list with {len(retry_list)} problems")
        print(f"Saved to: {retry_file}")
        
        return retry_list
    
    def analyze_successful_patterns(self):
        """Analyze patterns in successful kernels"""
        successful = []
        
        if "levels" in self.results:
            for level, problems in self.results["levels"].items():
                for problem_id, result in problems.items():
                    if result.get("correctness"):
                        successful.append({
                            "level": level,
                            "problem": problem_id,
                            "runtime": result.get("runtime", -1),
                            "metadata": result.get("metadata", {})
                        })
        
        print(f"\nSuccessful Kernels Analysis:")
        print(f"Total successful: {len(successful)}")
        
        if successful:
            # Performance statistics
            runtimes = [s["runtime"] for s in successful if s["runtime"] > 0]
            if runtimes:
                print(f"Average runtime: {sum(runtimes)/len(runtimes):.3f}ms")
                print(f"Best runtime: {min(runtimes):.3f}ms")
                print(f"Worst runtime: {max(runtimes):.3f}ms")
        
        return successful
    
    def generate_full_analysis(self):
        """Generate comprehensive AMD analysis"""
        print("Analyzing AMD-specific issues...\n")
        
        # 1. Categorize failures
        categorized = self.analyze_failures()
        print(f"Found {len(self.progress.get('failed', []))} total failures")
        
        # 2. Generate issues report
        self.generate_fixes_report()
        
        # 3. Suggest environment fixes
        env_fixes = self.suggest_environment_fixes()
        if env_fixes:
            print("\nSuggested Environment Variables:")
            for fix in env_fixes:
                print(f"- {fix['issue']}")
                print(f"  {fix['fix']}")
                print(f"  {fix['description']}\n")
        
        # 4. Generate retry list
        retry_list = self.generate_retry_list()
        
        # 5. Analyze successful patterns
        self.analyze_successful_patterns()
        
        # 6. Save comprehensive analysis
        analysis = {
            "timestamp": os.path.getmtime(self.progress_file),
            "total_failures": len(self.progress.get("failed", [])),
            "categorized_issues": {k: len(v) for k, v in categorized.items()},
            "environment_suggestions": env_fixes,
            "retry_candidates": len(retry_list)
        }
        
        analysis_file = f"{self.run_dir}/amd_analysis.json"
        self.save_json(analysis, analysis_file)
        
        print(f"\nComplete analysis saved to: {analysis_file}")

def main():
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Find latest run
        runs_dir = "/workspace/KernelBench/runs"
        if os.path.exists(runs_dir):
            run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("amd_mi300x_full_eval_")]
            if run_dirs:
                run_dirs.sort()
                run_dir = os.path.join(runs_dir, run_dirs[-1])
            else:
                print("No evaluation runs found.")
                return
        else:
            print("Runs directory not found.")
            return
    
    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return
    
    handler = AMDIssueHandler(run_dir)
    handler.generate_full_analysis()

if __name__ == "__main__":
    main()