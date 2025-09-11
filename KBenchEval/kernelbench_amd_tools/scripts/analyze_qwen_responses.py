#!/usr/bin/env python3
"""
Analyze saved Qwen3-8B responses to understand generation patterns
and identify common issues with code generation.
"""

import os
import re
import json
import glob
from typing import Dict, List, Optional
from collections import defaultdict


class QwenResponseAnalyzer:
    """Comprehensive analyzer for Qwen3-8B responses."""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.patterns = {
            'thinking': r'<think>(.*?)</think>',
            'code_block': r'```(?:python)?\n?(.*?)```',
            'triton_import': r'import\s+triton',
            'triton_jit': r'@triton\.jit',
            'model_class': r'class\s+ModelNew',
            'kernel_def': r'def\s+\w+_kernel',
            'forward_method': r'def\s+forward',
            'get_inputs': r'def\s+get_inputs',
            'get_init_inputs': r'def\s+get_init_inputs',
            'repetitive': r'(Make sure|Please ensure|You should|The code should|Don\'t forget)',
            'instruction_mode': r'(Step \d+:|First,|Second,|Finally,|Next,)',
            'error_messages': r'(error|Error|ERROR|failed|Failed|FAILED)',
            'tl_ops': r'tl\.(load|store|program_id|arange|maximum|minimum)',
            'grid_lambda': r'grid\s*=\s*lambda',
            'torch_import': r'import\s+torch',
            'nn_import': r'import\s+torch\.nn\s+as\s+nn',
        }
    
    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a single response file."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        analysis = {
            'file': filepath,
            'total_length': len(content),
            'has_thinking': False,
            'thinking_length': 0,
            'actual_response_length': 0,
            'code_blocks': [],
            'has_valid_triton': False,
            'component_scores': {},
            'issues': [],
            'pattern_matches': {}
        }
        
        # Check for thinking tags
        thinking_match = re.search(self.patterns['thinking'], content, re.DOTALL)
        if thinking_match:
            analysis['has_thinking'] = True
            analysis['thinking_length'] = len(thinking_match.group(1))
            
            # Get actual response after thinking
            parts = content.split('</think>')
            if len(parts) > 1:
                actual_response = parts[1].strip()
            else:
                actual_response = content
        else:
            actual_response = content
        
        analysis['actual_response_length'] = len(actual_response)
        
        # Extract code blocks
        code_blocks = re.findall(self.patterns['code_block'], actual_response, re.DOTALL)
        analysis['code_blocks'] = code_blocks
        
        # Analyze each code block
        best_score = 0
        best_block_idx = -1
        
        for idx, block in enumerate(code_blocks):
            score = self.score_code_block(block)
            if score > best_score:
                best_score = score
                best_block_idx = idx
        
        if best_block_idx >= 0:
            analysis['best_code_block'] = best_block_idx
            analysis['best_score'] = best_score
            analysis['has_valid_triton'] = best_score >= 6
        
        # Check for common patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern_name not in ['thinking', 'code_block']:
                matches = re.findall(pattern, actual_response, re.IGNORECASE)
                analysis['pattern_matches'][pattern_name] = len(matches)
        
        # Identify issues
        if not code_blocks:
            analysis['issues'].append('no_code_blocks')
        
        if analysis['pattern_matches'].get('repetitive', 0) > 3:
            analysis['issues'].append('repetitive_instructions')
        
        if analysis['pattern_matches'].get('instruction_mode', 0) > 2:
            analysis['issues'].append('stuck_in_instruction_mode')
        
        if not analysis['has_valid_triton']:
            analysis['issues'].append('invalid_triton_code')
        
        if analysis['has_thinking'] and analysis['actual_response_length'] < 200:
            analysis['issues'].append('short_response_after_thinking')
        
        return analysis
    
    def score_code_block(self, code: str) -> int:
        """Score a code block for Triton components."""
        score = 0
        
        # Essential components (3 points each)
        if re.search(self.patterns['triton_import'], code):
            score += 3
        if re.search(self.patterns['triton_jit'], code):
            score += 3
        if re.search(self.patterns['model_class'], code):
            score += 3
        
        # Important components (2 points each)
        if re.search(self.patterns['kernel_def'], code):
            score += 2
        if re.search(self.patterns['forward_method'], code):
            score += 2
        if re.search(self.patterns['get_inputs'], code):
            score += 2
        if re.search(self.patterns['get_init_inputs'], code):
            score += 2
        
        # Supporting components (1 point each)
        if re.search(self.patterns['torch_import'], code):
            score += 1
        if re.search(self.patterns['nn_import'], code):
            score += 1
        if re.search(self.patterns['tl_ops'], code):
            score += 1
        if re.search(self.patterns['grid_lambda'], code):
            score += 1
        
        return score
    
    def analyze_directory(self, directory: str) -> Dict:
        """Analyze all response files in a directory."""
        response_files = glob.glob(os.path.join(directory, "*.txt"))
        response_files.extend(glob.glob(os.path.join(directory, "*.json")))
        
        all_analyses = []
        summary_stats = defaultdict(int)
        issue_counts = defaultdict(int)
        
        for filepath in response_files:
            try:
                if filepath.endswith('.json'):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'response' in data:
                            # Save response to temp file for analysis
                            temp_file = filepath.replace('.json', '_response.txt')
                            with open(temp_file, 'w') as tf:
                                tf.write(data['response'])
                            analysis = self.analyze_file(temp_file)
                            os.remove(temp_file)
                        else:
                            continue
                else:
                    analysis = self.analyze_file(filepath)
                
                all_analyses.append(analysis)
                
                # Update summary stats
                summary_stats['total_files'] += 1
                if analysis['has_thinking']:
                    summary_stats['with_thinking'] += 1
                if analysis['has_valid_triton']:
                    summary_stats['valid_triton'] += 1
                if analysis['code_blocks']:
                    summary_stats['has_code'] += 1
                
                # Count issues
                for issue in analysis['issues']:
                    issue_counts[issue] += 1
                
            except Exception as e:
                print(f"Error analyzing {filepath}: {e}")
        
        return {
            'directory': directory,
            'analyses': all_analyses,
            'summary': dict(summary_stats),
            'issue_counts': dict(issue_counts)
        }
    
    def print_summary(self, results: Dict):
        """Print analysis summary."""
        print("="*70)
        print("QWEN3-8B RESPONSE ANALYSIS SUMMARY")
        print("="*70)
        print()
        
        summary = results['summary']
        total = summary.get('total_files', 0)
        
        if total == 0:
            print("No response files found")
            return
        
        print(f"Total files analyzed: {total}")
        print()
        
        print("Response Characteristics:")
        print(f"  With thinking tags: {summary.get('with_thinking', 0)}/{total} "
              f"({100*summary.get('with_thinking', 0)/total:.1f}%)")
        print(f"  Has code blocks: {summary.get('has_code', 0)}/{total} "
              f"({100*summary.get('has_code', 0)/total:.1f}%)")
        print(f"  Valid Triton code: {summary.get('valid_triton', 0)}/{total} "
              f"({100*summary.get('valid_triton', 0)/total:.1f}%)")
        print()
        
        if results['issue_counts']:
            print("Common Issues:")
            for issue, count in sorted(results['issue_counts'].items(), 
                                      key=lambda x: x[1], reverse=True):
                issue_desc = {
                    'no_code_blocks': 'No code blocks found',
                    'repetitive_instructions': 'Repetitive instruction text',
                    'stuck_in_instruction_mode': 'Stuck in instruction mode',
                    'invalid_triton_code': 'Invalid or incomplete Triton code',
                    'short_response_after_thinking': 'Short response after thinking'
                }.get(issue, issue)
                print(f"  {issue_desc}: {count}/{total} ({100*count/total:.1f}%)")
        print()
        
        # Detailed file analysis
        print("File-by-File Analysis:")
        print("-"*70)
        
        for analysis in results['analyses']:
            filename = os.path.basename(analysis['file'])
            status = "✓" if analysis['has_valid_triton'] else "✗"
            score = analysis.get('best_score', 0)
            
            print(f"{status} {filename}")
            print(f"   Score: {score}/21")
            if analysis['issues']:
                print(f"   Issues: {', '.join(analysis['issues'])}")
            print()


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Qwen3-8B responses')
    parser.add_argument('--dir', type=str, 
                       default='/workspace/KernelBench/runs',
                       help='Directory containing response files')
    parser.add_argument('--file', type=str, default=None,
                       help='Analyze a single file')
    
    args = parser.parse_args()
    
    analyzer = QwenResponseAnalyzer()
    
    if args.file:
        # Analyze single file
        analysis = analyzer.analyze_file(args.file)
        
        print("="*70)
        print(f"ANALYSIS: {os.path.basename(args.file)}")
        print("="*70)
        print()
        
        print(f"Total length: {analysis['total_length']} chars")
        print(f"Has thinking: {analysis['has_thinking']}")
        if analysis['has_thinking']:
            print(f"  Thinking length: {analysis['thinking_length']} chars")
            print(f"  Actual response: {analysis['actual_response_length']} chars")
        
        print(f"\nCode blocks: {len(analysis['code_blocks'])}")
        if analysis.get('best_score'):
            print(f"Best block score: {analysis['best_score']}/21")
            print(f"Valid Triton: {analysis['has_valid_triton']}")
        
        if analysis['pattern_matches']:
            print("\nPattern Matches:")
            for pattern, count in analysis['pattern_matches'].items():
                if count > 0:
                    print(f"  {pattern}: {count}")
        
        if analysis['issues']:
            print(f"\nIssues: {', '.join(analysis['issues'])}")
    
    else:
        # Analyze directory
        results = analyzer.analyze_directory(args.dir)
        analyzer.print_summary(results)
        
        # Save detailed results
        output_file = os.path.join(args.dir, 'response_analysis.json')
        with open(output_file, 'w') as f:
            # Convert analyses to be JSON serializable
            json_results = {
                'directory': results['directory'],
                'summary': results['summary'],
                'issue_counts': results['issue_counts'],
                'file_summaries': [
                    {
                        'file': os.path.basename(a['file']),
                        'has_valid_triton': a['has_valid_triton'],
                        'best_score': a.get('best_score', 0),
                        'issues': a['issues']
                    }
                    for a in results['analyses']
                ]
            }
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()