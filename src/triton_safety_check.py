"""
Triton kernel safety checker and patcher
Analyzes and patches Triton kernels to prevent memory errors
"""

import re
from typing import Dict, List, Tuple, Optional


def analyze_triton_kernel(kernel_code: str) -> Dict[str, any]:
    """
    Analyze a Triton kernel for potential memory safety issues
    """
    issues = []
    warnings = []
    
    # Check for shared memory allocations
    shared_mem_pattern = r'tl\.zeros\s*\(\s*\[([^\]]+)\][^,]*,\s*dtype\s*=\s*([^,\)]+)'
    shared_allocations = []
    
    for match in re.finditer(shared_mem_pattern, kernel_code):
        dims_str = match.group(1)
        dtype = match.group(2).strip()
        
        # Parse dimensions
        dims = []
        for dim in dims_str.split(','):
            dim = dim.strip()
            # Try to evaluate if it's a simple expression
            if dim.isdigit():
                dims.append(int(dim))
            else:
                # Look for BLOCK_SIZE patterns
                if 'BLOCK' in dim:
                    # Estimate based on common values
                    dims.append(1024)  # Conservative estimate
                else:
                    dims.append(None)  # Unknown
        
        # Calculate size
        if all(d is not None for d in dims):
            total_elements = 1
            for d in dims:
                total_elements *= d
            
            # Estimate bytes based on dtype
            bytes_per_element = 4  # Default float32
            if 'float16' in dtype or 'fp16' in dtype:
                bytes_per_element = 2
            elif 'float64' in dtype or 'fp64' in dtype:
                bytes_per_element = 8
            elif 'int8' in dtype:
                bytes_per_element = 1
            elif 'int16' in dtype:
                bytes_per_element = 2
            elif 'int64' in dtype:
                bytes_per_element = 8
            
            total_bytes = total_elements * bytes_per_element
            shared_allocations.append({
                'line': match.group(0),
                'dims': dims,
                'dtype': dtype,
                'bytes': total_bytes
            })
            
            # H100 has 232KB shared memory per SM
            if total_bytes > 232 * 1024:
                issues.append(f"Shared memory allocation of {total_bytes/1024:.1f}KB exceeds H100 limit of 232KB")
            elif total_bytes > 48 * 1024:
                warnings.append(f"Large shared memory allocation: {total_bytes/1024:.1f}KB")
    
    # Check for out-of-bounds access patterns
    load_patterns = [
        r'tl\.load\s*\(([^,]+),\s*([^,\)]+)',
        r'tl\.store\s*\(([^,]+),\s*([^,\)]+)'
    ]
    
    for pattern in load_patterns:
        for match in re.finditer(pattern, kernel_code):
            ptr_expr = match.group(1)
            mask_expr = match.group(2) if match.lastindex >= 2 else None
            
            # Check if there's a mask
            if not mask_expr or 'mask' not in mask_expr.lower():
                if 'tl.load' in match.group(0):
                    warnings.append(f"tl.load without mask: {match.group(0)[:50]}... - may cause out-of-bounds access")
                else:
                    warnings.append(f"tl.store without mask: {match.group(0)[:50]}... - may cause out-of-bounds access")
    
    # Check for block size configurations
    block_size_pattern = r'@triton\.jit.*?\n.*?def\s+\w+\s*\([^)]*\):'
    autotune_pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[([^\]]+)\]'
    
    # Look for excessive block sizes
    for match in re.finditer(r'BLOCK_[A-Z]+\s*[:=]\s*(\d+)', kernel_code):
        block_size = int(match.group(1))
        if block_size > 1024:
            warnings.append(f"Large block size detected: {block_size}")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'shared_allocations': shared_allocations,
        'has_serious_issues': len(issues) > 0
    }


def patch_triton_kernel(kernel_code: str, force_mask: bool = True) -> str:
    """
    Attempt to patch common Triton kernel issues
    """
    patched_code = kernel_code
    
    if force_mask:
        # Add masks to loads without them
        def add_mask_to_load(match):
            full_match = match.group(0)
            ptr_expr = match.group(1)
            
            # Check if there's already a mask
            if match.lastindex >= 2 and 'mask' in match.group(2).lower():
                return full_match
            
            # Try to infer mask from context
            # Look for range checks before the load
            lines_before = patched_code[:match.start()].split('\n')[-5:]
            
            mask_var = None
            for line in reversed(lines_before):
                if 'mask' in line and '=' in line:
                    # Found a mask definition
                    mask_var = line.split('=')[0].strip()
                    break
            
            if mask_var:
                return f"tl.load({ptr_expr}, mask={mask_var})"
            else:
                # Add a comment warning
                return f"tl.load({ptr_expr})  # WARNING: No mask - may cause out-of-bounds access"
        
        patched_code = re.sub(
            r'tl\.load\s*\(([^,\)]+)(?:,\s*([^,\)]+))?\)',
            add_mask_to_load,
            patched_code
        )
    
    return patched_code


def suggest_fixes(analysis: Dict[str, any]) -> List[str]:
    """
    Suggest fixes based on analysis
    """
    suggestions = []
    
    if analysis['has_serious_issues']:
        suggestions.append("CRITICAL: This kernel has serious memory safety issues that may cause GPU crashes")
    
    for issue in analysis['issues']:
        if 'exceeds H100 limit' in issue:
            suggestions.append("Reduce shared memory usage by:")
            suggestions.append("  - Using smaller block sizes")
            suggestions.append("  - Processing data in smaller chunks")
            suggestions.append("  - Using registers instead of shared memory where possible")
    
    for warning in analysis['warnings']:
        if 'without mask' in warning:
            suggestions.append("Add proper masking to prevent out-of-bounds access:")
            suggestions.append("  mask = (offset + tl.arange(0, BLOCK_SIZE)) < data_size")
            suggestions.append("  value = tl.load(ptr + offset, mask=mask)")
    
    if analysis['shared_allocations']:
        total_shared = sum(alloc['bytes'] for alloc in analysis['shared_allocations'])
        suggestions.append(f"Total shared memory usage: {total_shared/1024:.1f}KB")
        if total_shared > 48 * 1024:
            suggestions.append("Consider using multiple kernel launches with smaller working sets")
    
    return suggestions


def make_kernel_safer(kernel_code: str) -> Tuple[str, List[str]]:
    """
    Analyze and attempt to make a kernel safer
    Returns: (modified_code, list_of_changes)
    """
    analysis = analyze_triton_kernel(kernel_code)
    changes = []
    
    # Only patch if there are warnings but no critical issues
    if analysis['warnings'] and not analysis['has_serious_issues']:
        patched = patch_triton_kernel(kernel_code)
        if patched != kernel_code:
            changes.append("Added safety comments to unmaked memory operations")
            kernel_code = patched
    
    # Add analysis results as comments
    if analysis['issues'] or analysis['warnings']:
        header_comments = ["# SAFETY ANALYSIS:"]
        for issue in analysis['issues']:
            header_comments.append(f"# ERROR: {issue}")
        for warning in analysis['warnings']:
            header_comments.append(f"# WARNING: {warning}")
        
        suggestions = suggest_fixes(analysis)
        if suggestions:
            header_comments.append("# SUGGESTIONS:")
            for suggestion in suggestions:
                header_comments.append(f"#   {suggestion}")
        
        # Insert after imports
        import_end = 0
        for match in re.finditer(r'^import\s+\S+|^from\s+\S+\s+import', kernel_code, re.MULTILINE):
            import_end = max(import_end, match.end())
        
        if import_end > 0:
            kernel_code = (
                kernel_code[:import_end] + 
                "\n\n" + 
                "\n".join(header_comments) + 
                "\n" + 
                kernel_code[import_end:]
            )
            changes.append("Added safety analysis comments")
    
    return kernel_code, changes


# Example usage
if __name__ == "__main__":
    sample_kernel = '''
import triton
import triton.language as tl

@triton.jit
def unsafe_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel has several issues
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Issue 1: No mask on load
    x = tl.load(x_ptr + offsets)
    
    # Issue 2: Large shared memory allocation
    shared_mem = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Issue 3: No mask on store
    tl.store(y_ptr + offsets, x)
'''
    
    analysis = analyze_triton_kernel(sample_kernel)
    print("Analysis:", analysis)
    
    safer_kernel, changes = make_kernel_safer(sample_kernel)
    print("\nChanges made:", changes)
    print("\nSafer kernel:")
    print(safer_kernel)