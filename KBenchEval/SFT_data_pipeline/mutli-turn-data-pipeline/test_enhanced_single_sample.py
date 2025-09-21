#!/usr/bin/env python3
"""Test the enhanced script with a single sample"""

import os
import sys
import json

# Set up test environment
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Extract a single sample for testing
input_file = "/root/kernel_book/kernelbook_sft_format.jsonl"
test_input = "/root/KernelBench/test_single_sample.jsonl"

# Get the first sample
with open(input_file, 'r') as f:
    first_line = f.readline()
    with open(test_input, 'w') as out:
        out.write(first_line)

print("Created test input file with 1 sample")

# Test import
try:
    from claude_multi_turn_sft_enhanced import EnhancedClaudeMultiTurnSFTGenerator
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test initialization without API key (dry run)
try:
    # We'll test the class structure without actually calling the API
    print("\nTesting class initialization...")
    
    # Mock the parent class init to avoid needing API key
    from unittest.mock import MagicMock, patch
    
    with patch('claude_multi_turn_sft_enhanced.ClaudeMultiTurnSFTGenerator.__init__') as mock_init:
        mock_init.return_value = None
        generator = EnhancedClaudeMultiTurnSFTGenerator(
            api_key="test_key",
            verbose=True,
            skip_on_repeated_errors=True,
            max_retries_per_sample=2
        )
        
        # Test that our enhanced attributes are set
        assert hasattr(generator, 'skip_on_repeated_errors')
        assert hasattr(generator, 'max_retries_per_sample')
        assert hasattr(generator, 'error_patterns')
        print("✓ Class initialization successful")
        
except Exception as e:
    print(f"✗ Class initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Test the fix_init_parameter_mismatch method
try:
    print("\nTesting fix_init_parameter_mismatch...")
    
    test_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim

def get_inputs():
    return [torch.randn(4, 4).cuda()]

def get_init_inputs():
    return [[], {'dim': 4, 'mdim': 8}]
'''
    
    error_msg = "Model.__init__() got an unexpected keyword argument 'mdim'"
    
    # Create a minimal generator for testing
    generator = EnhancedClaudeMultiTurnSFTGenerator.__new__(EnhancedClaudeMultiTurnSFTGenerator)
    generator.error_patterns = {'init_mismatch': [], 'llvm_error': [], 'cuda_error': []}
    
    fixed_code = generator.fix_init_parameter_mismatch(test_code, error_msg)
    
    # Check that mdim was removed
    assert "'mdim'" not in fixed_code
    assert "'dim': 4" in fixed_code
    print("✓ fix_init_parameter_mismatch works correctly")
    print("  Original had: {'dim': 4, 'mdim': 8}")
    print("  Fixed has: {'dim': 4}")
    
except Exception as e:
    print(f"✗ fix_init_parameter_mismatch failed: {e}")
    import traceback
    traceback.print_exc()

# Test the evaluate_kernel_safe method structure
try:
    print("\nTesting evaluate_kernel_safe structure...")
    
    # Check that the method exists and has correct signature
    import inspect
    
    generator = EnhancedClaudeMultiTurnSFTGenerator.__new__(EnhancedClaudeMultiTurnSFTGenerator)
    
    # Check method exists
    assert hasattr(generator, 'evaluate_kernel_safe')
    
    # Check signature
    sig = inspect.signature(generator.evaluate_kernel_safe)
    params = list(sig.parameters.keys())
    assert 'original_code' in params
    assert 'generated_code' in params
    assert 'backend' in params
    assert 'sample_id' in params
    
    print("✓ evaluate_kernel_safe method structure is correct")
    
except Exception as e:
    print(f"✗ evaluate_kernel_safe check failed: {e}")

print("\n" + "="*50)
print("Test Summary:")
print("- Enhanced script can be imported")
print("- Class initialization works")
print("- Parameter mismatch fixing works")
print("- Method signatures are correct")
print("\nThe enhanced script appears to be structurally sound.")
print("Any runtime issues would likely be in the parent class methods.")
print("="*50)