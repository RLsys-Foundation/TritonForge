"""Complete list of Triton operations for validation."""

# Core Triton operations based on Triton documentation
# https://triton-lang.org/main/python-api/triton.language.html
TRITON_CORE_OPS = [
    # Memory operations
    'tl.load', 'tl.store', 'tl.make_block_ptr', 'tl.advance',
    
    # Program information
    'tl.program_id', 'tl.num_programs',
    
    # Creation ops
    'tl.arange', 'tl.zeros', 'tl.ones', 'tl.full', 'tl.broadcast_to',
    'tl.cat', 'tl.expand_dims', 'tl.reshape', 'tl.view',
    
    # Arithmetic ops
    'tl.sum', 'tl.max', 'tl.min', 'tl.argmax', 'tl.argmin',
    'tl.reduce', 'tl.cumsum', 'tl.cumprod',
    
    # Math functions
    'tl.exp', 'tl.exp2', 'tl.log', 'tl.log2',
    'tl.cos', 'tl.sin', 'tl.sqrt', 'tl.rsqrt',
    'tl.sigmoid', 'tl.softmax', 'tl.abs', 'tl.floor', 'tl.ceil',
    
    # Linear algebra
    'tl.dot', 'tl.trans', 'tl.permute',
    
    # Comparison and selection
    'tl.where', 'tl.maximum', 'tl.minimum',
    
    # Utility
    'tl.cdiv', 'tl.constexpr', 'tl.static_assert',
    'tl.atomic_add', 'tl.atomic_cas', 'tl.atomic_xchg',
    'tl.atomic_max', 'tl.atomic_min', 'tl.atomic_and',
    'tl.atomic_or', 'tl.atomic_xor',
    
    # Type casting
    'tl.float32', 'tl.float16', 'tl.bfloat16',
    'tl.int32', 'tl.int16', 'tl.int8', 'tl.uint8',
    
    # Indexing
    'tl.ravel', 'tl.swizzle2d', 'tl.broadcast',
    
    # Random
    'tl.rand', 'tl.randn', 'tl.randint',
]

# Additional operations that might be used
TRITON_EXTENDED_OPS = [
    'triton.jit', '@triton.jit',  # Decorator
    'tl.', 'triton.language.',  # Namespace prefixes
]