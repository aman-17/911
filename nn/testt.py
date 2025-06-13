from nn.rope import YaRNRotaryEmbedding
import torch
from nn.attention.multihead_latent_attention import precompute_freqs_cis, apply_rotary_emb


def create_test_tensors(batch_size=2, seq_len=128, num_heads=8, head_dim=64, device='cpu'):
    """Create test tensors for comparison."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    return q, k

def test_frequency_computation():
    """Test frequency computation between both implementations."""
    print("=" * 60)
    print("TESTING FREQUENCY COMPUTATION")
    print("=" * 60)
    
    # Parameters
    dim = 64
    seq_len = 128
    max_seq_len = 4096
    theta = 10000.0
    rope_factor = 1.0
    beta_fast = 32
    beta_slow = 1
    device = torch.device('cpu')
    
    # Class-based implementation
    yarn_rope = YaRNRotaryEmbedding(
        head_size=dim,
        theta=theta,
        max_seq_len=max_seq_len,
        rope_factor=rope_factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow
    )
    freqs_cis_class = yarn_rope._get_rotary_embedding(seq_len, device)
    
    # Functional implementation
    freqs_cis_func = precompute_freqs_cis(
        dim=dim,
        max_seq_len=max_seq_len,
        seq_len=seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        theta=theta,
        rope_factor=rope_factor,
    )
    
    print(f"Class-based freqs_cis shape: {freqs_cis_class.shape}")
    print(f"Functional freqs_cis shape: {freqs_cis_func.shape}")
    print(f"Are frequencies close? {torch.allclose(freqs_cis_class, freqs_cis_func, atol=1e-6)}")
    print(f"Max difference: {torch.max(torch.abs(freqs_cis_class - freqs_cis_func)).item()}")
    
    return freqs_cis_class, freqs_cis_func

def test_apply_rotary_emb():
    """Test apply_rotary_emb function between both implementations."""
    print("\n" + "=" * 60)
    print("TESTING APPLY_ROTARY_EMB")
    print("=" * 60)
    
    # Create test data
    batch_size, seq_len, num_heads, head_dim = 2, 32, 8, 64
    device = torch.device('cpu')
    
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Get frequencies
    yarn_rope = YaRNRotaryEmbedding(head_size=head_dim)
    freqs_cis = yarn_rope._get_rotary_embedding(seq_len, device)
    
    # Convert x to complex form for _apply_rotary_pos_emb
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))
    
    # Reshape freqs_cis for broadcasting: [seq_len, head_dim//2] -> [1, seq_len, 1, head_dim//2]
    freqs_cis_expanded = freqs_cis[None, :, None, :]
    
    # Apply rotary embeddings using class method
    x_rotated_class = yarn_rope._apply_rotary_pos_emb(freqs_cis_expanded, x_complex)
    
    # Apply rotary embeddings using functional approach
    x_rotated_func = apply_rotary_emb(x, freqs_cis)
    
    print(f"Input shape: {x.shape}")
    print(f"Complex input shape: {x_complex.shape}")
    print(f"Class-based output shape: {x_rotated_class.shape}")
    print(f"Functional output shape: {x_rotated_func.shape}")
    print(f"Are outputs close? {torch.allclose(x_rotated_class, x_rotated_func, atol=1e-6)}")
    print(f"Max difference: {torch.max(torch.abs(x_rotated_class - x_rotated_func)).item()}")

def test_full_forward_pass():
    """Test full forward pass with query and key tensors."""
    print("\n" + "=" * 60)
    print("TESTING FULL FORWARD PASS")
    print("=" * 60)
    
    # Parameters
    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64
    device = torch.device('cpu')
    
    # Create test tensors
    q, k = create_test_tensors(batch_size, seq_len, num_heads, head_dim, device)
    
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    
    # Class-based implementation
    yarn_rope = YaRNRotaryEmbedding(head_size=head_dim)
    q_rot_class, k_rot_class = yarn_rope.forward(q, k, head_first=True)
    
    print(f"Class-based Q output shape: {q_rot_class.shape}")
    print(f"Class-based K output shape: {k_rot_class.shape}")
    
    # For functional approach, we need to manually apply the same logic
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        max_seq_len=4096,
        seq_len=seq_len,
        beta_fast=32,
        beta_slow=1,
        device=device
    )
    
    # Convert to complex and apply rotation (simplified version)
    q_float = q.float()
    k_float = k.float()
    
    q_complex = torch.view_as_complex(q_float.reshape(*q_float.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k_float.reshape(*k_float.shape[:-1], -1, 2))
    
    # Apply rotation
    freqs_cis_expanded = freqs_cis[None, None, :, :]
    q_rot_func = q_complex * freqs_cis_expanded
    k_rot_func = k_complex * freqs_cis_expanded
    
    print(f"Functional Q output shape: {q_rot_func.shape}")
    print(f"Functional K output shape: {k_rot_func.shape}")
    
    # Note: Direct comparison is complex due to implementation differences
    print("\nNote: Direct comparison of forward pass requires more complex setup")
    print("due to implementation differences in the forward method.")

def test_long_sequence_behavior():
    """Test behavior with sequences longer than max_seq_len to trigger YaRN corrections."""
    print("\n" + "=" * 60)
    print("TESTING LONG SEQUENCE BEHAVIOR (YaRN CORRECTIONS)")
    print("=" * 60)
    
    # Parameters that will trigger YaRN corrections
    dim = 64
    seq_len = 8192  # Longer than max_seq_len
    max_seq_len = 4096
    device = torch.device('cpu')
    
    # Class-based implementation
    yarn_rope = YaRNRotaryEmbedding(
        head_size=dim,
        max_seq_len=max_seq_len,
        rope_factor=4.0,  # Non-default value to see corrections
        beta_fast=32,
        beta_slow=1
    )
    freqs_cis_class = yarn_rope._get_rotary_embedding(seq_len, device)
    
    # Functional implementation
    freqs_cis_func = precompute_freqs_cis(
        dim=dim,
        max_seq_len=max_seq_len,
        seq_len=seq_len,
        beta_fast=32,
        beta_slow=1,
        rope_factor=4.0,
        device=device
    )
    
    print(f"Long sequence length: {seq_len}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Class-based freqs_cis shape: {freqs_cis_class.shape}")
    print(f"Functional freqs_cis shape: {freqs_cis_func.shape}")
    print(f"Are frequencies close? {torch.allclose(freqs_cis_class, freqs_cis_func, atol=1e-5)}")
    print(f"Max difference: {torch.max(torch.abs(freqs_cis_class - freqs_cis_func)).item()}")

def run_all_tests():
    """Run all comparison tests."""
    print("🔍 YARN ROPE IMPLEMENTATION COMPARISON TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Frequency computation
        test_frequency_computation()
        
        # Test 2: Apply rotary embeddings
        test_apply_rotary_emb()
        
        # Test 3: Full forward pass
        test_full_forward_pass()
        
        # Test 4: Long sequence behavior
        test_long_sequence_behavior()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()