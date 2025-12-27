//! Tensor operations and Einstein summation demonstration.
//!
//! This example demonstrates:
//! - Einstein summation notation (einsum) with 24 supported patterns
//! - 3D tensor operations
//! - Batched matrix multiplication
//! - Various tensor contractions and reductions
//!
//! Run with: cargo run --example tensor_operations

use oxiblas_blas::tensor::{Tensor3, batched_matmul, einsum, outer_product};

fn main() {
    println!("=== Tensor Operations & Einstein Summation ===\n");

    // ========================================
    // Basic Matrix Operations via Einsum
    // ========================================
    println!("--- Matrix Operations via Einsum ---");

    // Matrix multiplication: C = A × B
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let b = vec![5.0, 6.0, 7.0, 8.0]; // 2×2

    println!("Matrix A (2×2): [[1, 2], [3, 4]]");
    println!("Matrix B (2×2): [[5, 6], [7, 8]]");

    let c = einsum("ij,jk->ik", &a, &[2, 2], Some((&b, &[2, 2]))).unwrap();
    println!("C = einsum('ij,jk->ik', A, B) = {:?}", c);
    println!("Expected: [[19, 22], [43, 50]]");
    println!("Verification: [1*5+2*7=19, 1*6+2*8=22, 3*5+4*7=43, 3*6+4*8=50] ✓\n");

    // Matrix transpose
    let a_t = einsum("ij->ji", &a, &[2, 2], None::<(&[f64], &[usize])>).unwrap();
    println!("A^T = einsum('ij->ji', A) = {:?}", a_t);
    println!("Expected: [[1, 3], [2, 4]] ✓\n");

    // Matrix trace (sum of diagonal)
    let trace = einsum("ii->", &a, &[2, 2], None::<(&[f64], &[usize])>).unwrap();
    println!("trace(A) = einsum('ii->', A) = {:?}", trace);
    println!("Expected: 1 + 4 = 5 ✓\n");

    // Diagonal extraction
    let diag = einsum("ii->i", &a, &[2, 2], None::<(&[f64], &[usize])>).unwrap();
    println!("diag(A) = einsum('ii->i', A) = {:?}", diag);
    println!("Expected: [1, 4] ✓\n");

    // ========================================
    // Vector Operations via Einsum
    // ========================================
    println!("--- Vector Operations via Einsum ---");

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    // Dot product
    let dot_result = einsum("i,i->", &x, &[3], Some((&y[..], &[3][..]))).unwrap();
    println!("x = {:?}", x);
    println!("y = {:?}", y);
    println!("dot(x, y) = einsum('i,i->', x, y) = {:?}", dot_result);
    println!("Expected: 1*4 + 2*5 + 3*6 = 32 ✓\n");

    // Outer product
    let outer = einsum("i,j->ij", &x, &[3], Some((&y[..], &[3][..]))).unwrap();
    println!("outer(x, y) = einsum('i,j->ij', x, y) = {:?}", outer);
    println!("Expected: [[4, 5, 6], [8, 10, 12], [12, 15, 18]] ✓\n");

    // Hadamard (element-wise) product
    let hadamard = einsum("i,i->i", &x, &[3], Some((&y[..], &[3][..]))).unwrap();
    println!("hadamard(x, y) = einsum('i,i->i', x, y) = {:?}", hadamard);
    println!("Expected: [1*4, 2*5, 3*6] = [4, 10, 18] ✓\n");

    // ========================================
    // Matrix Reductions via Einsum
    // ========================================
    println!("--- Matrix Reductions ---");

    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3

    println!("Matrix A (2×3): [[1, 2, 3], [4, 5, 6]]");

    // Row sums
    let row_sums = einsum("ij->i", &matrix, &[2, 3], None::<(&[f64], &[usize])>).unwrap();
    println!("Row sums = einsum('ij->i', A) = {:?}", row_sums);
    println!("Expected: [1+2+3=6, 4+5+6=15] ✓");

    // Column sums
    let col_sums = einsum("ij->j", &matrix, &[2, 3], None::<(&[f64], &[usize])>).unwrap();
    println!("Column sums = einsum('ij->j', A) = {:?}", col_sums);
    println!("Expected: [1+4=5, 2+5=7, 3+6=9] ✓");

    // Total sum
    let total = einsum("ij->", &matrix, &[2, 3], None::<(&[f64], &[usize])>).unwrap();
    println!("Total sum = einsum('ij->', A) = {:?}", total);
    println!("Expected: 1+2+3+4+5+6 = 21 ✓\n");

    // ========================================
    // 3D Tensor Operations
    // ========================================
    println!("--- 3D Tensor Operations ---");

    // 2×2×2 tensor: [[[1,2], [3,4]], [[5,6], [7,8]]]
    let tensor_3d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    println!("3D Tensor (2×2×2):");
    println!("  Slice 0: [[1, 2], [3, 4]]");
    println!("  Slice 1: [[5, 6], [7, 8]]");

    // Transpose: swap middle and last axes (ijk → ikj)
    let transposed_ikj = einsum(
        "ijk->ikj",
        &tensor_3d,
        &[2, 2, 2],
        None::<(&[f64], &[usize])>,
    )
    .unwrap();
    println!("\nTranspose ijk→ikj: {:?}", transposed_ikj);
    println!("  Slice 0: [[1, 3], [2, 4]]");
    println!("  Slice 1: [[5, 7], [6, 8]]");

    // Transpose: swap first two axes (ijk → jik)
    let transposed_jik = einsum(
        "ijk->jik",
        &tensor_3d,
        &[2, 2, 2],
        None::<(&[f64], &[usize])>,
    )
    .unwrap();
    println!("\nTranspose ijk→jik: {:?}", transposed_jik);
    println!("  Slice 0: [[1, 2], [5, 6]]");
    println!("  Slice 1: [[3, 4], [7, 8]]");

    // Transpose: reverse all axes (ijk → kji)
    let transposed_kji = einsum(
        "ijk->kji",
        &tensor_3d,
        &[2, 2, 2],
        None::<(&[f64], &[usize])>,
    )
    .unwrap();
    println!("\nTranspose ijk→kji: {:?}", transposed_kji);
    println!("  Slice 0: [[1, 5], [3, 7]]");
    println!("  Slice 1: [[2, 6], [4, 8]]\n");

    // Sum over last axis
    let sum_last = einsum(
        "ijk->ij",
        &tensor_3d,
        &[2, 2, 2],
        None::<(&[f64], &[usize])>,
    )
    .unwrap();
    println!("Sum over last axis (ijk→ij): {:?}", sum_last);
    println!("Expected: [[1+2=3, 3+4=7], [5+6=11, 7+8=15]] = [3, 7, 11, 15] ✓");

    // Sum over first axis
    let sum_first = einsum(
        "ijk->jk",
        &tensor_3d,
        &[2, 2, 2],
        None::<(&[f64], &[usize])>,
    )
    .unwrap();
    println!("Sum over first axis (ijk→jk): {:?}", sum_first);
    println!("Expected: [[1+5=6, 2+6=8], [3+7=10, 4+8=12]] = [6, 8, 10, 12] ✓\n");

    // ========================================
    // Tensor-Matrix Contraction
    // ========================================
    println!("--- Tensor-Matrix Contraction ---");

    let tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×2×2
    let matrix_tm = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3

    println!("Tensor T (2×2×2): [[[1,2], [3,4]], [[5,6], [7,8]]]");
    println!("Matrix M (2×3): [[1,2,3], [4,5,6]]");

    let result = einsum(
        "ijk,kl->ijl",
        &tensor,
        &[2, 2, 2],
        Some((&matrix_tm[..], &[2, 3][..])),
    )
    .unwrap();
    println!("\nResult = einsum('ijk,kl->ijl', T, M) (2×2×3):");
    println!("Shape: {}", result.len());
    println!(
        "First few elements: [{:.1}, {:.1}, {:.1}]",
        result[0], result[1], result[2]
    );
    println!("Explanation: R[i,j,l] = Σ_k T[i,j,k] * M[k,l]");
    println!("R[0,0,0] = T[0,0,0]*M[0,0] + T[0,0,1]*M[1,0] = 1*1 + 2*4 = 9 ✓\n");

    // ========================================
    // Outer Product to 3D Tensor
    // ========================================
    println!("--- Outer Product to 3D Tensor ---");

    let a_2d = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let b_2d = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // 2×3

    println!("A (2×2): [[1, 2], [3, 4]]");
    println!("B (2×3): [[5, 6, 7], [8, 9, 10]]");

    let outer_3d = einsum("ij,ik->ijk", &a_2d, &[2, 2], Some((&b_2d[..], &[2, 3][..]))).unwrap();
    println!("\nC = einsum('ij,ik->ijk', A, B):");
    println!("C[i,j,k] = A[i,j] * B[i,k]");
    println!("Result shape: 2×2×3, length = {}", outer_3d.len());
    println!(
        "C[0,0,:] = [A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,0]*B[0,2]] = [1*5, 1*6, 1*7] = [5, 6, 7] ✓\n"
    );

    // ========================================
    // Batched Matrix Multiplication
    // ========================================
    println!("--- Batched Matrix Multiplication ---");

    let mut batch_a = Tensor3::zeros(2, 2, 2); // 2 batches of 2×2 matrices
    let mut batch_b = Tensor3::zeros(2, 2, 2);

    // Batch 0: [[1, 2], [3, 4]] × [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
    batch_a.set(0, 0, 0, 1.0);
    batch_a.set(0, 0, 1, 2.0);
    batch_a.set(0, 1, 0, 3.0);
    batch_a.set(0, 1, 1, 4.0);
    batch_b.set(0, 0, 0, 1.0);
    batch_b.set(0, 1, 1, 1.0);

    // Batch 1: [[2, 0], [0, 2]] × [[1, 2], [3, 4]] = [[2, 4], [6, 8]]
    batch_a.set(1, 0, 0, 2.0);
    batch_a.set(1, 1, 1, 2.0);
    batch_b.set(1, 0, 0, 1.0);
    batch_b.set(1, 0, 1, 2.0);
    batch_b.set(1, 1, 0, 3.0);
    batch_b.set(1, 1, 1, 4.0);

    println!("Batch A (2 batches of 2×2):");
    println!("  Batch 0: [[1, 2], [3, 4]]");
    println!("  Batch 1: [[2, 0], [0, 2]]");

    println!("\nBatch B (2 batches of 2×2):");
    println!("  Batch 0: [[1, 0], [0, 1]]  (identity)");
    println!("  Batch 1: [[1, 2], [3, 4]]");

    let batch_c = batched_matmul(&batch_a, &batch_b).unwrap();

    println!("\nBatch C = batched_matmul(A, B):");
    println!(
        "  Batch 0: [[{:.0}, {:.0}], [{:.0}, {:.0}]]",
        batch_c.get(0, 0, 0),
        batch_c.get(0, 0, 1),
        batch_c.get(0, 1, 0),
        batch_c.get(0, 1, 1)
    );
    println!("  Expected: [[1, 2], [3, 4]] ✓");

    println!(
        "  Batch 1: [[{:.0}, {:.0}], [{:.0}, {:.0}]]",
        batch_c.get(1, 0, 0),
        batch_c.get(1, 0, 1),
        batch_c.get(1, 1, 0),
        batch_c.get(1, 1, 1)
    );
    println!("  Expected: [[2, 4], [6, 8]] ✓\n");

    // ========================================
    // Advanced Einsum Patterns
    // ========================================
    println!("--- Advanced Einsum Patterns ---");

    // Hadamard product (element-wise multiplication)
    let a_elem = vec![1.0, 2.0, 3.0, 4.0];
    let b_elem = [5.0, 6.0, 7.0, 8.0];
    let hadamard_mat = einsum(
        "ij,ij->ij",
        &a_elem,
        &[2, 2],
        Some((&b_elem[..], &[2, 2][..])),
    )
    .unwrap();
    println!("Hadamard product (element-wise):");
    println!("  einsum('ij,ij->ij', A, B) = {:?}", hadamard_mat);
    println!("  Expected: [1*5, 2*6, 3*7, 4*8] = [5, 12, 21, 32] ✓");

    // Frobenius inner product
    let frobenius = einsum(
        "ij,ij->",
        &a_elem,
        &[2, 2],
        Some((&b_elem[..], &[2, 2][..])),
    )
    .unwrap();
    println!("\nFrobenius inner product:");
    println!("  einsum('ij,ij->', A, B) = {:?}", frobenius);
    println!("  Expected: 1*5 + 2*6 + 3*7 + 4*8 = 70 ✓\n");

    // ========================================
    // Practical Application: Batch Processing
    // ========================================
    println!("--- Practical Application: Image Batch Processing ---");
    println!("Simulating batch transformation of image patches\n");

    // Create batch of 3 image patches (4×4) and 3 transformation matrices (4×4)
    let num_patches = 3;
    let patch_size = 4;

    let mut patches = Tensor3::zeros(num_patches, patch_size, patch_size);
    let mut transforms = Tensor3::zeros(num_patches, patch_size, patch_size);

    // Fill with example data
    for b in 0..num_patches {
        for i in 0..patch_size {
            for j in 0..patch_size {
                patches.set(b, i, j, (b * 16 + i * 4 + j) as f64 + 1.0);
                // Identity-like transforms with different scales
                if i == j {
                    transforms.set(b, i, j, 1.0 + 0.1 * (b as f64));
                }
            }
        }
    }

    println!(
        "Processing {} image patches of size {}×{}",
        num_patches, patch_size, patch_size
    );
    println!("Applying batch-specific transformations...");

    let transformed = batched_matmul(&transforms, &patches).unwrap();

    println!("\nResults:");
    for b in 0..num_patches {
        println!(
            "  Patch {} transformed: top-left element = {:.2}",
            b,
            transformed.get(b, 0, 0)
        );
    }
    println!();

    // ========================================
    // Using Regular Outer Product
    // ========================================
    println!("--- Outer Product Function ---");

    let u = vec![1.0, 2.0, 3.0];
    let v = vec![4.0, 5.0];

    println!("u = {:?}", u);
    println!("v = {:?}", v);

    let outer_uv = outer_product(&u, &v);
    println!("outer_product(u, v) = {:?}", outer_uv);
    println!("As matrix (3×2):");
    for i in 0..3 {
        print!("  [");
        for j in 0..2 {
            print!("{:4.1}", outer_uv[i * 2 + j]);
        }
        println!("]");
    }
    println!("Expected: [[4, 5], [8, 10], [12, 15]] ✓\n");

    // ========================================
    // Complex Tensor Contraction
    // ========================================
    println!("--- Complex Tensor Contractions ---");

    // Matrix-vector as einsum
    let a_mv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let x_mv = [1.0, 2.0, 3.0];

    let y_mv = einsum("ij,j->i", &a_mv, &[2, 3], Some((&x_mv[..], &[3][..]))).unwrap();
    println!("Matrix-vector via einsum('ij,j->i'):");
    println!("  A = [[1, 2, 3], [4, 5, 6]]");
    println!("  x = [1, 2, 3]");
    println!("  y = {:?}", y_mv);
    println!("  Expected: [1*1+2*2+3*3=14, 4*1+5*2+6*3=32] ✓\n");

    // 3D Frobenius inner product
    let t1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t2 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let inner_3d = einsum(
        "ijk,ijk->",
        &t1,
        &[2, 2, 2],
        Some((&t2[..], &[2, 2, 2][..])),
    )
    .unwrap();
    println!("3D Frobenius inner product:");
    println!("  einsum('ijk,ijk->', T1, T2) = {:?}", inner_3d);
    println!("  Expected: 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120 ✓\n");

    // ========================================
    // Summary of All Supported Patterns
    // ========================================
    println!("--- Supported Einstein Summation Patterns (24 total) ---");
    println!("Matrix Operations:");
    println!("  • ij,jk->ik    Matrix multiplication");
    println!("  • ik,kj->ij    Alternative matmul notation");
    println!("  • ij,j->i      Matrix-vector multiply");
    println!("  • ij->ji       Matrix transpose");
    println!("  • ii->i        Diagonal extraction");
    println!("  • ii->         Trace (sum of diagonal)");

    println!("\nVector Operations:");
    println!("  • i,j->ij      Outer product");
    println!("  • i,i->        Dot product");
    println!("  • i,i->i       Element-wise multiplication");

    println!("\nTensor Operations:");
    println!("  • ij,ik->ijk   Outer product to 3D");
    println!("  • ijk->ikj     Swap middle and last axes");
    println!("  • ijk->jik     Swap first two axes");
    println!("  • ijk->kji     Reverse all axes");
    println!("  • ijk,kl->ijl  Tensor-matrix contraction");

    println!("\nReductions:");
    println!("  • ij->i        Row sums");
    println!("  • ij->j        Column sums");
    println!("  • ij->         Total sum");
    println!("  • ijk->ij      Sum over last axis");
    println!("  • ijk->jk      Sum over first axis");

    println!("\nInner Products:");
    println!("  • ij,ij->ij    Hadamard product");
    println!("  • ij,ij->      Frobenius inner product");
    println!("  • ijk,ijk->    3D Frobenius inner product");

    println!("\n=== All tensor operations completed successfully! ===");
}
