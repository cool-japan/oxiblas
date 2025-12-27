//! Extended precision operations demonstration.
//!
//! This example shows:
//! - Quad precision (f128) arithmetic using QuadFloat
//! - Kahan summation for improved numerical accuracy
//! - Pairwise summation for large vector operations
//! - Mixed precision operations (f32 → f64)
//!
//! Run with: cargo run --example extended_precision --features f128

#[cfg(feature = "f128")]
use oxiblas_core::scalar::{QuadFloat, Real as ScalarReal, Scalar};

use oxiblas_blas::level1::{dot, dot_kahan, dot_pairwise, dsdot, sdsdot};

fn main() {
    println!("=== Extended Precision Operations ===\n");

    // ========================================
    // Quad Precision (f128) Arithmetic
    // ========================================
    #[cfg(feature = "f128")]
    {
        println!("--- Quad Precision (f128) Arithmetic ---");
        println!("QuadFloat provides ~31 decimal digits of precision\n");

        // Basic arithmetic
        let x = QuadFloat::from(3.0);
        let y = QuadFloat::from(4.0);
        let sum = x + y;
        let product = x * y;
        let quotient = x / y;

        println!("x = 3.0 (quad precision)");
        println!("y = 4.0 (quad precision)");
        println!("x + y = {:?}", sum);
        println!("x * y = {:?}", product);
        println!("x / y = {:?}", quotient);

        // Square root with extremely high precision
        let two = QuadFloat::from(2.0);
        let sqrt_two = ScalarReal::sqrt(two);
        println!("\nHigh-precision square root:");
        println!("sqrt(2) = {:?}", sqrt_two);
        println!("sqrt(2)² = {:?}", sqrt_two * sqrt_two);

        let error = Scalar::abs(sqrt_two * sqrt_two - two);
        println!("Error: {:?}", error);
        println!("(Error should be < 1e-28)\n");

        // Demonstration of precision advantage
        println!("Precision comparison:");
        let x_f64 = 1.0_f64 / 3.0_f64;
        let x_f128 = QuadFloat::from(1.0) / QuadFloat::from(3.0);

        println!("1/3 in f64:  {:.17}", x_f64);
        println!("1/3 in f128: {:?}", x_f128);
        println!("(f128 has significantly more precision)\n");

        // High-precision constants
        let pi_approx = QuadFloat::from(std::f64::consts::PI);
        println!("π ≈ {:?}", pi_approx);

        let e_approx = QuadFloat::from(std::f64::consts::E);
        println!("e ≈ {:?}\n", e_approx);
    }

    #[cfg(not(feature = "f128"))]
    {
        println!("--- Quad Precision (f128) ---");
        println!("Note: Run with --features f128 to enable quad precision examples\n");
    }

    // ========================================
    // Kahan Summation
    // ========================================
    println!("--- Kahan Summation (Compensated Summation) ---");
    println!("Reduces numerical errors in sum of many numbers\n");

    // Create a scenario where naive summation loses precision
    let large_numbers = vec![1e16; 1000];
    let small_numbers = vec![1.0; 1000];
    let mut mixed_data = Vec::new();
    mixed_data.extend_from_slice(&large_numbers);
    mixed_data.extend_from_slice(&small_numbers);

    // Naive dot product (may lose precision)
    let ones: Vec<f64> = vec![1.0; mixed_data.len()];
    let naive_sum = dot(&mixed_data, &ones);

    // Kahan dot product (compensated summation)
    let kahan_sum = dot_kahan(&mixed_data, &ones);

    println!("Summing 1000 × 1e16 + 1000 × 1.0:");
    println!("Naive sum:  {:.10e}", naive_sum);
    println!("Kahan sum:  {:.10e}", kahan_sum);
    println!("Expected:   1.000000001e19");
    println!("(Kahan summation preserves the small contributions)\n");

    // ========================================
    // Pairwise Summation
    // ========================================
    println!("--- Pairwise Summation (Divide-and-Conquer) ---");
    println!("O(log n) error growth vs O(n) for naive summation\n");

    // Large vector where order matters for precision
    let n = 100000;
    let data: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let ones_large: Vec<f64> = vec![1.0; n];

    let naive_result = dot(&data, &ones_large);
    let pairwise_result = dot_pairwise(&data, &ones_large);

    println!("Sum of 1/i for i=1 to {}:", n);
    println!("Naive:    {:.15}", naive_result);
    println!("Pairwise: {:.15}", pairwise_result);
    println!("Difference: {:.2e}", (naive_result - pairwise_result).abs());
    println!("(Pairwise is more accurate for long sums)\n");

    // ========================================
    // Mixed Precision Operations
    // ========================================
    println!("--- Mixed Precision (f32 → f64) ---");
    println!("Compute in f32, accumulate in f64 for speed + accuracy\n");

    let x_f32 = vec![1.5_f32, 2.5, 3.5, 4.5];
    let y_f32 = vec![5.5_f32, 6.5, 7.5, 8.5];

    // Mixed precision dot product (DSDOT - result in f64)
    let result_f64 = dsdot(&x_f32, &y_f32);

    // Mixed precision with alpha (SDSDOT - result in f32)
    let result_f32 = sdsdot(0.0, &x_f32, &y_f32);

    println!("x (f32) = {:?}", x_f32);
    println!("y (f32) = {:?}", y_f32);
    println!("dsdot(x, y) [f64 result] = {:.15}", result_f64);
    println!("sdsdot(0, x, y) [f32 result] = {:.7}", result_f32);

    let expected = 1.5 * 5.5 + 2.5 * 6.5 + 3.5 * 7.5 + 4.5 * 8.5;
    println!("Expected: {:.15}", expected);
    println!("Error (dsdot): {:.2e}\n", (result_f64 - expected).abs());

    // ========================================
    // Practical Application
    // ========================================
    println!("--- Practical Application: Financial Calculations ---");

    // Simulating portfolio value calculation with many small transactions
    let prices: Vec<f64> = vec![100.5, 50.25, 75.75, 200.0, 10.5];
    let quantities: Vec<f64> = vec![1000.0, 2000.0, 500.0, 100.0, 10000.0];

    let portfolio_value_naive = dot(&prices, &quantities);
    let portfolio_value_kahan = dot_kahan(&prices, &quantities);

    println!("Portfolio calculation (price × quantity):");
    println!("Prices:     {:?}", prices);
    println!("Quantities: {:?}", quantities);
    println!("Total value (naive):  ${:.2}", portfolio_value_naive);
    println!("Total value (Kahan):  ${:.2}", portfolio_value_kahan);
    println!(
        "Difference: ${:.2e}",
        (portfolio_value_naive - portfolio_value_kahan).abs()
    );
    println!("(For financial calculations, Kahan reduces rounding errors)\n");

    // ========================================
    // Accuracy Comparison
    // ========================================
    println!("--- Accuracy Comparison: Kahan vs Pairwise vs Naive ---");

    // Harmonic series partial sum - known to accumulate error
    let n_harmonic = 10_000_000;
    let harmonic: Vec<f64> = (1..=n_harmonic).map(|i| 1.0 / i as f64).collect();
    let ones_h: Vec<f64> = vec![1.0; n_harmonic];

    let naive_harmonic = dot(&harmonic, &ones_h);
    let kahan_harmonic = dot_kahan(&harmonic, &ones_h);
    let pairwise_harmonic = dot_pairwise(&harmonic, &ones_h);

    println!("Harmonic series H_{} = Σ(1/i):", n_harmonic);
    println!("Naive:    {:.15}", naive_harmonic);
    println!("Kahan:    {:.15}", kahan_harmonic);
    println!("Pairwise: {:.15}", pairwise_harmonic);
    println!(
        "Difference (Kahan - Naive):    {:.2e}",
        (kahan_harmonic - naive_harmonic).abs()
    );
    println!(
        "Difference (Pairwise - Naive): {:.2e}",
        (pairwise_harmonic - naive_harmonic).abs()
    );
    println!("(Kahan typically has best accuracy, Pairwise is a good compromise)\n");

    println!("=== Extended precision operations completed! ===");

    #[cfg(not(feature = "f128"))]
    println!("\nTip: Run with --features f128 to see quad-precision examples!");
}
