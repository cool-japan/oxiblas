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
    println!("dot_kahan carries the rounding error of each addition into the next one\n");

    // Sixteen terms 1e16 (one per lane of the vectorised `dot` kernel on
    // aarch64) followed by 16 000 unit terms. The exact sum, 1.6e17 + 16 000,
    // is a double, but a partial sum of 1e16 or more (doubles there are at
    // least 2 apart) rounds a lone `+ 1` away.
    let mut mixed_data = vec![1e16; 16];
    mixed_data.extend(std::iter::repeat_n(1.0, 16_000));
    let ones: Vec<f64> = vec![1.0; mixed_data.len()];
    let exact_mixed = 16.0 * 1e16 + 16_000.0;

    let dot_sum = dot(&mixed_data, &ones);
    let pairwise_sum = dot_pairwise(&mixed_data, &ones);
    let kahan_sum = dot_kahan(&mixed_data, &ones);

    println!("Summing 16 × 1e16 + 16000 × 1.0:");
    for (name, value) in [
        ("dot", dot_sum),
        ("dot_pairwise", pairwise_sum),
        ("dot_kahan", kahan_sum),
    ] {
        println!(
            "{name:<13} {value:.0} (error {:.0})",
            (value - exact_mixed).abs()
        );
    }
    println!("Exact:        {exact_mixed:.0}");
    println!("(dot and dot_pairwise drop the unit terms added to a partial sum");
    println!(" that holds a 1e16; Kahan summation keeps them)\n");

    // ========================================
    // Pairwise Summation
    // ========================================
    println!("--- Pairwise Summation (Divide-and-Conquer) ---");
    println!("dot_pairwise halves the vector down to blocks of at most 128 terms and");
    println!("sums each block in turn: its error bound grows like 128 + log2(n/128)");
    println!("roundings instead of the n of one running sum. dot is not one running");
    println!("sum either: on aarch64 it keeps 16 NEON lanes flushed every 1024 terms");
    println!("(about 72 + n/1024 roundings), elsewhere four accumulators (about n/4),");
    println!("so neither result is reliably the more accurate one.\n");

    // Large vector where order matters for precision
    let n = 100000;
    let data: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let ones_large: Vec<f64> = vec![1.0; n];

    let dot_result = dot(&data, &ones_large);
    let pairwise_result = dot_pairwise(&data, &ones_large);
    let kahan_result = dot_kahan(&data, &ones_large);

    println!("Sum of 1/i for i=1 to {}:", n);
    println!("dot:          {:.15}", dot_result);
    println!("dot_pairwise: {:.15}", pairwise_result);
    println!("dot_kahan:    {:.15} (reference)", kahan_result);
    println!(
        "|dot - Kahan| = {:.2e}, |pairwise - Kahan| = {:.2e}\n",
        (dot_result - kahan_result).abs(),
        (pairwise_result - kahan_result).abs()
    );

    // ========================================
    // Mixed Precision Operations
    // ========================================
    println!("--- Mixed Precision (f32 → f64) ---");
    println!("dsdot / sdsdot widen every f32 element to f64 and multiply and");
    println!("accumulate in f64 (sdsdot rounds the final sum back to f32)\n");

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

    let portfolio_value_dot = dot(&prices, &quantities);
    let portfolio_value_kahan = dot_kahan(&prices, &quantities);

    println!("Portfolio calculation (price × quantity):");
    println!("Prices:     {:?}", prices);
    println!("Quantities: {:?}", quantities);
    println!("Total value (dot):    ${:.2}", portfolio_value_dot);
    println!("Total value (Kahan):  ${:.2}", portfolio_value_kahan);
    println!(
        "Difference: ${:.2e}",
        (portfolio_value_dot - portfolio_value_kahan).abs()
    );
    println!("(For financial calculations, Kahan reduces rounding errors)\n");

    // ========================================
    // Accuracy Comparison
    // ========================================
    println!("--- Accuracy Comparison: Kahan vs Pairwise vs dot ---");

    // Harmonic series partial sum - known to accumulate error
    let n_harmonic = 10_000_000;
    let harmonic: Vec<f64> = (1..=n_harmonic).map(|i| 1.0 / i as f64).collect();
    let ones_h: Vec<f64> = vec![1.0; n_harmonic];

    let dot_harmonic = dot(&harmonic, &ones_h);
    let kahan_harmonic = dot_kahan(&harmonic, &ones_h);
    let pairwise_harmonic = dot_pairwise(&harmonic, &ones_h);

    println!("Harmonic series H_{} = Σ(1/i):", n_harmonic);
    println!("dot:      {:.15}", dot_harmonic);
    println!("Kahan:    {:.15}", kahan_harmonic);
    println!("Pairwise: {:.15}", pairwise_harmonic);
    println!(
        "Difference (dot - Kahan):      {:.2e}",
        (dot_harmonic - kahan_harmonic).abs()
    );
    println!(
        "Difference (Pairwise - Kahan): {:.2e}",
        (pairwise_harmonic - kahan_harmonic).abs()
    );
    println!("(Kahan has the smallest error bound, about two roundings of the sum;");
    println!(" pairwise and the multi-accumulator dot trade some accuracy for speed)\n");

    println!("=== Extended precision operations completed! ===");

    #[cfg(not(feature = "f128"))]
    println!("\nTip: Run with --features f128 to see quad-precision examples!");
}
