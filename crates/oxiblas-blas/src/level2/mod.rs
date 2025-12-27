//! BLAS Level 2 - Matrix-Vector Operations.
//!
//! This module provides matrix-vector operations:
//!
//! - `gemv` - General matrix-vector multiply: y = α·A·x + β·y
//! - `symv` - Symmetric matrix-vector multiply: y = α·A·x + β·y
//! - `hemv` - Hermitian matrix-vector multiply: y = α·A·x + β·y
//! - `ger` - Rank-1 update: A = α·x·y^T + A
//! - `trmv` - Triangular matrix-vector multiply: x = A·x
//! - `trsv` - Triangular solve: x = A⁻¹·b
//! - `syr` - Symmetric rank-1 update: A = α·x·x^T + A
//! - `her` - Hermitian rank-1 update: A = α·x·x^H + A
//!
//! ## Packed Operations
//!
//! For symmetric/Hermitian/triangular matrices stored in packed format:
//!
//! - `spmv` - Symmetric packed matrix-vector multiply
//! - `hpmv` - Hermitian packed matrix-vector multiply
//! - `tpmv` - Triangular packed matrix-vector multiply
//! - `tpsv` - Triangular packed solve
//! - `spr` - Symmetric packed rank-1 update
//! - `hpr` - Hermitian packed rank-1 update
//! - `spr2` - Symmetric packed rank-2 update
//! - `hpr2` - Hermitian packed rank-2 update
//!
//! # Example
//!
//! ```
//! use oxiblas_blas::level2::{gemv, GemvTrans};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0],
//!     &[4.0, 5.0, 6.0],
//! ]);
//! let x = [1.0f64, 2.0, 3.0];
//! let mut y = [0.0f64, 0.0];
//!
//! // y = A * x
//! gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
//!
//! // y[0] = 1*1 + 2*2 + 3*3 = 14
//! assert!((y[0] - 14.0).abs() < 1e-10);
//! ```

mod gbmv;
mod gemv;
mod ger;
mod hbmv;
mod hemv;
mod her;
mod her2;
mod hpmv;
mod hpr;
mod hpr2;
mod sbmv;
mod spmv;
mod spr;
mod spr2;
mod symv;
mod syr;
mod syr2;
mod tbmv;
mod tbsv;
mod tpmv;
mod tpsv;
mod trmv;
mod trsv;

pub use gbmv::{GbmvError, GbmvTrans, gbmv, gbmv_new};
pub use gemv::{
    GemvTrans, gemv, gemv_add, gemv_add_inplace, gemv_simple, gemv_sum2, gemv_with_par,
};
pub use ger::{ger, gerc};
pub use hbmv::{HbmvError, HbmvUplo, hbmv, hbmv_new};
pub use hemv::{HemvError, HemvUplo, hemv, hemv_new};
pub use her::{HerError, HerUplo, her, her_new};
pub use her2::{Her2Error, Her2Uplo, her2, her2_new};
pub use hpmv::{HpmvError, HpmvUplo, hpmv, hpmv_new};
pub use hpr::{HprError, HprUplo, hpr, hpr_new};
pub use hpr2::{Hpr2Error, Hpr2Uplo, hpr2, hpr2_new};
pub use sbmv::{SbmvError, SbmvUplo, sbmv, sbmv_new};
pub use spmv::{SpmvError, SpmvUplo, spmv, spmv_new};
pub use spr::{SprError, SprUplo, spr, spr_new};
pub use spr2::{Spr2Error, Spr2Uplo, spr2, spr2_new};
pub use symv::{SymvError, SymvUplo, symv, symv_new};
pub use syr::{SyrError, SyrUplo, syr, syr_new};
pub use syr2::{Syr2Error, Syr2Uplo, syr2, syr2_new};
pub use tbmv::{TbmvDiag, TbmvError, TbmvTrans, TbmvUplo, tbmv, tbmv_new};
pub use tbsv::{TbsvDiag, TbsvError, TbsvTrans, TbsvUplo, tbsv, tbsv_new};
pub use tpmv::{TpmvDiag, TpmvError, TpmvTrans, TpmvUplo, tpmv, tpmv_new};
pub use tpsv::{TpsvDiag, TpsvError, TpsvTrans, TpsvUplo, tpsv, tpsv_new};
pub use trmv::{DiagKind, TrmvError, TrmvOp, TrmvUplo, trmv, trmv_alloc};
pub use trsv::{TriangularMode, TriangularSide, trsv, trsv_in_place};
