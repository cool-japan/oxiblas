//! LAPACK FFI - Linear system solve routines.
//!
//! Split into modules:
//! - `basic_solve`: Basic solve routines (GESV, GETRS, POTRS, GELS)
//! - `expert`: Expert drivers (GESVX, POSVX, SYSVX)
//! - `refinement`: Iterative refinement (GERFS, PORFS, SYRFS)

pub mod basic_solve;
pub mod expert;
pub mod refinement;

// Re-export all functions
pub use basic_solve::*;
pub use expert::*;
pub use refinement::*;
