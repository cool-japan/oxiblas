//! Matrix Market format support.
//!
//! This module provides reading and writing of sparse matrices in
//! Matrix Market (MM) format, the standard exchange format used by
//! the SuiteSparse Matrix Collection (formerly Florida Matrix Collection).
//!
//! # Format Overview
//!
//! Matrix Market files consist of:
//! 1. A header line: `%%MatrixMarket matrix coordinate real general`
//! 2. Optional comment lines starting with `%`
//! 3. A size line: `nrows ncols nnz`
//! 4. Data lines: `row col value` (1-indexed)
//!
//! # Supported Types
//!
//! - **Coordinate format** (sparse): real, complex, pattern, integer
//! - **Array format** (dense): not yet implemented
//!
//! # Symmetry
//!
//! - **general**: No symmetry assumed
//! - **symmetric**: Only lower triangle stored, A = A^T
//! - **skew-symmetric**: Only lower triangle stored, A = -A^T
//! - **hermitian**: Only lower triangle stored, A = A^H (complex only)
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_sparse::mtx::{read_matrix_market, write_matrix_market};
//!
//! // Read a matrix from file
//! let csr = read_matrix_market::<f64>("matrix.mtx")?;
//!
//! // Write a matrix to file
//! write_matrix_market(&csr, "output.mtx", "My matrix")?;
//! ```

use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;
use num_traits::ToPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Error type for Matrix Market operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtxError {
    /// Invalid header format.
    InvalidHeader(String),
    /// Invalid data format.
    InvalidData(String),
    /// Unsupported matrix type.
    UnsupportedType(String),
    /// I/O error.
    IoError(String),
    /// Parse error.
    ParseError(String),
    /// Missing size line.
    MissingSizeLine,
    /// Index out of bounds.
    IndexOutOfBounds {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
        /// Matrix rows.
        nrows: usize,
        /// Matrix columns.
        ncols: usize,
    },
}

impl core::fmt::Display for MtxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader(s) => write!(f, "Invalid Matrix Market header: {s}"),
            Self::InvalidData(s) => write!(f, "Invalid data: {s}"),
            Self::UnsupportedType(s) => write!(f, "Unsupported matrix type: {s}"),
            Self::IoError(s) => write!(f, "I/O error: {s}"),
            Self::ParseError(s) => write!(f, "Parse error: {s}"),
            Self::MissingSizeLine => write!(f, "Missing size line"),
            Self::IndexOutOfBounds {
                row,
                col,
                nrows,
                ncols,
            } => {
                write!(
                    f,
                    "Index ({row}, {col}) out of bounds for {nrows}×{ncols} matrix"
                )
            }
        }
    }
}

impl std::error::Error for MtxError {}

/// Matrix Market object type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxObject {
    /// Matrix.
    Matrix,
    /// Vector.
    Vector,
}

/// Matrix Market format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxFormat {
    /// Coordinate (sparse) format.
    Coordinate,
    /// Array (dense) format.
    Array,
}

/// Matrix Market field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxField {
    /// Real values.
    Real,
    /// Complex values.
    Complex,
    /// Pattern only (structure, no values).
    Pattern,
    /// Integer values.
    Integer,
}

/// Matrix Market symmetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxSymmetry {
    /// General (no symmetry).
    General,
    /// Symmetric: A = A^T.
    Symmetric,
    /// Skew-symmetric: A = -A^T.
    SkewSymmetric,
    /// Hermitian: A = A^H.
    Hermitian,
}

/// Matrix Market header information.
#[derive(Debug, Clone)]
pub struct MtxHeader {
    /// Object type (matrix or vector).
    pub object: MtxObject,
    /// Format type (coordinate or array).
    pub format: MtxFormat,
    /// Field type (real, complex, pattern, integer).
    pub field: MtxField,
    /// Symmetry type.
    pub symmetry: MtxSymmetry,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Number of stored entries (for coordinate format).
    pub nnz: usize,
    /// Comment lines (without leading %).
    pub comments: Vec<String>,
}

/// Parses a Matrix Market header line.
fn parse_header_line(
    line: &str,
) -> Result<(MtxObject, MtxFormat, MtxField, MtxSymmetry), MtxError> {
    let line = line.to_lowercase();

    if !line.starts_with("%%matrixmarket") {
        return Err(MtxError::InvalidHeader(
            "Header must start with %%MatrixMarket".to_string(),
        ));
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return Err(MtxError::InvalidHeader(
            "Header must have 5 parts".to_string(),
        ));
    }

    let object = match parts[1] {
        "matrix" => MtxObject::Matrix,
        "vector" => MtxObject::Vector,
        other => {
            return Err(MtxError::UnsupportedType(format!(
                "Unknown object type: {other}"
            )));
        }
    };

    let format = match parts[2] {
        "coordinate" => MtxFormat::Coordinate,
        "array" => MtxFormat::Array,
        other => {
            return Err(MtxError::UnsupportedType(format!(
                "Unknown format: {other}"
            )));
        }
    };

    let field = match parts[3] {
        "real" => MtxField::Real,
        "double" => MtxField::Real,
        "complex" => MtxField::Complex,
        "pattern" => MtxField::Pattern,
        "integer" => MtxField::Integer,
        other => {
            return Err(MtxError::UnsupportedType(format!(
                "Unknown field type: {other}"
            )));
        }
    };

    let symmetry = match parts[4] {
        "general" => MtxSymmetry::General,
        "symmetric" => MtxSymmetry::Symmetric,
        "skew-symmetric" => MtxSymmetry::SkewSymmetric,
        "hermitian" => MtxSymmetry::Hermitian,
        other => {
            return Err(MtxError::UnsupportedType(format!(
                "Unknown symmetry: {other}"
            )));
        }
    };

    Ok((object, format, field, symmetry))
}

/// Reads a Matrix Market file header.
///
/// Returns the header information and the reader positioned after comments.
pub fn read_header<R: BufRead>(reader: &mut R) -> Result<MtxHeader, MtxError> {
    let mut line = String::new();

    // Read header line
    reader
        .read_line(&mut line)
        .map_err(|e| MtxError::IoError(e.to_string()))?;

    let (object, format, field, symmetry) = parse_header_line(line.trim())?;

    // Read comments
    let mut comments = Vec::new();
    loop {
        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|e| MtxError::IoError(e.to_string()))?;

        if line.is_empty() {
            return Err(MtxError::MissingSizeLine);
        }

        let trimmed = line.trim();
        if trimmed.starts_with('%') {
            comments.push(trimmed[1..].trim().to_string());
        } else {
            // This is the size line
            break;
        }
    }

    // Parse size line
    let size_parts: Vec<&str> = line.split_whitespace().collect();

    let (nrows, ncols, nnz) = match format {
        MtxFormat::Coordinate => {
            if size_parts.len() < 3 {
                return Err(MtxError::InvalidData(
                    "Coordinate size line must have 3 values".to_string(),
                ));
            }
            let nrows = size_parts[0]
                .parse::<usize>()
                .map_err(|_| MtxError::ParseError("Invalid nrows".to_string()))?;
            let ncols = size_parts[1]
                .parse::<usize>()
                .map_err(|_| MtxError::ParseError("Invalid ncols".to_string()))?;
            let nnz = size_parts[2]
                .parse::<usize>()
                .map_err(|_| MtxError::ParseError("Invalid nnz".to_string()))?;
            (nrows, ncols, nnz)
        }
        MtxFormat::Array => {
            if size_parts.len() < 2 {
                return Err(MtxError::InvalidData(
                    "Array size line must have 2 values".to_string(),
                ));
            }
            let nrows = size_parts[0]
                .parse::<usize>()
                .map_err(|_| MtxError::ParseError("Invalid nrows".to_string()))?;
            let ncols = size_parts[1]
                .parse::<usize>()
                .map_err(|_| MtxError::ParseError("Invalid ncols".to_string()))?;
            (nrows, ncols, nrows * ncols)
        }
    };

    Ok(MtxHeader {
        object,
        format,
        field,
        symmetry,
        nrows,
        ncols,
        nnz,
        comments,
    })
}

/// Reads a Matrix Market file and returns a CSR matrix.
///
/// # Arguments
///
/// * `path` - Path to the Matrix Market file
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The format is not supported
/// - The data is invalid
pub fn read_matrix_market<T: Scalar<Real = T> + Clone + Field + Real, P: AsRef<Path>>(
    path: P,
) -> Result<CsrMatrix<T>, MtxError> {
    let file = std::fs::File::open(path).map_err(|e| MtxError::IoError(e.to_string()))?;

    let mut reader = BufReader::new(file);
    read_matrix_market_from_reader(&mut reader)
}

/// Reads a Matrix Market file from a reader.
pub fn read_matrix_market_from_reader<T: Scalar<Real = T> + Clone + Field + Real, R: BufRead>(
    reader: &mut R,
) -> Result<CsrMatrix<T>, MtxError> {
    let header = read_header(reader)?;

    if header.format != MtxFormat::Coordinate {
        return Err(MtxError::UnsupportedType(
            "Only coordinate format is supported".to_string(),
        ));
    }

    if header.field == MtxField::Complex {
        return Err(MtxError::UnsupportedType(
            "Complex matrices not supported for real type".to_string(),
        ));
    }

    // Read data entries
    let mut rows = Vec::with_capacity(header.nnz);
    let mut cols = Vec::with_capacity(header.nnz);
    let mut vals = Vec::with_capacity(header.nnz);

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| MtxError::IoError(e.to_string()))?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(MtxError::InvalidData(format!(
                "Invalid data line: {trimmed}"
            )));
        }

        let row: usize = parts[0]
            .parse()
            .map_err(|_| MtxError::ParseError(format!("Invalid row: {}", parts[0])))?;
        let col: usize = parts[1]
            .parse()
            .map_err(|_| MtxError::ParseError(format!("Invalid col: {}", parts[1])))?;

        // Convert from 1-indexed to 0-indexed
        if row == 0 || col == 0 {
            return Err(MtxError::IndexOutOfBounds {
                row,
                col,
                nrows: header.nrows,
                ncols: header.ncols,
            });
        }
        let row = row - 1;
        let col = col - 1;

        if row >= header.nrows || col >= header.ncols {
            return Err(MtxError::IndexOutOfBounds {
                row: row + 1,
                col: col + 1,
                nrows: header.nrows,
                ncols: header.ncols,
            });
        }

        let val = if header.field == MtxField::Pattern {
            T::one()
        } else {
            if parts.len() < 3 {
                return Err(MtxError::InvalidData(format!(
                    "Missing value on line: {trimmed}"
                )));
            }
            parts[2]
                .parse::<f64>()
                .map_err(|_| MtxError::ParseError(format!("Invalid value: {}", parts[2])))
                .and_then(|v| {
                    T::from_f64(v)
                        .ok_or_else(|| MtxError::ParseError(format!("Cannot convert value: {v}")))
                })?
        };

        rows.push(row);
        cols.push(col);
        vals.push(val.clone());

        // Handle symmetry
        if row != col {
            match header.symmetry {
                MtxSymmetry::Symmetric => {
                    rows.push(col);
                    cols.push(row);
                    vals.push(val);
                }
                MtxSymmetry::SkewSymmetric => {
                    rows.push(col);
                    cols.push(row);
                    vals.push(T::zero() - val);
                }
                MtxSymmetry::Hermitian => {
                    // For real matrices, Hermitian = Symmetric
                    rows.push(col);
                    cols.push(row);
                    vals.push(val);
                }
                MtxSymmetry::General => {}
            }
        }
    }

    // Build COO and convert to CSR
    let coo = CooMatrix::new(header.nrows, header.ncols, rows, cols, vals)
        .map_err(|e| MtxError::InvalidData(format!("Failed to create COO matrix: {e:?}")))?;

    Ok(crate::convert::coo_to_csr(&coo))
}

/// Reads a Matrix Market file and returns a COO matrix.
pub fn read_matrix_market_coo<T: Scalar<Real = T> + Clone + Field + Real, P: AsRef<Path>>(
    path: P,
) -> Result<CooMatrix<T>, MtxError> {
    let csr: CsrMatrix<T> = read_matrix_market(path)?;
    Ok(crate::convert::csr_to_coo(&csr))
}

/// Writes a CSR matrix to Matrix Market format.
///
/// # Arguments
///
/// * `csr` - The matrix to write
/// * `path` - Path to write to
/// * `comment` - Optional comment to include in the file
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_matrix_market<T: Scalar + Clone + Field + ToPrimitive, P: AsRef<Path>>(
    csr: &CsrMatrix<T>,
    path: P,
    comment: Option<&str>,
) -> Result<(), MtxError> {
    let file = std::fs::File::create(path).map_err(|e| MtxError::IoError(e.to_string()))?;

    let mut writer = std::io::BufWriter::new(file);
    write_matrix_market_to_writer(csr, &mut writer, comment)
}

/// Writes a CSR matrix to Matrix Market format using a writer.
pub fn write_matrix_market_to_writer<T: Scalar + Clone + Field + ToPrimitive, W: Write>(
    csr: &CsrMatrix<T>,
    writer: &mut W,
    comment: Option<&str>,
) -> Result<(), MtxError> {
    let eps = <T as Scalar>::epsilon();

    // Count actual non-zeros
    let mut nnz = 0;
    for (_, _, val) in csr.iter() {
        if Scalar::abs(val.clone()) > eps {
            nnz += 1;
        }
    }

    // Write header
    writeln!(writer, "%%MatrixMarket matrix coordinate real general")
        .map_err(|e| MtxError::IoError(e.to_string()))?;

    // Write comment if provided
    if let Some(c) = comment {
        for line in c.lines() {
            writeln!(writer, "% {line}").map_err(|e| MtxError::IoError(e.to_string()))?;
        }
    }

    // Write size line
    writeln!(writer, "{} {} {}", csr.nrows(), csr.ncols(), nnz)
        .map_err(|e| MtxError::IoError(e.to_string()))?;

    // Write data (1-indexed)
    for (row, col, val) in csr.iter() {
        if Scalar::abs(val.clone()) > eps {
            let f = val.to_f64().unwrap_or(0.0);
            writeln!(writer, "{} {} {}", row + 1, col + 1, f)
                .map_err(|e| MtxError::IoError(e.to_string()))?;
        }
    }

    Ok(())
}

/// Writes a symmetric CSR matrix to Matrix Market format.
///
/// Only writes the lower triangle, with symmetric flag.
pub fn write_matrix_market_symmetric<T: Scalar + Clone + Field + ToPrimitive, P: AsRef<Path>>(
    csr: &CsrMatrix<T>,
    path: P,
    comment: Option<&str>,
) -> Result<(), MtxError> {
    let file = std::fs::File::create(path).map_err(|e| MtxError::IoError(e.to_string()))?;

    let mut writer = std::io::BufWriter::new(file);
    let eps = <T as Scalar>::epsilon();

    // Count lower triangle non-zeros
    let mut nnz = 0;
    for (row, col, val) in csr.iter() {
        if row >= col && Scalar::abs(val.clone()) > eps {
            nnz += 1;
        }
    }

    // Write header
    writeln!(writer, "%%MatrixMarket matrix coordinate real symmetric")
        .map_err(|e| MtxError::IoError(e.to_string()))?;

    if let Some(c) = comment {
        for line in c.lines() {
            writeln!(writer, "% {line}").map_err(|e| MtxError::IoError(e.to_string()))?;
        }
    }

    writeln!(writer, "{} {} {}", csr.nrows(), csr.ncols(), nnz)
        .map_err(|e| MtxError::IoError(e.to_string()))?;

    // Write lower triangle only (1-indexed)
    for (row, col, val) in csr.iter() {
        if row >= col && Scalar::abs(val.clone()) > eps {
            let f = val.to_f64().unwrap_or(0.0);
            writeln!(writer, "{} {} {}", row + 1, col + 1, f)
                .map_err(|e| MtxError::IoError(e.to_string()))?;
        }
    }

    Ok(())
}

/// Reads Matrix Market format from a string.
pub fn read_matrix_market_str<T: Scalar<Real = T> + Clone + Field + Real>(
    s: &str,
) -> Result<CsrMatrix<T>, MtxError> {
    let mut reader = BufReader::new(s.as_bytes());
    read_matrix_market_from_reader(&mut reader)
}

/// Writes a CSR matrix to Matrix Market format as a string.
pub fn write_matrix_market_str<T: Scalar + Clone + Field + ToPrimitive>(
    csr: &CsrMatrix<T>,
    comment: Option<&str>,
) -> Result<String, MtxError> {
    let mut buf = Vec::new();
    write_matrix_market_to_writer(csr, &mut buf, comment)?;
    String::from_utf8(buf).map_err(|e| MtxError::IoError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header() {
        let (obj, fmt, field, sym) =
            parse_header_line("%%MatrixMarket matrix coordinate real general").unwrap();

        assert_eq!(obj, MtxObject::Matrix);
        assert_eq!(fmt, MtxFormat::Coordinate);
        assert_eq!(field, MtxField::Real);
        assert_eq!(sym, MtxSymmetry::General);
    }

    #[test]
    fn test_parse_header_symmetric() {
        let (_, _, _, sym) =
            parse_header_line("%%MatrixMarket matrix coordinate real symmetric").unwrap();

        assert_eq!(sym, MtxSymmetry::Symmetric);
    }

    #[test]
    fn test_read_simple_matrix() {
        let mtx = r#"%%MatrixMarket matrix coordinate real general
% A simple test matrix
3 3 5
1 1 1.0
1 3 2.0
2 2 3.0
3 1 4.0
3 3 5.0
"#;

        let csr: CsrMatrix<f64> = read_matrix_market_str(mtx).unwrap();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 5);

        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(0, 2), Some(&2.0));
        assert_eq!(csr.get(1, 1), Some(&3.0));
        assert_eq!(csr.get(2, 0), Some(&4.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_read_symmetric_matrix() {
        let mtx = r#"%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 1.0
2 1 2.0
2 2 3.0
3 3 4.0
"#;

        let csr: CsrMatrix<f64> = read_matrix_market_str(mtx).unwrap();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);

        // Symmetric entries
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(1, 0), Some(&2.0));
        assert_eq!(csr.get(0, 1), Some(&2.0)); // Symmetric fill-in
        assert_eq!(csr.get(1, 1), Some(&3.0));
        assert_eq!(csr.get(2, 2), Some(&4.0));
    }

    #[test]
    fn test_read_pattern_matrix() {
        let mtx = r#"%%MatrixMarket matrix coordinate pattern general
2 2 2
1 1
2 2
"#;

        let csr: CsrMatrix<f64> = read_matrix_market_str(mtx).unwrap();

        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(1, 1), Some(&1.0));
    }

    #[test]
    fn test_write_read_roundtrip() {
        // Create a simple matrix
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        // Write to string
        let mtx_str = write_matrix_market_str(&csr, Some("Test matrix")).unwrap();

        // Read back
        let csr2: CsrMatrix<f64> = read_matrix_market_str(&mtx_str).unwrap();

        assert_eq!(csr.nrows(), csr2.nrows());
        assert_eq!(csr.ncols(), csr2.ncols());
        assert_eq!(csr.nnz(), csr2.nnz());

        for row in 0..3 {
            for col in 0..3 {
                let v1 = csr.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_header_parsing_error() {
        let result = parse_header_line("invalid header");
        assert!(result.is_err());
    }

    #[test]
    fn test_index_error() {
        let mtx = r#"%%MatrixMarket matrix coordinate real general
2 2 1
3 1 1.0
"#;

        let result: Result<CsrMatrix<f64>, _> = read_matrix_market_str(mtx);
        assert!(result.is_err());
    }
}
