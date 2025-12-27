//! Build script for oxiblas-ffi.
//!
//! Note: The C header file (include/oxiblas.h) is manually maintained
//! because cbindgen doesn't fully support Rust 2024's `#[unsafe(no_mangle)]` syntax.
//!
//! To regenerate the header automatically in the future when cbindgen adds support,
//! uncomment the cbindgen code below.

fn main() {
    // Rerun if source files change
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=include/oxiblas.h");

    // Note: cbindgen doesn't fully support Rust 2024's #[unsafe(no_mangle)] yet.
    // The C header (include/oxiblas.h) is manually maintained.
    // Uncomment below when cbindgen adds support:
    /*
    use std::env;
    use std::path::PathBuf;

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&crate_dir);

    let config = cbindgen::Config::from_file("cbindgen.toml")
        .unwrap_or_else(|_| cbindgen::Config::default());

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .map(|bindings| {
            bindings.write_to_file(out_dir.join("include").join("oxiblas_generated.h"));
        })
        .ok();
    */
}
