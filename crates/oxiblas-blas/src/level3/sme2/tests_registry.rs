//! The registry walk and the source scan: every registered kernel passes the
//! template lint, every streaming region in the sources is registered, and
//! every rule of the scan rejects a source that breaks it.
//!
//! [`registry`] is the one list of registered modules. A module that gets its
//! first `sme_kernels!` invocation (Wave 4 `pack_za.rs`, Wave 5
//! `level2/gemv/sme2.rs`) adds its `SME_KERNELS` here; until it does, the scan
//! fails and names the module.

use super::*;
use crate::level3::sme2::lint::{lint_kernel, lint_probe_template, report};
use crate::level3::sme2::{detect, kernel_f32, kernel_f64, test_support, tests_smoke};

/// Every registered streaming kernel of the crate.
fn registry() -> Vec<KernelEntry> {
    [
        kernel_f32::SME_KERNELS,
        kernel_f64::SME_KERNELS,
        tests_smoke::SME_KERNELS,
    ]
    .concat()
}

/// A registry entry for the scan tests (its template is never linted here).
fn entry(name: &'static str, file: &'static str) -> KernelEntry {
    KernelEntry {
        name,
        file,
        template: "",
        consts: &[],
        f64f64: false,
    }
}

/// `(path, text)` pairs for [`check_sources`].
fn sources(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
        .collect()
}

/// A module source with one invocation holding the named kernels.
fn kernel_module(names: &[&str]) -> String {
    let kernels: String = names
        .iter()
        .map(|name| {
            format!(
                "    /// Docs.\n    pub(super) streaming fn {name}(args: &MacroArgs<f32>) {{\n        \
                 extensions: [],\n        consts: [],\n        prologue: [],\n        \
                 body: [\"ptrue p0.s\"],\n    }}\n"
            )
        })
        .collect();
    format!("use super::asm::sme_kernels;\n\nsme_kernels! {{\n{kernels}}}\n")
}

#[test]
fn test_sme2_registry_lints_every_registered_kernel() {
    let registry = registry();
    assert!(registry.len() >= 2, "the smoke kernels are registered");
    for entry in &registry {
        if let Err(errors) = lint_kernel(entry) {
            panic!("{} ({}):\n{}", entry.name, entry.file, report(&errors));
        }
    }
    // The other templates of the module: the probe and the FPSR accessors.
    if let Err(errors) = lint_probe_template(detect::PROBE_TEMPLATE) {
        panic!("probe: {}", report(&errors));
    }
    for template in test_support::FPSR_TEMPLATES {
        if let Err(errors) = crate::level3::sme2::lint::lint_fpsr_template(template) {
            panic!("{template}: {}", report(&errors));
        }
    }
}

#[test]
fn test_sme2_registry_matches_the_sources() {
    let registry = registry();
    let (scanned, from_disk) = super::sources();
    let paths: Vec<&str> = scanned.iter().map(|(path, _)| path.as_str()).collect();
    for required in [
        "level3/sme2/asm.rs",
        "level3/sme2/tests_smoke.rs",
        "level2/gemv/mod.rs",
    ] {
        assert!(
            paths.contains(&required),
            "{required} not scanned: {paths:?}"
        );
    }
    let problems = check_sources(&scanned, &registry);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // The embedded copies pass as well, and cover exactly the tree.
    let embedded = embedded_sources();
    let problems = check_sources(&embedded, &registry);
    assert!(problems.is_empty(), "embedded: {}", problems.join("\n"));
    if from_disk {
        // The whole crate was read, subdirectories included.
        for required in ["lib.rs", "level3/gemm.rs", "level3/gemm_kernel/mod.rs"] {
            assert!(paths.contains(&required), "{required} not scanned");
        }
        let listed: Vec<&str> = EMBEDDED_SOURCES.iter().map(|(path, _)| *path).collect();
        let mismatch = embedded_mismatch(&paths, &listed);
        assert!(mismatch.is_empty(), "{}", mismatch.join("\n"));
        for (site, _) in ASM_SITES {
            assert!(
                paths.contains(site),
                "ASM_SITES names {site}, which is gone"
            );
        }
    } else {
        println!("source tree not readable: scanned the embedded copies");
    }
}

#[test]
fn test_sme2_registry_streaming_text_only_in_invocations() {
    assert!(mentions_streaming("smstart"));
    assert!(mentions_streaming(".arch_extension sme2"));
    assert!(mentions_streaming("SMSTOP"));
    assert!(mentions_streaming("nop\n.inst 0xd503467f"));
    assert!(mentions_streaming(".word 0"));
    assert!(mentions_streaming(".long 0"));
    assert!(mentions_streaming(".4byte 0"));
    assert!(!mentions_streaming("prfm pldl1keep, [{0}]"));
    assert!(!mentions_streaming("see the .words list and .longest run"));
    // Outside an invocation, in any file other than the lint's own.
    for (path, text) in [
        (
            "level3/gemm.rs",
            "const T: &str = \"nop\\n.inst 0xd503467f\";\n",
        ),
        ("level1/mod.rs", "const T: &str = \"smstart\";\n"),
        (
            "level3/sme2/kernel_f32.rs",
            "const T: &str = \".word 0\";\n",
        ),
        ("level3/sme2/pack.rs", "const T: &str = r#\"smstop\"#;\n"),
    ] {
        let problems = check_sources(&sources(&[(path, text)]), &[]);
        assert!(
            problems.iter().any(|p| p.contains("streaming-mode code")),
            "{path}: {problems:?}"
        );
    }
    // Inside the file's invocation it is the kernel text the lint checks.
    let module = kernel_module(&["k1"]).replace("ptrue p0.s", "smstop");
    assert_eq!(scan(&module).streaming_text, 0);
    let outside = format!("{module}const T: &str = \"smstop\";\n");
    assert_eq!(scan(&outside).streaming_text, 1);
    // The lint's own files may hold such text anywhere.
    let problems = check_sources(
        &sources(&[("level3/sme2/tests_lint.rs", "const T: &str = \"smstart\";")]),
        &[],
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn test_sme2_registry_scan_covers_the_whole_crate() {
    let raw = "fn f() { unsafe { core::arch::asm!(\"nop\") } }\n";
    // A subdirectory of an embedded directory, a sibling whose name only
    // starts like one, and any other file of the crate.
    for path in [
        "level3/sme2/extra/raw.rs",
        "level3/sme2_extra.rs",
        "level1/mod.rs",
        "lib.rs",
    ] {
        let problems = check_sources(&sources(&[(path, raw)]), &[]);
        assert!(
            problems.iter().any(|p| p.contains("raw asm!")),
            "{path}: {problems:?}"
        );
    }
    // A wrapper macro is an asm! site where it is defined.
    let wrapper = "macro_rules! emit { ($($t:tt)*) => { core::arch::asm!($($t)*) }; }\n";
    let problems = check_sources(&sources(&[("level1/mod.rs", wrapper)]), &[]);
    assert!(
        problems.iter().any(|p| p.contains("raw asm!")),
        "{problems:?}"
    );
    // Whitespace between the name and `!` is still an invocation.
    for text in [
        "fn f() { unsafe { core::arch::asm ! (\"nop\") } }",
        "fn f() { unsafe { core::arch::asm\n!(\"nop\") } }",
        "fn f() { unsafe { r#asm!(\"nop\") } }",
    ] {
        assert_eq!(scan(text).asm, 1, "{text}");
    }
    for text in [
        "let asm = 1; if asm != 2 {}",
        "use super::asm::sme_kernels;",
        "sme_asm!(x)",
    ] {
        assert_eq!(scan(text).asm, 0, "{text}");
    }
    // An assembly macro named through a metavariable, include!, and a
    // module path that leaves src/.
    for (text, needle) in [
        (
            "macro_rules! m { ($n:ident) => { core::arch::$n!(\"nop\") }; }",
            "metavariable",
        ),
        ("include!(\"../../outside.rs\");", "include!"),
        (
            "#[path = \"../../outside.rs\"]\nmod outside;",
            "leaves src/",
        ),
        ("#[ path=\"/abs/outside.rs\" ]\nmod outside;", "leaves src/"),
    ] {
        let problems = check_sources(&sources(&[("level3/gemm_thin.rs", text)]), &[]);
        assert!(
            problems.iter().any(|p| p.contains(needle)),
            "{text}: {problems:?}"
        );
    }
    assert!(
        scan("#[path = \"gemm_tests.rs\"]\nmod tests;")
            .escaping_paths
            .is_empty()
    );
    // The embedded list must cover every file under the embedded
    // directories, subdirectories included, compared by component.
    let tree = [
        "level3/sme2/asm.rs",
        "level3/sme2/extra/raw.rs",
        "level3/sme2_extra.rs",
    ];
    let mismatch = embedded_mismatch(&tree, &["level3/sme2/asm.rs"]);
    assert_eq!(mismatch.len(), 1, "{mismatch:?}");
    assert!(mismatch[0].contains("level3/sme2/extra/raw.rs"));
    assert!(in_embedded_dir("level2/gemv/sme2.rs"));
    assert!(!in_embedded_dir("level2/gemv_sme2.rs"));
}

#[test]
fn test_sme2_registry_scan_rejects_unregistered_regions() {
    let module = kernel_module(&["alpha_kernel", "beta_kernel"]);
    let path = "level3/sme2/kernel_f32.rs";
    let file = "crates/oxiblas-blas/src/level3/sme2/kernel_f32.rs";
    let both = [entry("alpha_kernel", file), entry("beta_kernel", file)];
    assert_eq!(
        check_sources(&sources(&[(path, &module)]), &both),
        Vec::<String>::new()
    );
    // One kernel without an entry.
    let problems = check_sources(&sources(&[(path, &module)]), &both[..1]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("beta_kernel"), "{problems:?}");
    // An entry whose kernel is gone, and an entry for another file.
    let problems = check_sources(
        &sources(&[(path, &kernel_module(&["alpha_kernel"]))]),
        &both,
    );
    assert!(
        problems.iter().any(|p| p.contains("`beta_kernel`")),
        "{problems:?}"
    );
    let elsewhere = [entry(
        "alpha_kernel",
        "crates/oxiblas-blas/src/level3/sme2/pack.rs",
    )];
    let problems = check_sources(
        &sources(&[(path, &kernel_module(&["alpha_kernel"]))]),
        &elsewhere,
    );
    assert_eq!(problems.len(), 2, "{problems:?}");
    // A registered name twice.
    let twice = [entry("alpha_kernel", file), entry("alpha_kernel", file)];
    let problems = check_sources(
        &sources(&[(path, &kernel_module(&["alpha_kernel"]))]),
        &twice,
    );
    assert!(
        problems.iter().any(|p| p.contains("registered twice")),
        "{problems:?}"
    );
}

#[test]
fn test_sme2_registry_scan_covers_the_gemv_directory() {
    // Today the GEMV directory has no SME code; when Wave 5 adds
    // `level2/gemv/sme2.rs` with kernels, they must be registered.
    let path = "level2/gemv/sme2.rs";
    let module = kernel_module(&["gemv_notrans_f32"]);
    let problems = check_sources(&sources(&[(path, &module)]), &[]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("level2/gemv/sme2.rs") && problems[0].contains("gemv_notrans_f32")
    );
    let registered = [entry(
        "gemv_notrans_f32",
        "crates/oxiblas-blas/src/level2/gemv/sme2.rs",
    )];
    assert_eq!(
        check_sources(&sources(&[(path, &module)]), &registered),
        Vec::<String>::new()
    );
    // A raw region there is caught as well.
    let raw = "fn f() { unsafe { core::arch::asm!(\"smstart\", \"smstop\") } }";
    let problems = check_sources(&sources(&[(path, raw)]), &[]);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("raw asm!"), "{problems:?}");
    assert!(problems[1].contains("streaming-mode code"), "{problems:?}");
}

#[test]
fn test_sme2_registry_scan_rejects_raw_asm_and_extra_invocations() {
    let raw =
        "pub(super) unsafe fn k() {\n    unsafe { core::arch::asm!(\"smstart\", \"smstop\") }\n}\n";
    // Not allowed in a kernel file (and its text is streaming code outside
    // an invocation), allowed (once) at an ASM_SITES entry.
    assert_eq!(
        check_sources(&sources(&[("level3/sme2/kernel_f64.rs", raw)]), &[]).len(),
        2
    );
    assert_eq!(
        check_sources(&sources(&[("level3/sme2/detect.rs", raw)]), &[]),
        Vec::<String>::new()
    );
    let twice = format!("{raw}{raw}");
    assert_eq!(
        check_sources(&sources(&[("level3/sme2/detect.rs", &twice)]), &[]).len(),
        1
    );
    // A site that lost its asm! is reported too (the list must stay exact).
    assert_eq!(
        check_sources(&sources(&[("level3/sme2/detect.rs", "fn f() {}")]), &[]).len(),
        1
    );
    // Module-level assembly is never allowed.
    for text in ["global_asm!(\".text\");", "naked_asm!(\"ret\")"] {
        let problems = check_sources(&sources(&[("level3/sme2/pack.rs", text)]), &[]);
        assert!(
            problems.iter().any(|p| p.contains("global_asm")),
            "{text}: {problems:?}"
        );
    }
    // Two invocations in one file.
    let one = kernel_module(&["first_kernel"]);
    let two = format!(
        "{one}\n{}",
        kernel_module(&["second_kernel"]).replace("use super::asm::sme_kernels;\n", "")
    );
    let file = "crates/oxiblas-blas/src/level3/sme2/kernel_f32.rs";
    let both = [entry("first_kernel", file), entry("second_kernel", file)];
    let problems = check_sources(&sources(&[("level3/sme2/kernel_f32.rs", &two)]), &both);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("2 sme_kernels! invocations"),
        "{problems:?}"
    );
    // A `streaming fn` outside any invocation (another macro, say).
    let stray = "other_macro! { streaming fn hidden(args: &A) {} }";
    let problems = check_sources(&sources(&[("level3/sme2/pack.rs", stray)]), &[]);
    assert!(
        problems.iter().any(|p| p.contains("outside sme_kernels!")),
        "{problems:?}"
    );
}

#[test]
fn test_sme2_registry_lexer_ignores_comments_and_literals() {
    let text = concat!(
        "// asm!(\"smstart\") in a comment\n",
        "/* block /* nested asm!() */ still comment sme_kernels! { streaming fn c() } */\n",
        "let s = \"core::arch::asm!(\\\"x\\\") and streaming fn s()\";\n",
        "let r = r#\"asm!(\"raw\") streaming fn r()\"#;\n",
        "let b = br\"asm!\";\n",
        "let c = '\"'; let d = '\\''; fn f<'a>(x: &'a str) {}\n",
        "core::arch::asm!(\"nop\");\n",
        "sme_asm!(not counted);\n",
    );
    let found = scan(text);
    assert_eq!(
        found,
        Found {
            asm: 1,
            module_asm: 0,
            invocations: 0,
            kernels: Vec::new(),
            generated_names: 0,
            asm_imports: 0,
            includes: 0,
            generated_arch_paths: 0,
            escaping_paths: Vec::new(),
            streaming_text: 0,
        }
    );
    let lexed = lex(text);
    assert_eq!(lexed.code.lines().count(), text.lines().count());
    assert!(lexed.strings.iter().any(|s| s.contains("raw")));
    assert!(lexed.code.contains("fn f<'a>(x: &'a str)"));
    // The macro definition's own pattern is not a kernel.
    let definition = "($vis:vis streaming fn $name:ident(args: &$args:ty) {})";
    assert_eq!(scan(definition).kernels, Vec::<String>::new());
    // A real kernel is found.
    assert_eq!(scan(&kernel_module(&["k1"])).kernels, ["k1"]);
    assert_eq!(scan(&kernel_module(&["k1"])).invocations, 1);
    // `file!()` paths are reduced to `src/`-relative ones.
    let entry = entry("k", "crates/oxiblas-blas/src/level3/sme2/kernel_f32.rs");
    assert_eq!(entry_path(&entry), "level3/sme2/kernel_f32.rs");
}

#[test]
fn test_sme2_registry_scan_rejects_generated_kernel_names() {
    // A wrapper macro that passes the kernel name through a metavariable
    // defines regions the scan cannot match with the registry.
    let wrapper = concat!(
        "use super::asm::sme_kernels;\n",
        "macro_rules! wrap { ($n:ident) => { sme_kernels! {\n",
        "    /// Docs.\n",
        "    pub(super) streaming fn $n(args: &MacroArgs<f32>) {\n",
        "        extensions: [], consts: [], prologue: [], body: [\"ptrue p0.s\"],\n",
        "    }\n",
        "} } }\n",
        "wrap!(hidden_kernel);\n",
    );
    let found = scan(wrapper);
    assert_eq!((found.generated_names, found.kernels.len()), (1, 0));
    for path in ["level3/sme2/kernel_f32.rs", "level2/gemv/sme2.rs"] {
        let problems = check_sources(&sources(&[(path, wrapper)]), &[]);
        assert!(
            problems.iter().any(|p| p.contains("must be literal")),
            "{path}: {problems:?}"
        );
    }
    // Only the definition of `sme_kernels!` may name kernels that way.
    let problems = check_sources(&sources(&[("level3/sme2/asm.rs", wrapper)]), &[]);
    assert!(
        !problems.iter().any(|p| p.contains("must be literal")),
        "{problems:?}"
    );
}

#[test]
fn test_sme2_registry_scan_rejects_imported_asm_macros() {
    for text in [
        "use core::arch::asm as emit;\nfn f() { unsafe { emit!(\"smstart\") } }\n",
        "use std::arch::{asm as a, global_asm};\n",
        "use core::arch::asm;\n",
        "pub(crate) use ::core::arch::naked_asm as n;\n",
        "use core::{arch::asm as hidden, mem};\n",
    ] {
        assert_eq!(asm_imports(&lex(text).code), 1, "{text}");
        let problems = check_sources(&sources(&[("level3/sme2/pack.rs", text)]), &[]);
        let flagged = problems
            .iter()
            .any(|p| p.contains("`use` of an assembly macro"));
        assert!(flagged, "{text}: {problems:?}");
    }
    for text in [
        "use super::asm::sme_kernels;\n",
        "use core::arch::aarch64::*;\n",
        "#[used]\nstatic X: u8 = 0;\n",
        "// use core::arch::asm as e;\n",
        "let s = \"use core::arch::asm;\";\n",
        "fn reuse_arch_asm() {}\n",
    ] {
        assert_eq!(asm_imports(&lex(text).code), 0, "{text}");
    }
}
