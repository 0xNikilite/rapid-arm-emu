#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(
    // F32 -> F64, Int -> Float, etc.
    clippy::cast_precision_loss,
    // U32 -> U8, etc.
    clippy::cast_possible_truncation,
    // Signed -> Unsigned, etc.
    clippy::cast_possible_wrap,
    // Signed -> Unsigned
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects,
    reason = "emulators require precise bit-level accuracy; \
              implicit casts can introduce subtle, hard-to-debug architectural discrepancies"
)]

mod ir;
pub(crate) mod sync;
pub mod cpu_fabric;

pub mod mmu;
pub mod vaddr;
pub mod halt_reason;
pub mod a64;
