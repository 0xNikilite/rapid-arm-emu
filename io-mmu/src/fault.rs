#[derive(Debug, thiserror::Error)]
pub enum MemoryFaultReason {
    #[error("invalid memory permisions")]
    GeneralProtection,
    #[error("invalid memory access {0}")]
    MemoryBus(anyhow::Error),
}

/// Fault returned when a memory access is invalid.
///
/// This is returned when an access:
/// - targets an unmapped page,
/// - violates page permissions,
/// - overflows the virtual address range,
/// - fails an address-alignment check required by a specific operation,
/// - crosses into an unmapped or insufficiently-permitted page,
/// - or otherwise fails MMU validation.
#[derive(Debug, thiserror::Error)]
#[error("memory fault at {vaddr}: {reason}")]
pub struct MemoryFault {
    vaddr: u64,
    reason: MemoryFaultReason,
}

impl MemoryFault {
    #[inline(always)]
    #[cold]
    pub const fn general_protection(vaddr: u64) -> Self {
        std::hint::cold_path();
        Self {
            vaddr,
            reason: MemoryFaultReason::GeneralProtection,
        }
    }

    pub const fn memory_bus(vaddr: u64, reason: anyhow::Error) -> Self {
        std::hint::cold_path();
        Self {
            vaddr,
            reason: MemoryFaultReason::MemoryBus(reason),
        }
    }
}

macro_rules! ensure {
    (vaddr: $vaddr: expr, $($expr: expr),+ $(,)?) => {
        if !($({ $expr })&&+) {
            ::std::hint::cold_path();
            return Err(MemoryFault::general_protection($vaddr))
        }
    };
}

pub(crate) use ensure;
