use crate::memory::{Page, PageNumber, TlbIdentifierToken};
use std::num::NonZero;

/// This allows exposing things only inbetween the internal crates
pub trait AsFFI {
    type Inetrface<'a>
    where
        Self: 'a;

    fn as_ffi(&self) -> Self::Inetrface<'_>;
}

pub trait GetTlbIdentifier {
    fn tlb_ident(&self) -> TlbIdentifierToken;
}

pub trait IoMMUByteRawAccess {
    type Error;

    fn load_byte_raw(&self, vaddr: u64) -> Result<(PageNumber, &Page, u8), Self::Error>;

    fn store_byte_raw(&self, vaddr: u64, value: u8) -> Result<(PageNumber, &Page), Self::Error>;
}

pub trait IoMMURawIntAccess<T: bytemuck::Pod>: IoMMUByteRawAccess {
    fn load_raw(&self, vaddr: u64) -> Result<(PageNumber, &Page, Option<&Page>, T), Self::Error>;

    fn store_raw(
        &self,
        vaddr: u64,
        value: T,
    ) -> Result<(PageNumber, &Page, Option<&Page>), Self::Error>;
}

pub trait GetTlbGeneration {
    fn get_generation(&self) -> NonZero<u64>;
}

pub trait ResetTlbGeneration {
    /// # Safety
    ///
    /// must ensure that
    unsafe fn reset_generation(&mut self);
}
