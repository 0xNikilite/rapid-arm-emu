pub(crate) mod sealed {
    use core::hash::Hash;
    use std::sync::atomic::AtomicU8;
    use crate::mmu::{MemoryFault, MMU};

    /// # Safety
    ///
    /// - `PAGE_SIZE` must be a power of 2 greater than or equal to 64
    pub unsafe trait VAddr: 'static + Send + Sync + Copy + Ord + Hash + From<u8> {
        const NULL: Self;
        const PAGE_SIZE: usize;
        const PAGE_SIZE_SELF: Self;

        type InsnWord;
        type ExclusiveMonitorLoadValue: Eq + Hash + Copy;

        fn reservation_index(self) -> usize;

        fn add_addr(self, other: Self) -> Option<Self>;
        fn add_offset(self, other: usize) -> Option<Self>;

        #[inline]
        fn inc(self) -> Option<Self> {
            self.add_addr(Self::from(1))
        }

        fn is_multiple_of(self, other: Self) -> bool;

        fn is_page_aligned(self) -> bool {
            self.is_multiple_of(Self::PAGE_SIZE_SELF)
        }

        fn try_to_usize(self) -> Option<usize>;

        fn div_rem_page_size(self) -> (Self, usize);

        unsafe fn div_page_size_unchecked(self) -> Self;

        fn fetch_insn_word(self, mmu: &MMU<Self>) -> Result<Self::InsnWord, MemoryFault>;

        unsafe fn load64_le(ptr: *const AtomicU8) -> u64;
        unsafe fn store64_le(ptr: *const AtomicU8, value: u64);

        unsafe fn load32_le(ptr: *const AtomicU8) -> u32;
        unsafe fn store32_le(ptr: *const AtomicU8, value: u32);

        unsafe fn load16_le(ptr: *const AtomicU8) -> u16;
        unsafe fn store16_le(ptr: *const AtomicU8, value: u16);


        // Same exact function as (load/store)N_le but the pointer is guarenteed to be aligned
        unsafe fn load64_le_aligned(ptr: *const AtomicU8) -> u64;
        unsafe fn store64_le_aligned(ptr: *const AtomicU8, value: u64);

        unsafe fn load32_le_aligned(ptr: *const AtomicU8) -> u32;
        unsafe fn store32_le_aligned(ptr: *const AtomicU8, value: u32);

        unsafe fn load16_le_aligned(ptr: *const AtomicU8) -> u16;
        unsafe fn store16_le_aligned(ptr: *const AtomicU8, value: u16);


        unsafe fn load_byte(ptr: *const AtomicU8) -> u8;
        unsafe fn store_byte(ptr: *const AtomicU8, value: u8);
    }
}

pub trait VAddr: sealed::VAddr {}

impl<A: sealed::VAddr> VAddr for A {}
