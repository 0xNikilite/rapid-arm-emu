pub(crate) mod sealed {
    use core::hash::Hash;
    use crate::mmu::{MemoryFault, MMU};

    /// # Safety
    ///
    /// - `PAGE_SIZE` must be a power of 2 greater than or equal to 64
    pub unsafe trait VAddr: 'static + Send + Sync + Copy + Ord + Hash + From<u8> {
        const NULL: Self;
        const PAGE_SIZE: usize;
        const PAGE_SIZE_SELF: Self;

        type InsnWord;

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
    }
}

pub trait VAddr: sealed::VAddr {}

impl<A: sealed::VAddr> VAddr for A {}
