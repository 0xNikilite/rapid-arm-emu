pub(crate) mod sealed {
    use core::hash::Hash;

    pub unsafe trait VAddr: Copy + Ord + Hash + From<u8> {
        const NULL: Self;
        const PAGE_SIZE: usize;

        fn reservation_index(self) -> usize;

        fn add_addr(self, other: Self) -> Option<Self>;
        fn add_offset(self, other: usize) -> Option<Self>;

        #[inline]
        fn inc(self) -> Option<Self> {
            self.add_addr(Self::from(1))
        }

        fn is_page_aligned(self) -> bool;

        fn try_to_usize(self) -> Option<usize>;

        fn div_rem_page_size(self) -> (Self, usize);

        unsafe fn div_page_size_unchecked(self) -> Self;
    }
}

pub trait VAddr: sealed::VAddr {}

impl<A: sealed::VAddr> VAddr for A {}
