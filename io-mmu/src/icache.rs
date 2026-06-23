use emu_abi::memory::PagePointer;

mod sealed {
    use crate::icache::ICache;

    /// # Safety
    ///
    /// must provide a pointer with the metadata of self
    pub unsafe trait DynUpgrade {
        fn get_metadata_ptr<'a>(&self) -> *const (dyn ICache + 'a)
        where
            Self: 'a;
    }

    unsafe impl<T: Sized + ICache> DynUpgrade for T {
        fn get_metadata_ptr<'a>(&self) -> *const (dyn ICache + 'a)
        where
            T: 'a,
        {
            self
        }
    }
}

pub trait ICache: 'static + Send + Sync + sealed::DynUpgrade {
    fn invalidate(&self, page: PagePointer);
}
