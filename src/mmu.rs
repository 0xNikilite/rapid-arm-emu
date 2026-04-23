use std::cell::Cell;
use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicU8, Ordering};
use crate::vaddr::VAddr;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct PageFlags: u8 {
        const READ    = 0b0001;
        const WRITE   = 0b0010;
        const EXECUTE = 0b0100;
        const DIRTY   = 0b1000;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryProtections: u8 {
        const READ = PageFlags::READ.bits();
        const WRITE = PageFlags::WRITE.bits();
        const EXECUTE = PageFlags::EXECUTE.bits();
    }
}


impl MemoryProtections {
    // Self is a superset of PageFlags
    pub(crate) fn into_page_flags(self) -> PageFlags {
        PageFlags::from_bits_retain((self & Self::all()).bits())
    }
}


pub(crate) struct Page<VAddr> {
    ptr: Option<NonNull<AtomicU8>>,
    page_flags: Cell<PageFlags>,
    _addr_ty: PhantomData<VAddr>
}

#[inline(always)]
unsafe fn assert_valid_page_read<A: VAddr>(
    page_base_ptr: *const AtomicU8,
    offset: usize,
    len: usize,
) {
    unsafe {
        core::hint::assert_unchecked(!page_base_ptr.is_null());
        core::hint::assert_unchecked(len <= A::PAGE_SIZE.unchecked_sub(offset));
        core::hint::assert_unchecked(page_base_ptr.addr().is_multiple_of(A::PAGE_SIZE));
    }
}

#[inline(always)]
unsafe fn memory_op<A: VAddr>(
    page_base_ptr: *const AtomicU8,
    offset: usize,
    len: usize,
    mut op: impl FnMut(*const AtomicU8, usize),
) {
    unsafe {
        assert_valid_page_read::<A>(page_base_ptr, offset, len);
        let ptr = page_base_ptr.add(offset);
        for i in 0..len {
            op(ptr.add(i), i)
        }
    }
}

impl<A: VAddr> Page<A> {
    #[inline(always)]
    fn get_data_ptr(&self, flags: PageFlags) -> Option<NonNull<AtomicU8>> {
        if (!self.page_flags.get().contains(flags)) | self.ptr.is_none() {
            cold_path();
            return None
        }
        
        self.ptr
    }


    #[inline(always)]
    fn has_access(&self, flags: PageFlags) -> bool {
        self.get_data_ptr(flags).is_some()
    }
    
    #[inline(always)]
    unsafe fn access(
        &self,
        offset: usize,
        len: usize,
        flags: PageFlags,
        op: impl FnMut(*const AtomicU8, usize),
    ) -> Result<(), ()> {
        let Some(ptr) = self.get_data_ptr(flags) else {
            cold_path();
            return Err(())
        };
        
        unsafe { assert_valid_page_read::<A>(ptr.as_ptr(), offset, len) }
        unsafe {
            memory_op::<A>(
                ptr.as_ptr(),
                offset,
                len,
                op
            )
        }

        Ok(())
    }

    #[inline(always)]
    pub(crate) unsafe fn load(&self, offset: usize, mem: &mut [MaybeUninit<u8>]) -> Result<(), ()> {
        unsafe {
            let mem_ptr = mem.as_mut_ptr().cast::<u8>();
            self.access(
                offset,
                mem.len(),
                PageFlags::READ,
                move |ptr, i| {
                    let value = (*ptr).load(Ordering::Relaxed);
                    std::ptr::write(mem_ptr.add(i), value)
                }
            )
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn store(&self, offset: usize, mem: &[u8]) -> Result<(), ()> {
        unsafe {
            let mem_ptr = mem.as_ptr();
            let result = self.access(
                offset,
                mem.len(),
                PageFlags::READ,
                move |ptr, i| {
                    let value = std::ptr::read(mem_ptr.add(i));
                    (*ptr).store(value, Ordering::Relaxed);
                }
            );

            result.map(|()| self.page_flags.update(|flags| flags | PageFlags::DIRTY))
        }
    }
}

macro_rules! impl_load_ops {
    {
        ty: $ty: ty,
        load: $load_name: ident,
        store: $store_name: ident $(,)?
    } => {
        impl<A: VAddr> Page<A> {
            #[inline(always)]
            pub(crate) unsafe fn $load_name(&self, offset: usize) -> Result<$ty, ()> {
                let value: [u8; size_of::<$ty>()] = unsafe {
                    let mut value = [const { MaybeUninit::<u8>::uninit() }; size_of::<$ty>()];
                    self.load(offset, &mut value)?;
                    core::mem::transmute(value)
                };

                Ok(<$ty>::from_le_bytes(value))
            }

            #[inline(always)]
            pub(crate) unsafe fn $store_name(&self, offset: usize, value: $ty) -> Result<(), ()> {
                let value: [u8; size_of::<$ty>()] = value.to_le_bytes();
                unsafe { self.store(offset, &value) }
            }
        }
    };

    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            $(impl_load_ops! {
                ty: [<u $bits>],
                load: [<load $bits>],
                store: [<store $bits>]
            })+
        }
    };
}

impl_load_ops! { 64, 32, 16, 8 }

pub(crate) struct MMU<VAddr> {
    pages: Vec<Page<VAddr>>,
}

unsafe impl<A> Send for MMU<A> {}
unsafe impl<A> Sync for MMU<A> {}

impl<A: VAddr> MMU<A> {
    pub fn new() -> Self {
        Self {
            pages: vec![]
        }
    }

    pub(crate) unsafe fn map_memory(
        &mut self,
        base: A,
        ptr: *mut u8,
        size: A,
        protections: MemoryProtections,
    ) {
        unsafe {
            core::hint::assert_unchecked(base.add_addr(size).is_some());
            core::hint::assert_unchecked(base.is_page_aligned());
            core::hint::assert_unchecked(size.is_page_aligned());
            core::hint::assert_unchecked(ptr.addr().is_multiple_of(A::PAGE_SIZE));

            let end_page = base
                .add_addr(size)
                .unwrap_unchecked()
                .div_page_size_unchecked()
                .try_to_usize();

            let Some(end_page) = end_page else {
                panic!("could not map pages into view; out of host memory")
            };

            // Safety: start page is smaller than end page, and end page fits in ram
            let start_page = base.div_page_size_unchecked().try_to_usize().unwrap_unchecked();

            if self.pages.len() < end_page {
                self.pages.resize_with(end_page, || const {
                    Page {
                        ptr: None,
                        page_flags: Cell::new(PageFlags::empty()),
                        _addr_ty: PhantomData,
                    }
                })
            }

            let base_ptr = NonNull::new_unchecked(ptr).cast::<AtomicU8>();
            let page_flags = protections.into_page_flags();
            for page_idx in start_page..end_page {
                let page = self.pages.get_unchecked_mut(page_idx);
                *page = Page {
                    ptr: Some(base_ptr.add(page_idx.unchecked_mul(A::PAGE_SIZE))),
                    page_flags: Cell::new(page_flags),
                    _addr_ty: PhantomData
                };
            }
        }
    }

    pub(crate) fn load<'a>(
        &self,
        vaddr: A,
        mem: &'a mut [MaybeUninit<u8>]
    ) -> Option<&'a mut [u8]> {
        if mem.is_empty() {
            return Some(&mut [])
        }
        
        let result = vaddr.add_offset(mem.len())
            .map(A::div_rem_page_size)
            .and_then(|(end_page, offset)| Some((end_page.try_to_usize()?, offset)))
            .filter(|&(end_page, ..)| end_page < self.pages.len())
            .map(|(end_page, end_offset)| {
                let (start_page, start_offset) = vaddr.div_rem_page_size();
                let start_page = unsafe { start_page.try_to_usize().unwrap_unchecked() };

                (start_page, end_page, start_offset, end_offset)
            });
        
        let Some((start_page, end_page, start_offset, end_offset)) = result else {
            cold_path();
            return None
        };
        
        
        // [----][----][----]
        //   ^            ^
        //   |____________|
        //   region to load
        
        unsafe {
            for page in self.pages.get_unchecked(start_page..=end_page) {
                if !page.has_access(PageFlags::READ) {
                    return None
                }
            }
            
            let pages_dif = end_page - start_page;
            let end_page_mem_offset = pages_dif.unchecked_mul(A::PAGE_SIZE);
            
            let start_unaligned = start_offset != 0;
            let end_unaligned = end_offset != 0;
            
            let head_offsset = core::hint::select_unpredictable(
                start_unaligned,
                A::PAGE_SIZE - start_offset,
                0
            );

            let mem_ptr = mem.as_mut_ptr();


            let head = slice::from_raw_parts_mut(
                mem_ptr,
                head_offsset
            );
            
            let middle = slice::from_raw_parts_mut(
                mem_ptr.add(start_offset),
                start_offset.unchecked_sub(end_page_mem_offset)
            );
            
            let tail = slice::from_raw_parts_mut(
                mem_ptr.add(end_page_mem_offset),
                end_offset
            );
            
            let aligned_page_start = start_page + start_unaligned as usize;
            let aligned_page_end = end_page - end_unaligned as usize;
            
            let aligned_pages = self
                .pages
                .get_unchecked(aligned_page_start..aligned_page_end);

            if !head.is_empty() {
                self.pages.get_unchecked(start_page).load(start_offset, head).unwrap_unchecked();
            }

            for (page_idx, page) in aligned_pages.iter().enumerate() {
                let start = page_idx.unchecked_mul(A::PAGE_SIZE);
                let end = start.unchecked_add(A::PAGE_SIZE);
                page.load(0, middle.get_unchecked_mut(start..end)).unwrap_unchecked()
            }
            
            if !tail.is_empty() {
                self.pages.get_unchecked(start_page).load(end_offset, tail).unwrap_unchecked();
            }
            
            Some(mem.assume_init_mut())
        }
    }
}