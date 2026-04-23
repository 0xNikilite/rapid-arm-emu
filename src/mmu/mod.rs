//! Page-based virtual memory abstraction backed by host memory.
//!
//! This module implements a small MMU-like layer that maps guest virtual pages
//! onto host memory and enforces per-page access permissions.
//!
//! Core behavior:
//! - Memory is mapped in page-sized chunks.
//! - Each page may be readable, writable, and/or executable.
//! - Unmapped access or permission violations return [`MemoryFault`].
//! - Byte-range reads and writes may span multiple pages.
//! - Aligned 8/16/32/64-bit scalar loads/stores are provided as fast paths.
//! - Multi-byte scalar accesses use tearing-friendly memory operations.
//!
//! Concurrency model:
//! - Backing memory is represented as `AtomicU8` bytes.
//! - Reads and writes use relaxed atomic byte access or platform-specific
//!   tearing operations.
//! - Concurrent access is allowed.
//! - Tearing is permitted and is part of the contract, not undefined behavior.
//!
//! Safety model:
//! - Mapping requires the caller to provide page-aligned addresses and pointers.
//! - Several internal operations rely on alignment and bounds invariants and use
//!   unchecked assertions to preserve performance.
//! - The implementation is designed so that concurrent access does not create UB,
//!   even when read/write races produce torn values.
//!
//! Typical usage:
//! 1. Construct an [`MMU`].
//! 2. Map one or more host memory regions with [`MMU::map_memory`].
//! 3. Access bytes through [`MMU::load`] / [`MMU::store`].
//! 4. Access aligned scalars through `load8/load16/load32/load64` and
//!    `store8/store16/store32/store64`.


mod tear_mem_ops;

use std::cell::Cell;
use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
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


/// Fault returned when a memory access is invalid.
///
/// This is returned when an access:
/// - targets an unmapped page,
/// - loadN/storeN was used with an unaligned pointer
/// - violates page permissions,
/// - overflows the virtual address range,
/// - or otherwise fails MMU validation.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid memory access")]
pub struct MemoryFault(());

impl MemoryFault {
    #[inline(always)]
    #[cold]
    pub const fn fault() -> Self {
        cold_path();
        Self(())
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

impl<A: VAddr> Page<A> {
    #[inline(always)]
    fn get_data_ptr(&self, flags: PageFlags) -> Result<NonNull<AtomicU8>, MemoryFault> {
        if (!self.page_flags.get().contains(flags)) | self.ptr.is_none() {
            return Err(MemoryFault::fault())
        }
        
        self.ptr.ok_or_else(MemoryFault::fault)
    }


    #[inline(always)]
    fn has_access(&self, flags: PageFlags) -> bool {
        self.get_data_ptr(flags).is_ok()
    }
    
    #[inline(always)]
    unsafe fn access(
        &self,
        offset: usize,
        len: usize,
        flags: PageFlags,
        mut op: impl FnMut(*const AtomicU8, usize),
    ) -> Result<(), MemoryFault> {
        let ptr = self.get_data_ptr(flags)?;
        unsafe {
            core::hint::assert_unchecked(len <= A::PAGE_SIZE.unchecked_sub(offset));
            core::hint::assert_unchecked(ptr.addr().get().is_multiple_of(A::PAGE_SIZE));
        }

        unsafe {
            let ptr = ptr.add(offset).as_ptr().cast_const();
            for i in 0..len {
                op(ptr.add(i), i)
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) unsafe fn load(&self, offset: usize, mem: &mut [MaybeUninit<u8>]) -> Result<(), MemoryFault> {
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
    pub(crate) unsafe fn store(&self, offset: usize, mem: &[u8]) -> Result<(), MemoryFault> {
        unsafe {
            let mem_ptr = mem.as_ptr();
            self.access(
                offset,
                mem.len(),
                PageFlags::READ,
                move |ptr, i| {
                    let value = std::ptr::read(mem_ptr.add(i));
                    (*ptr).store(value, Ordering::Relaxed);
                }
            )?;

            self.page_flags.update(|flags| flags | PageFlags::DIRTY);

            Ok(())
        }
    }
}

macro_rules! impl_load_ops {
    {
        bits: $bits: tt,
        ty: $ty: ty,
        load_function: $aligned_load_name: ident,
        store_function: $aligned_store_name: ident,
        load: $load_name: ident,
        fetch: $fetch_name: ident,
        store: $store_name: ident $(,)?
    } => {
        impl<A: VAddr> Page<A> {
            /// # Safety
            ///
            #[doc = concat!("`offset` must be propperly aligned to (", stringify!($bits), "/8) bytes")]
            #[inline(always)]
            pub(crate) unsafe fn $load_name(&self, offset: usize) -> Result<$ty, MemoryFault> {
                unsafe { core::hint::assert_unchecked(offset.is_multiple_of(size_of::<$ty>())) }

                let ptr = self.get_data_ptr(PageFlags::READ)?;

                let value = unsafe { tear_mem_ops::$aligned_load_name(ptr.as_ptr().add(offset)) };

                Ok(value)
            }

            /// # Safety
            ///
            #[doc = concat!("`offset` must be propperly aligned to (", stringify!($bits), "/8) bytes")]
            #[inline(always)]
            #[allow(dead_code)]
            pub(crate) unsafe fn $fetch_name(&self, offset: usize) -> Result<$ty, MemoryFault> {
                unsafe { core::hint::assert_unchecked(offset.is_multiple_of(size_of::<$ty>())) }

                let ptr = self.get_data_ptr(PageFlags::EXECUTE)?;

                let value = unsafe { tear_mem_ops::$aligned_load_name(ptr.as_ptr().add(offset)) };

                Ok(value)
            }

            /// # Safety
            ///
            #[doc = concat!("`offset` must be propperly aligned to (", stringify!($bits), "/8) bytes")]
            #[inline(always)]
            pub(crate) unsafe fn $store_name(&self, offset: usize, value: $ty) -> Result<(), MemoryFault> {
                unsafe { core::hint::assert_unchecked(offset.is_multiple_of(size_of::<$ty>())) }

                let ptr = self.get_data_ptr(PageFlags::WRITE)?;

                unsafe { tear_mem_ops::$aligned_store_name(ptr.as_ptr().add(offset), value) }

                self.page_flags.update(|flags| flags | PageFlags::DIRTY);

                Ok(())
            }
        }
    };

    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            $(impl_load_ops! {
                bits: $bits,
                ty: [<u $bits>],
                load_function: [<load_le_ $bits _aligned>],
                store_function: [<store_le_ $bits _aligned>],
                load: [<load $bits>],
                fetch: [<fetch $bits>],
                store: [<store $bits>]
            })+
        }
    };
}

impl_load_ops! { 64, 32, 16, 8 }


/// Page-mapped virtual memory view over host-backed storage.
///
/// `MMU<A>` maps virtual addresses of type `A` onto page-aligned host memory.
/// Access permissions are checked per page, and invalid access returns
/// [`MemoryFault`].
///
/// The implementation permits concurrent access and models tearing behavior
/// explicitly for multi-byte operations.
pub struct MMU<VAddr> {
    pages: Vec<Page<VAddr>>,
}

// Safety: all interior mutability is guareded explicitly with mutable references;
//         and all access is atomic/tearing and doesn't lead to UB
unsafe impl<A> Send for MMU<A> {}
unsafe impl<A> Sync for MMU<A> {}

impl<A: VAddr> MMU<A> {
    /// Creates an empty MMU with no mapped pages.
    ///
    /// All accesses fault until memory is mapped with [`MMU::map_memory`].
    pub fn new() -> Self {
        Self {
            pages: vec![]
        }
    }


    /// Maps a host memory region into the MMU page table.
    ///
    /// `base` is the starting virtual address, `ptr` is the backing host pointer,
    /// and `size` is the mapping size in bytes.
    ///
    /// Requirements:
    /// - `base` must be page-aligned,
    /// - `size` must be page-aligned,
    /// - `ptr` must be aligned to the page size,
    /// - `base + size` must not overflow.
    ///
    /// Permissions are applied to every mapped page in the region.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr .. ptr + size` is valid for the lifetime of Self,
    /// - the pointed-to memory can be safely treated as page-backed `AtomicU8`
    ///
    /// Returns [`MemoryFault`] if alignment or address validation fails.
    pub unsafe fn map_memory(
        &mut self,
        base: A,
        ptr: *const AtomicU8,
        size: A,
        protections: MemoryProtections,
    ) -> Result<(), MemoryFault> {
        let valid_mapping = base.add_addr(size).is_some()
            && base.is_page_aligned()
            && size.is_page_aligned()
            && ptr.addr().is_multiple_of(A::PAGE_SIZE);

        if !valid_mapping {
            return Err(MemoryFault::fault())
        }

        let end_vaddr = base
            .add_addr(size)
            .ok_or_else(MemoryFault::fault)?;

        // Safety: both base and size are page aligned, and if a mod x = 0 and b mod x = 0
        //         then (a + b) mod x = 0
        let end_page = unsafe { end_vaddr.div_page_size_unchecked() };

        let Some(end_page) = end_page.try_to_usize() else {
            panic!("could not map pages into view; out of host memory")
        };

        unsafe {
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

            let base_ptr = NonNull::new_unchecked(ptr.cast_mut());
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

        Ok(())
    }


    /// # Safety
    ///
    /// `vaddr_start` <= `vaddr_end`
    unsafe fn for_each_page_chunk(
        &self,
        vaddr_start: A,
        vaddr_end: A,
        required: PageFlags,
        mut f: impl FnMut(&Page<A>, usize, usize, usize),
    ) -> Result<(), MemoryFault> {
        unsafe { core::hint::assert_unchecked(vaddr_start <= vaddr_end) }

        if vaddr_start == vaddr_end {
            return Ok(())
        }

        let (end_page, end_offset) = vaddr_end.div_rem_page_size();
        let end_page = end_page.try_to_usize().ok_or_else(MemoryFault::fault)?;

        if self.pages.len() <= end_page {
            return Err(MemoryFault::fault())
        }

        let (start_page, start_offset) = vaddr_start.div_rem_page_size();
        // Safety: end_page fits in usize and end_page >= start_page
        let start_page = unsafe { start_page.try_to_usize().unwrap_unchecked() };

        // [----][----][----]
        //   ^            ^
        //   |____________|
        //   region to load

        let pages = unsafe { self.pages.get_unchecked(start_page..=end_page) };
        for page in pages {
            if !page.has_access(required) {
                return Err(MemoryFault::fault());
            }
        }

        let mut buf_offset = 0usize;
        for (i, page) in pages.iter().enumerate() {
            let page_idx = unsafe { start_page.unchecked_add(i) };

            let page_off = if page_idx == start_page { start_offset } else { 0 };
            let page_end = if page_idx == end_page { end_offset } else { A::PAGE_SIZE };
            let chunk_len = unsafe { page_end.unchecked_sub(page_off) };
            f(page, page_off, buf_offset, chunk_len);
            buf_offset = unsafe { buf_offset.unchecked_add(chunk_len) };
        }

        Ok(())
    }

    fn for_each_page_chunk_len(
        &self,
        vaddr: A,
        len: usize,
        required: PageFlags,
        f: impl FnMut(&Page<A>, usize, usize, usize),
    ) -> Result<(), MemoryFault> {
        let end = vaddr.add_offset(len).ok_or_else(MemoryFault::fault)?;
        // Safety: end is vaddr + len, with no overflow, and so this it must be bigger
        unsafe {
            self.for_each_page_chunk(
                vaddr,
                end,
                required,
                f
            )
        }
    }


    /// Loads a byte slice from virtual memory into `mem`.
    ///
    /// The load may span multiple pages. Every covered page must be mapped and have
    /// read permission.
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unreadable, or if address
    /// arithmetic overflows.
    pub fn load<'a>(
        &self,
        vaddr: A,
        mem: &'a mut [MaybeUninit<u8>]
    ) -> Result<&'a mut [u8], MemoryFault> {
        let result = self.for_each_page_chunk_len(
            vaddr,
            mem.len(),
            PageFlags::READ,
            |page, page_off, buf_off, chunk_len| unsafe {
                let dst = mem.get_unchecked_mut(buf_off..buf_off + chunk_len);
                page.load(page_off, dst).unwrap_unchecked();
            },
        );

        // Safety: mem has been filled
        result.map(|()| unsafe { mem.assume_init_mut() })
    }

    /// Stores a byte slice into virtual memory.
    ///
    /// The store may span multiple pages. Every covered page must be mapped and
    /// have write permission.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unwritable, or if address
    /// arithmetic overflows.
    #[inline(always)]
    pub fn store(&mut self, vaddr: A, mem: &[u8]) -> Result<(), MemoryFault> {
        self.for_each_page_chunk_len(
            vaddr,
            mem.len(),
            PageFlags::WRITE,
            |page, page_off, buf_off, chunk_len| unsafe {
                let src = mem.get_unchecked(buf_off..buf_off + chunk_len);
                page.store(page_off, src).unwrap_unchecked();
            },
        )
    }
}

impl<A: VAddr> MMU<A> {
    pub(crate) fn aligned_static_acces<const BYTES: u8>(&self, vaddr: A) -> Result<(&Page<A>, usize), MemoryFault> {
        const {
            assert!(BYTES.is_power_of_two());
            assert!(A::PAGE_SIZE.is_power_of_two());
            assert!(A::PAGE_SIZE.is_multiple_of(BYTES as usize));
        }

        if !vaddr.is_multiple_of(A::from(BYTES)) {
            return Err(MemoryFault::fault())
        }

        let (page, offset) = vaddr.div_rem_page_size();
        let page = page
            .try_to_usize()
            .and_then(|page_idx| self.pages.get(page_idx))
            .ok_or_else(MemoryFault::fault)?;

        Ok((page, offset))
    }
}


macro_rules! emit_load_store {
    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            $(#[allow(
                clippy::cast_possible_truncation,
                reason = "cast is checked to not overflow"
            )]
            const [<U $bits _BYTE_COUNT>]: u8 = {
                let size = size_of::< [<u $bits>] >();
                assert!(size <= u8::MAX as usize);
                size as u8
            };)+

            impl<A: VAddr> MMU<A> {$(
                // Safety:
                // offset is propperly aligned;
                // since the only time a page is returned is if the address was propperly aligned


                #[doc = concat!("Loads a little-endian `u", stringify!($bits), "` from virtual memory.")]
                ///
                #[doc = concat!(
                    "The address must be aligned to `size_of::<u",
                    stringify!($bits),
                    ">()`"
                )]
                ///
                /// Returns [`MemoryFault`] on misalignment, unmapped access, or permission
                /// failure.
                #[inline(always)]
                pub fn [<load $bits>](&mut self, vaddr: A) -> Result<[<u $bits>], MemoryFault> {
                    let (page, offset) = self.aligned_static_acces::<[<U $bits _BYTE_COUNT>]>(vaddr)?;
                    unsafe { page.[<load $bits>](offset) }
                }

                #[doc = concat!("Stores a little-endian `u", stringify!($bits), "` from virtual memory.")]
                ///
                #[doc = concat!(
                    "The address must be aligned to `size_of::<u",
                    stringify!($bits),
                    ">()`"
                )]
                ///
                /// Returns [`MemoryFault`] on misalignment, unmapped access, or permission failure.
                #[inline(always)]
                pub fn [<store $bits>](&mut self, vaddr: A, value: [<u $bits>]) -> Result<(), MemoryFault> {
                    let (page, offset) = self.aligned_static_acces::<[<U $bits _BYTE_COUNT>]>(vaddr)?;
                    unsafe { page.[<store $bits>](offset, value) }
                }
            )+}
        }
    };
}

emit_load_store! { 64, 32, 16, 8 }


impl<A: VAddr> MMU<A> {
    pub fn fetch(&self, vaddr: A) -> Result<A::InsnWord, MemoryFault> {
        A::fetch_insn_word(vaddr, self)
    }
}


impl<A: VAddr> Default for MMU<A> {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use std::alloc::Layout;
    use super::*;
    use std::mem::MaybeUninit;
    use std::ops::{Deref, DerefMut};
    use std::sync::atomic::AtomicU8;

    type Addr = u64;

    const PAGE_SIZE: usize = <Addr as crate::vaddr::sealed::VAddr>::PAGE_SIZE;

    fn addr(x: usize) -> Addr {
        Addr::try_from(x).expect("addr overflow")
    }

    struct BackingStorage {
        ptr: NonNull<AtomicU8>,
        len: usize,
    }

    impl Deref for BackingStorage {
        type Target = [AtomicU8];

        fn deref(&self) -> &Self::Target {
            unsafe {
                std::slice::from_raw_parts(
                    self.ptr.as_ptr(),
                    self.len
                )
            }
        }
    }

    impl DerefMut for BackingStorage {
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.ptr.as_ptr(),
                    self.len
                )
            }
        }
    }

    impl Drop for BackingStorage {
        fn drop(&mut self) {
            if self.len != 0 {
                unsafe {
                    std::alloc::dealloc(
                        self.ptr.as_ptr().cast(),
                        Layout::from_size_align_unchecked(self.len, PAGE_SIZE)
                    )
                }
            }
        }
    }

    fn new_backing(pages: usize) -> BackingStorage {
        let len = pages * PAGE_SIZE;
        let ptr = match len {
            0 => NonNull::dangling(),
            _ => {
                let layout = Layout::from_size_align(len, PAGE_SIZE).unwrap();
                match NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }) {
                    Some(ptr) => ptr.cast::<AtomicU8>(),
                    None => std::alloc::handle_alloc_error(layout),
                }
            }
        };

        BackingStorage { ptr, len }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn hash_index(i: usize) -> u8 {
        i.wrapping_mul(3) as u8
    }


    fn map_region(
        mmu: &mut MMU<Addr>,
        backing: &mut [AtomicU8],
        base: usize,
        size: usize,
        prot: MemoryProtections,
    ) {
        assert_eq!(base % PAGE_SIZE, 0, "test base must be page aligned");
        assert_eq!(size % PAGE_SIZE, 0, "test size must be page aligned");

        unsafe {
            mmu.map_memory(
                addr(base),
                backing.as_ptr(),
                addr(size),
                prot,
            )
                .expect("map_memory should succeed");
        }
    }

    #[test]
    fn unmapped_load_faults() {
        let mmu = MMU::<Addr>::new();
        let mut buf = [MaybeUninit::<u8>::uninit(); 4];

        let err = mmu.load(addr(0), &mut buf).unwrap_err();
        let _ = err;
    }

    #[test]
    fn unmapped_store_faults() {
        let mut mmu = MMU::<Addr>::new();
        let err = mmu.store(addr(0), &[1, 2, 3, 4]).unwrap_err();
        let _ = err;
    }

    #[test]
    fn map_and_load_single_page() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        for (i, byte) in backing.iter_mut().enumerate() {
            *byte.get_mut() = hash_index(i);
        }

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ,
        );

        let mut buf = [MaybeUninit::<u8>::uninit(); 16];
        let got = mmu.load(addr(8), &mut buf).expect("load should succeed");

        let expected: Vec<u8> = (8..24).map(hash_index).collect();
        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn load_across_page_boundary() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(2);

        for (i, byte) in backing.iter_mut().enumerate() {
            byte.store(hash_index(i), Ordering::Relaxed);
        }

        map_region(
            &mut mmu,
            &mut backing,
            0,
            2 * PAGE_SIZE,
            MemoryProtections::READ,
        );

        let start = PAGE_SIZE - 8;
        let mut buf = [MaybeUninit::<u8>::uninit(); 16];
        let got = mmu.load(addr(start), &mut buf).expect("cross-page load should succeed");

        let expected: Vec<u8> = (start..start + 16).map(hash_index).collect();
        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn store_across_page_boundary() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(2);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            2 * PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        let start = PAGE_SIZE - 4;
        let data = [10, 20, 30, 40, 50, 60, 70, 80];

        mmu.store(addr(start), &data)
            .expect("cross-page store should succeed");

        for (i, expected) in data.into_iter().enumerate() {
            let got = backing[start + i].load(Ordering::Relaxed);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn read_permission_is_enforced() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::WRITE,
        );

        let mut buf = [MaybeUninit::<u8>::uninit(); 1];
        let err = mmu.load(addr(0), &mut buf).unwrap_err();
        let _ = err;
    }

    #[test]
    fn write_permission_is_enforced() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ,
        );

        let err = mmu.store(addr(0), &[0xAA]).unwrap_err();
        let _ = err;
    }

    #[test]
    fn scalar_load_store_8() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        mmu.store8(addr(7), 0xAB).expect("store8 should succeed");
        let got = mmu.load8(addr(7)).expect("load8 should succeed");

        assert_eq!(got, 0xAB);
    }

    #[test]
    fn scalar_load_store_16() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        mmu.store16(addr(6), 0xBEEF).expect("store16 should succeed");
        let got = mmu.load16(addr(6)).expect("load16 should succeed");

        assert_eq!(got, 0xBEEF);
    }

    #[test]
    fn scalar_load_store_32() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        mmu.store32(addr(8), 0xDEADBEEF)
            .expect("store32 should succeed");
        let got = mmu.load32(addr(8)).expect("load32 should succeed");

        assert_eq!(got, 0xDEADBEEF);
    }

    #[test]
    fn scalar_load_store_64() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        mmu.store64(addr(16), 0x0123_4567_89AB_CDEF)
            .expect("store64 should succeed");
        let got = mmu.load64(addr(16)).expect("load64 should succeed");

        assert_eq!(got, 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn scalar_alignment_is_enforced() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        assert!(mmu.load16(addr(1)).is_err());
        assert!(mmu.load32(addr(2)).is_err());
        assert!(mmu.load64(addr(4)).is_err());

        assert!(mmu.store16(addr(1), 1).is_err());
        assert!(mmu.store32(addr(2), 1).is_err());
        assert!(mmu.store64(addr(4), 1).is_err());
    }

    #[test]
    fn scalar_access_must_not_cross_page() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(2);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            2 * PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        let off16 = PAGE_SIZE - 1;
        let off32 = PAGE_SIZE - 2;
        let off64 = PAGE_SIZE - 4;

        assert!(mmu.load16(addr(off16)).is_err());
        assert!(mmu.load32(addr(off32)).is_err());
        assert!(mmu.load64(addr(off64)).is_err());

        assert!(mmu.store16(addr(off16), 1).is_err());
        assert!(mmu.store32(addr(off32), 1).is_err());
        assert!(mmu.store64(addr(off64), 1).is_err());
    }

    #[test]
    fn byte_range_access_must_fault_when_range_runs_off_mapped_pages() {
        let mut mmu = MMU::<Addr>::new();
        let mut backing = new_backing(1);

        map_region(
            &mut mmu,
            &mut backing,
            0,
            PAGE_SIZE,
            MemoryProtections::READ | MemoryProtections::WRITE,
        );

        let start = PAGE_SIZE - 4;
        let mut buf = [MaybeUninit::<u8>::uninit(); 8];
        assert!(mmu.load(addr(start), &mut buf).is_err());
        assert!(mmu.store(addr(start), &[1, 2, 3, 4, 5, 6, 7, 8]).is_err());
    }

    #[test]
    fn zero_length_load_is_ok() {
        let mmu = MMU::<Addr>::new();
        let got = mmu.load(addr(0), &mut []).expect("zero-length load should succeed");
        assert!(got.is_empty());
    }

    #[test]
    fn zero_length_store_is_ok() {
        let mut mmu = MMU::<Addr>::new();
        mmu.store(addr(0), &[]).expect("zero-length store should succeed");
    }

    #[test]
    fn map_memory_rejects_unaligned_inputs() {
        let mut mmu = MMU::<Addr>::new();
        let backing = new_backing(2);

        unsafe {
            assert!(mmu
                .map_memory(
                    addr(1),
                    backing.as_ptr(),
                    addr(PAGE_SIZE),
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    addr(0),
                    backing.as_ptr(),
                    addr(PAGE_SIZE - 1),
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    addr(0),
                    backing.as_ptr().byte_add(1),
                    addr(PAGE_SIZE),
                    MemoryProtections::READ,
                )
                .is_err());
        }
    }
}