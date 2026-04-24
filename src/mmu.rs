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
//! - Scalar load/store helpers are provided for 8/16/32/64-bit little-endian
//!   accesses.
//! - Scalar accesses may cross page boundaries when both pages are mapped with
//!   the required permissions.
//!
//! Concurrency model:
//! - Backing memory may be accessed concurrently through multiple threads.
//! - Concurrent loads and stores through the MMU are allowed.
//! - Public safe APIs must not produce undefined behavior, even when accesses
//!   race.
//! - Atomicity follows the single-copy atomicity guarantees of the active ARM
//!   CPU profile:
//!   - on 32-bit ARM: naturally aligned 8-, 16-, and 32-bit operations are
//!     single-copy atomic;
//!   - on 64-bit ARM: naturally aligned 8-, 16-, 32-, and 64-bit operations are
//!     single-copy atomic.
//! - Operations outside the CPU's single-copy atomic width, or operations split
//!   across pages, may be observed as multiple smaller operations.
//!
//! Safety model:
//! - All public safe functions are required to be UB-free.
//! - Unsafe functions may rely on their documented caller obligations.
//! - Mapping requires the caller to provide valid, page-aligned backing memory
//!   for the lifetime of the MMU mapping.
//! - Once memory is mapped into an MMU, the backing pointer must not be accessed
//!   directly while the mapping is alive, except for use as backing memory for an
//!   MMU mapping under the same aliasing/concurrency rules that means, MMUs
//!   must use the same bit width, so MMU<u64> and MMU<u32> can't share pages.
//!
//! Typical usage:
//! 1. Construct an [`MMU`].
//! 2. Map one or more host memory regions with [`MMU::map_memory`].
//! 3. Access byte ranges through [`MMU::load`] / [`MMU::store`].
//! 4. Access scalars through `load_byte/load16/load32/load64` and
//!    `store_byte/store16/store32/store64`.


use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};
use crate::vaddr::VAddr;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct PageFlags: u8 {
        const READ       = 0b0001;
        const WRITE      = 0b0010;
        const EXECUTE    = 0b0100;
        const INSN_DIRTY = 0b1000;
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
/// - violates page permissions,
/// - overflows the virtual address range,
/// - fails a required alignment check,
/// - crosses into an unmapped or insufficiently-permitted page,
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

macro_rules! ensure {
    ($($expr: expr),+ $(,)?) => {
        if !($({ $expr })&+) {
            return Err(MemoryFault::fault())
        }
    };
}


impl MemoryProtections {
    // Self is a superset of PageFlags
    pub(crate) fn into_page_flags(self) -> PageFlags {
        PageFlags::from_bits_retain((self & Self::all()).bits())
    }
}


// invariant: if any of the memory protection bits in page_flags is on, then ptr MUST be Some(...)
pub(crate) struct Page<VAddr> {
    ptr: Option<NonNull<AtomicU8>>,
    page_flags: AtomicU8,
    _addr_ty: PhantomData<VAddr>
}

impl<A: VAddr> Page<A> {
    #[allow(clippy::declare_interior_mutable_const)]
    pub(crate) const UNMAPPED: Self = Self {
        ptr: None,
        page_flags: AtomicU8::new(0),
        _addr_ty: PhantomData,
    };

    pub(crate) fn mapped(
        ptr: NonNull<AtomicU8>,
        memory_protections: MemoryProtections,
    ) -> Self {
        let page_flags = memory_protections.into_page_flags().bits();
        Self {
            ptr: Some(ptr),
            page_flags: AtomicU8::new(page_flags),
            _addr_ty: PhantomData,
        }
    }

    pub(crate) fn unmap(&mut self) {
        *self = Self::UNMAPPED
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.ptr.is_some()
    }

    // Safety: self must be mapped
    pub(crate) unsafe fn mem_protect(&mut self, memory_protections: MemoryProtections) {
        debug_assert!(self.is_mapped());
        *self.page_flags.get_mut() = memory_protections.into_page_flags().bits();
    }

    #[inline(always)]
    pub(crate) fn load_flags(&self) -> PageFlags {
        PageFlags::from_bits_retain(self.page_flags.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub(crate) fn set_insn_dirty(&self) {
        let initial_flags = self.load_flags();
        if initial_flags.contains(PageFlags::EXECUTE) {
            cold_path();
            if !initial_flags.contains(PageFlags::INSN_DIRTY) {
                cold_path();
                self.page_flags.fetch_or(PageFlags::INSN_DIRTY.bits(), Ordering::Acquire);
            }
        }
    }

    #[inline(always)]
    fn has_access(&self, flags: PageFlags) -> bool {
        self.load_flags().contains(flags)
    }

    #[inline(always)]
    pub(crate) unsafe fn get_data_ptr_unchecked(&self) -> NonNull<AtomicU8> {
        debug_assert!(self.ptr.is_some());

        unsafe { self.ptr.unwrap_unchecked() }
    }

    #[inline(always)]
    pub(crate) fn get_data_ptr(&self, flags: PageFlags) -> Result<NonNull<AtomicU8>, MemoryFault> {
        ensure!(self.has_access(flags));
        // Safety: if self.has_access returns true, its always ok to call get_data_ptr_unchecked
        Ok(unsafe { self.get_data_ptr_unchecked() })
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
                    let value = A::load_byte(ptr);
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
                PageFlags::WRITE,
                move |ptr, i| {
                    let value = std::ptr::read(mem_ptr.add(i));
                    A::store_byte(ptr, value)
                }
            )?;

            self.set_insn_dirty();

            Ok(())
        }
    }
}

macro_rules! impl_load_ops {
    {
        $(bits: $bits: tt,
        ty: $ty: ty,
        load_function: $load_op_name: ident,
        store_function: $store_op_name: ident,
        load: $load_name: ident,
        fetch: $fetch_name: ident,
        store: $store_name: ident
        ),+
        $(,)?
    } => {
        impl<A: VAddr> Page<A> {$(
            /// # Safety
            ///
            #[doc = concat!("`offset` must be <= A::PAGE_SIZE - size_of<", stringify!($ty), ">()")]
            #[inline(always)]
            pub(crate) unsafe fn $load_name(&self, offset: usize) -> Result<$ty, MemoryFault> {
                let ptr = self.get_data_ptr(PageFlags::READ)?;
                let value = unsafe { A::$load_op_name(ptr.as_ptr().add(offset)) };
                Ok(value)
            }

            /// # Safety
            ///
            #[doc = concat!("`offset` must be <= A::PAGE_SIZE - size_of<", stringify!($ty), ">()")]
            #[inline(always)]
            pub(crate) unsafe fn $store_name(&self, offset: usize, value: $ty) -> Result<(), MemoryFault> {
                let ptr = self.get_data_ptr(PageFlags::WRITE)?;

                unsafe { A::$store_op_name(ptr.as_ptr().add(offset), value) }

                self.set_insn_dirty();

                Ok(())
            }
        )+}
    };

    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            impl_load_ops! {$(
                bits: $bits,
                ty: [<u $bits>],
                load_function: [<load $bits _le>],
                store_function: [<store $bits _le>],
                load: [<load $bits>],
                fetch: [<fetch $bits>],
                store: [<store $bits>]
            ),+}
        }
    };
}

impl_load_ops! { 64, 32, 16 }

impl_load_ops! {
    bits: 8,
    ty: u8,
    load_function: load_byte,
    store_function: store_byte,
    load: load_byte,
    fetch: fetch_byte,
    store: store_byte
}

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

// Safety: all interior mutability is guarded explicitly with mutable references;
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
    /// - `ptr .. ptr + size` is valid for the lifetime of this MMU mapping,
    /// - the pointed-to memory is initialized,
    /// - the pointed-to memory is valid for both reads and writes at the host-memory
    ///   level, regardless of the guest permissions applied by `protections`,
    /// - the backing memory is not accessed directly while this mapping is alive,
    ///   except by other MMU mappings that obey the same aliasing and concurrency
    ///   rules,
    /// - the backing memory remains page-aligned and is not deallocated, reallocated,
    ///   or otherwise invalidated while mapped.
    ///
    /// Guest read/write/execute permissions are enforced by the MMU. They do not
    /// relax the host-memory validity requirements above.
    ///
    /// Returns [`MemoryFault`] if alignment or address validation fails.
    pub unsafe fn map_memory(
        &mut self,
        base: A,
        ptr: *mut u8,
        size: A,
        protections: MemoryProtections,
    ) -> Result<(), MemoryFault> {
        ensure!(
            base.add_addr(size).is_some(),
            base.is_page_aligned(),
            size.is_page_aligned(),
            ptr.addr().is_multiple_of(A::PAGE_SIZE)
        );

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
                self.pages.resize_with(end_page, || Page::UNMAPPED)
            }

            let base_ptr = NonNull::new_unchecked(ptr.cast::<AtomicU8>());
            for page_idx in start_page..end_page {
                let page = self.pages.get_unchecked_mut(page_idx);
                let backing_page_idx = page_idx.unchecked_sub(start_page);
                *page = Page::mapped(
                    base_ptr.add(backing_page_idx.unchecked_mul(A::PAGE_SIZE)),
                    protections
                );
            }
        }

        Ok(())
    }

    fn get_pages_mut(&mut self, start: A, size: A) -> Result<&mut [Page<A>], MemoryFault> {
        let end = start.add_addr(size).ok_or_else(MemoryFault::fault)?;
        let (end_page, end_remainder) = end.div_rem_page_size();
        let (start_page, start_remainder) = start.div_rem_page_size();

        ensure!(end_remainder == 0, start_remainder == 0);

        let (start_page, end_page) = start_page
            .try_to_usize()
            .and_then(|start_page| Some((start_page, end_page.try_to_usize()?)))
            .ok_or_else(MemoryFault::fault)?;


        self.pages.get_mut(start_page..end_page).ok_or_else(MemoryFault::fault)
    }


    pub fn unmap_memory(&mut self, start: A, size: A) -> Result<(), MemoryFault> {
        for page in self.get_pages_mut(start, size)? {
            page.unmap()
        }
        Ok(())
    }

    pub fn mem_protect(
        &mut self,
        start: A,
        size: A,
        protections: MemoryProtections
    ) -> Result<(), MemoryFault> {
        let pages = self.get_pages_mut(start, size)?;
        for page in &mut *pages {
            ensure!(page.is_mapped());
        }

        // Safety: all pages are mapped
        for page in pages {
            unsafe { page.mem_protect(protections) }
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

        let (end_page, end_offset) = vaddr_end.div_rem_page_size();
        let end_page = end_page.try_to_usize().ok_or_else(MemoryFault::fault)?;

        ensure!(end_page < self.pages.len());

        let (start_page, start_offset) = vaddr_start.div_rem_page_size();
        // Safety: end_page fits in usize and end_page >= start_page
        let start_page = unsafe { start_page.try_to_usize().unwrap_unchecked() };

        let pages = unsafe { self.pages.get_unchecked(start_page..=end_page) };
        for page in pages {
            ensure!(page.has_access(required))
        }

        let mut buf_offset = 0usize;
        for (i, page) in pages.iter().enumerate() {
            let page_idx = unsafe { start_page.unchecked_add(i) };

            let page_off = if page_idx == start_page { start_offset } else { 0 };
            let page_end = if page_idx == end_page {
                // end_offset is < A::PAGE_SIZE
                // which is some usize, that means there is some usize bigger than us
                // so this can be incremented safely
                unsafe { end_offset.unchecked_add(1) }
            } else {
                A::PAGE_SIZE
            };

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
        if len == 0 {
            return Ok(())
        }

        let extra = unsafe { len.unchecked_sub(1) };

        let end = vaddr.add_offset(extra).ok_or_else(MemoryFault::fault)?;
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
    /// Concurrent stores are allowed. The returned bytes may reflect a mixture of
    /// values from racing stores, according to the atomicity guarantees of the
    /// underlying target operations.
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unreadable, or if address
    /// arithmetic overflows.
    ///
    /// This safe function must not invoke undefined behavior.
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
                let range = buf_off..buf_off.unchecked_add(chunk_len);
                let dst = mem.get_unchecked_mut(range);
                page.load(page_off, dst).unwrap_unchecked();
            },
        );

        // Safety: mem has been filled
        result.map(|()| unsafe { mem.assume_init_mut() })
    }

    /// Stores a byte slice into virtual memory.
    ///
    /// The store may span multiple pages. Every covered page must be mapped and have
    /// write permission.
    ///
    /// Concurrent loads and stores are allowed. Other threads may observe the write
    /// as a sequence of byte or scalar operations according to the atomicity
    /// guarantees of the underlying target operations.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unwritable, or if address
    /// arithmetic overflows.
    ///
    /// This safe function must not invoke undefined behavior.
    #[inline(always)]
    pub fn store(&self, vaddr: A, mem: &[u8]) -> Result<(), MemoryFault> {
        self.for_each_page_chunk_len(
            vaddr,
            mem.len(),
            PageFlags::WRITE,
            |page, page_off, buf_off, chunk_len| unsafe {
                let range = buf_off..buf_off.unchecked_add(chunk_len);
                let src = mem.get_unchecked(range);
                page.store(page_off, src).unwrap_unchecked();
            },
        )
    }
}

struct SecondPage<'a, A: VAddr> {
    page: &'a Page<A>,
    overflow_amount: NonZero<u8>,
}

struct SmallAccess<'a, A: VAddr> {
    base_page: &'a Page<A>,
    base_page_offset: usize,
    second_page: Option<SecondPage<'a, A>>
}

impl<A: VAddr> MMU<A> {
    fn static_small_multibyte_acces<const BYTES: u8>(
        &self,
        vaddr: A
    ) -> Result<SmallAccess<'_, A>, MemoryFault> {
        const {
            assert!(BYTES > 1);
            assert!((BYTES as usize) < A::PAGE_SIZE);
            assert!(A::PAGE_SIZE.checked_add(BYTES as usize).is_some());
        }

        let (base_page_idx, base_page_offset) = vaddr.div_rem_page_size();
        let (base_page_idx, base_page) = base_page_idx
            .try_to_usize()
            .and_then(|page_idx| Some((page_idx, self.pages.get(page_idx)?)))
            .ok_or_else(MemoryFault::fault)?;

        // TODO safety comments

        let end_offset = unsafe { base_page_offset.unchecked_add(usize::from(BYTES)) };

        let second_page = match end_offset > A::PAGE_SIZE {
            false => None,
            true => {
                cold_path();
                let overflow_amount = unsafe {
                    u8::try_from(end_offset.unchecked_sub(A::PAGE_SIZE)).unwrap_unchecked()
                };

                unsafe { core::hint::assert_unchecked(overflow_amount < BYTES) }

                let overflow_amount = unsafe {
                    NonZero::new_unchecked(overflow_amount)
                };

                let second_page = base_page_idx
                    .checked_add(1)
                    .and_then(|second_page_idx| self.pages.get(second_page_idx))
                    .ok_or_else(MemoryFault::fault)?;

                Some(SecondPage {
                    page: second_page,
                    overflow_amount
                })
            }
        };

        Ok(SmallAccess {
            base_page,
            base_page_offset,
            second_page
        })
    }
}


macro_rules! emit_multi_word_load_store {
    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            impl<A: VAddr> MMU<A> {$(
                /// Loads a little-endian scalar from virtual memory.
                ///
                /// The access requires read permission for every page it touches. If the access
                /// crosses a page boundary, both pages must be mapped and readable.
                ///
                /// Atomicity follows the target ARM CPU's single-copy atomicity guarantees for
                /// naturally aligned scalar accesses of this width. Cross-page accesses may be
                /// implemented as multiple operations and should not be treated as a single
                /// atomic access.
                ///
                /// Returns [`MemoryFault`] on unmapped access, permission failure, or overflow.
                ///
                /// This safe function must not invoke undefined behavior.
                #[inline(always)]
                pub fn [<load $bits _le>](&self, vaddr: A) -> Result<[<u $bits>], MemoryFault> {
                    let access = self.static_small_multibyte_acces::<{ $bits / 8 }>(vaddr)?;
                    match access.second_page {
                        // SAFETY:
                        // `static_small_multibyte_acces` returned `None` for `second_page`, so
                        // this access is fully contained in `base_page`.
                        //
                        // Therefore:
                        //
                        //   base_page_offset + size_of::<u$bits>() <= A::PAGE_SIZE
                        //
                        // which satisfies the safety requirement of `Page::load$bits`.
                        None => unsafe { access.base_page.[<load $bits>](access.base_page_offset) },

                        Some(second_page) => unsafe {
                            cold_path();

                            // The access crosses the page boundary.
                            //
                            // `overflow_amount` is the number of bytes that must be read from
                            // the start of the second page. The remaining bytes come from the
                            // end of the base page.
                            //
                            // Instead of doing byte-by-byte loads, we load:
                            //
                            //   1. one aligned little-endian word ending at the end of the base page
                            //   2. one aligned little-endian word starting at the beginning of the second page
                            //
                            // Then we shift/or the two words to reconstruct the requested
                            // little-endian value.

                            let hi_page_ptr = second_page.page.get_data_ptr(PageFlags::READ)?;
                            let lo_page_ptr = access.base_page.get_data_ptr(PageFlags::READ)?;


                            // SAFETY:
                            // In the crossing case:
                            //
                            //   end_offset      = base_page_offset + BYTES
                            //   overflow_amount = end_offset - A::PAGE_SIZE
                            //
                            // Therefore:
                            //
                            //   base_page_offset - overflow_amount
                            // = base_page_offset - (base_page_offset + BYTES - A::PAGE_SIZE)
                            // = A::PAGE_SIZE - BYTES
                            //
                            // Since `static_small_multibyte_acces` guarantees
                            // `BYTES < A::PAGE_SIZE`, this subtraction cannot underflow.
                            //
                            // The result is the offset of a full `$bits`-wide word ending
                            // exactly at the end of the base page.
                            let lo_offset = access
                                .base_page_offset
                                .unchecked_sub(usize::from(second_page.overflow_amount.get()));

                            // SAFETY:
                            // `hi_page_ptr` points to the start of the second page's readable
                            // data. Offset `0` is valid for a full `$bits`-wide load because
                            // `BYTES == $bits / 8` and `BYTES < A::PAGE_SIZE`.
                            //
                            // The aligned operation is valid because page data is assumed to be
                            // aligned sufficiently for these aligned page-boundary loads.
                            let hi_ptr = hi_page_ptr.as_ptr();

                            // SAFETY:
                            // From the proof above:
                            //
                            //   lo_offset == A::PAGE_SIZE - BYTES
                            //
                            // so `lo_ptr` points to the first byte of the final `$bits`-wide
                            // word in the base page.
                            //
                            // This means the load is fully contained inside the base page and
                            // ends exactly at the page boundary.
                            //
                            // The aligned operation is valid because this pointer is page-end
                            // aligned for a `$bits`-wide word, assuming `A::PAGE_SIZE` is a
                            // multiple of `BYTES`.
                            let lo_ptr = lo_page_ptr.byte_add(lo_offset).as_ptr();


                            let hi = A::[<load $bits _le_aligned>](hi_ptr);
                            let lo = A::[<load $bits _le_aligned>](lo_ptr);

                            // SAFETY:
                            // `overflow_amount` is a `NonZero<u8>`, so it is at least `1`.
                            // `static_small_multibyte_acces` also guarantees
                            // `overflow_amount < BYTES`.
                            //
                            // Therefore:
                            //
                            //   0 < overflow_amount * 8 < $bits
                            //
                            // So multiplying by 8 cannot overflow `u8` for the supported
                            // widths, and the resulting bit offset is strictly less than the
                            // integer width.
                            let bit_offset = u32::from(
                                second_page.overflow_amount.get().unchecked_mul(8)
                            );

                            // SAFETY:
                            // Since `bit_offset < $bits`, this subtraction cannot underflow.
                            //
                            // Also, because `bit_offset > 0`, `hi_shift` is strictly less than
                            // `$bits`.
                            let hi_shift = ($bits as u32).unchecked_sub(bit_offset);
                            let lo_shift = bit_offset;

                            // SAFETY:
                            // Both shift amounts are in `1..$bits`, so neither unchecked shift
                            // uses an invalid shift amount.
                            //
                            // `lo >> lo_shift` discards the bytes before the requested virtual
                            // address in the base-page word.
                            //
                            // `hi << hi_shift` moves the bytes from the second page into the
                            // high end of the result.
                            //
                            // OR-ing both pieces reconstructs the requested little-endian
                            // `$bits` value spanning the two pages.
                            Ok(hi.unchecked_shl(hi_shift) | lo.unchecked_shr(lo_shift))
                        }
                    }
                }

                /// Stores a little-endian scalar into virtual memory.
                ///
                /// The access requires write permission for every page it touches. If the access
                /// crosses a page boundary, both pages must be mapped and writable.
                ///
                /// Atomicity follows the target ARM CPU's single-copy atomicity guarantees for
                /// naturally aligned scalar accesses of this width. Cross-page accesses may be
                /// implemented as multiple operations and should not be treated as a single
                /// atomic access.
                ///
                /// Returns [`MemoryFault`] on unmapped access, permission failure, overflow, or
                /// required alignment failure.
                ///
                /// This safe function must not invoke undefined behavior.
                #[inline(always)]
                pub fn [<store $bits _le>](&self, vaddr: A, value: [<u $bits>]) -> Result<(), MemoryFault> {
                    let access = self.static_small_multibyte_acces::<{ $bits / 8 }>(vaddr)?;

                    match access.second_page {
                        // SAFETY:
                        // `static_small_multibyte_acces` returned `None` for `second_page`, so
                        // this access is fully contained in `base_page`.
                        //
                        // Therefore:
                        //
                        //   base_page_offset + size_of::<u$bits>() <= A::PAGE_SIZE
                        //
                        // which satisfies the safety requirement of `Page::store$bits`.
                        None => unsafe {
                            access.base_page.[<store $bits>](access.base_page_offset, value)
                        },

                        Some(second_page) => unsafe {
                            // Note: we can't load 2 words and combine them like the load case
                            //       since that would alter/mess with the atomicity of the bytes
                            //       next to the value
                            let bytes = value.to_le_bytes();

                            let hi_page_ptr = second_page.page.get_data_ptr(PageFlags::WRITE)?;
                            let lo_page_ptr = access.base_page.get_data_ptr(PageFlags::WRITE)?;

                            let overflow = usize::from(second_page.overflow_amount.get());


                            let mut active_ptr = hi_page_ptr.add(overflow).as_ptr();
                            let mut i = bytes.len();
                            for _ in 0..overflow {
                                active_ptr = active_ptr.sub(1);
                                i = i.unchecked_sub(1);
                                let byte = *bytes.get_unchecked(i);
                                A::store_byte(active_ptr, byte)
                            }

                            active_ptr = lo_page_ptr
                                .add(const { A::PAGE_SIZE.strict_sub(1) })
                                .as_ptr();

                            loop {
                                i = i.unchecked_sub(1);
                                let byte = *bytes.get_unchecked(i);
                                A::store_byte(active_ptr, byte);

                                if i == 0 {
                                    break
                                }
                                active_ptr = active_ptr.sub(1)
                            }


                            access.base_page.set_insn_dirty();
                            second_page.page.set_insn_dirty();
                            Ok(())
                        }
                    }
                }
            )+}
        }
    };
}

emit_multi_word_load_store! { 64, 32, 16 }

impl<A: VAddr> MMU<A> {
    pub(crate) fn single_page_aligned_access<const ALIGN: u8>(
        &self,
        vaddr: A
    ) -> Result<(&Page<A>, usize), MemoryFault> {
        const {
            assert!(ALIGN.is_power_of_two());
            assert!(A::PAGE_SIZE.is_power_of_two());
            assert!(A::PAGE_SIZE.is_multiple_of(ALIGN as usize));
        }

        ensure!(vaddr.is_multiple_of(A::from(ALIGN)));

        let (page, offset) = vaddr.div_rem_page_size();
        let page = page
            .try_to_usize()
            .and_then(|page_idx| self.pages.get(page_idx))
            .ok_or_else(MemoryFault::fault)?;

        Ok((page, offset))
    }

    pub fn load_byte(&self, vaddr: A) -> Result<u8, MemoryFault> {
        const ALIGN: u8 = 1;
        unsafe {
            // Safety: no store is done to the page
            let (page, offset) = self.single_page_aligned_access::<ALIGN>(vaddr)?;
            // Safety: offset is the result of x % PAGE_SIZE and so must be smaller than page size
            page.load_byte(offset)
        }
    }

    pub fn store_byte(&self, vaddr: A, value: u8) -> Result<(), MemoryFault> {
        const ALIGN: u8 = 1;
        let (page, offset) = self.single_page_aligned_access::<ALIGN>(vaddr)?;
        // Safety: offset is the result of x % PAGE_SIZE and so must be smaller than page size
        unsafe { page.store_byte(offset, value) }
    }
}


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
mod mmu_tests {
    use super::*;

    use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    const BASE: u64 = 0;

    const PAGE_SIZE: usize = <u64 as crate::vaddr::sealed::VAddr>::PAGE_SIZE;

    fn page_size() -> usize {
        PAGE_SIZE
    }

    fn page_addr(page: usize) -> u64 {
        BASE.strict_add(u64::try_from(page.strict_mul(page_size())).unwrap())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn pattern_byte(i: usize) -> u8 {
        (i as u8).wrapping_mul(37).wrapping_add(0x51)
    }

    fn pattern_array<const N: usize>(start: usize) -> [u8; N] {
        std::array::from_fn(|i| pattern_byte(start.wrapping_add(i)))
    }

    struct PageBacking {
        ptr: NonNull<u8>,
        len: usize,
    }

    impl PageBacking {
        fn new(pages: usize) -> Self {
            let len = pages.checked_mul(page_size()).unwrap();
            let layout = Layout::from_size_align(len, page_size()).unwrap();

            let raw = match len{
                0 => core::ptr::dangling_mut(),
                _ => unsafe { alloc_zeroed(layout) }
            };
            let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));

            Self { ptr, len }
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            self.ptr.as_ptr()
        }

        fn get_page(&mut self, page: usize) -> &mut [u8; PAGE_SIZE] {
            let index = page.strict_mul(page_size());
            assert!(index < self.len);
            unsafe { &mut *self.ptr.as_ptr().add(page.strict_mul(page_size())).cast() }
        }

        unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    impl Drop for PageBacking {
        fn drop(&mut self) {
            if self.len != 0 {
                unsafe {
                    dealloc(
                        self.ptr.as_ptr(),
                        Layout::from_size_align_unchecked(self.len, PAGE_SIZE)
                    );
                }
            }
        }
    }

    struct Fixture {
        mmu: MMU<u64>,
        _backing: PageBacking,
    }

    impl Fixture {
        fn new(pages: usize, protections: MemoryProtections) -> Self {
            Self::with_bytes(pages, protections, |_| 0)
        }

        fn with_bytes(
            pages: usize,
            protections: MemoryProtections,
            mut byte: impl FnMut(usize) -> u8,
        ) -> Self {
            let mut backing = PageBacking::new(pages);

            unsafe {
                for (i, dst) in backing.as_mut_slice().iter_mut().enumerate() {
                    *dst = byte(i);
                }
            }

            let mut mmu = MMU::<u64>::new();
            unsafe {
                mmu.map_memory(
                    BASE,
                    backing.as_mut_ptr(),
                    pages.strict_mul(page_size()) as u64,
                    protections,
                )
                    .unwrap();
            }

            Self {
                mmu,
                _backing: backing,
            }
        }

        fn with_page_protections(protections: &[MemoryProtections]) -> Self {
            Self::with_page_protections_and_bytes(protections, |_| 0)
        }

        fn with_page_protections_and_bytes(
            protections: &[MemoryProtections],
            mut byte: impl FnMut(usize) -> u8,
        ) -> Self {
            assert!(!protections.is_empty());

            let mut backing = PageBacking::new(protections.len());

            unsafe {
                for (i, dst) in backing.as_mut_slice().iter_mut().enumerate() {
                    *dst = byte(i);
                }
            }

            let mut mmu = MMU::<u64>::new();
            for (page, protections) in protections.iter().copied().enumerate() {
                unsafe {
                    mmu.map_memory(
                        page_addr(page),
                        backing.get_page(page).as_mut_ptr(),
                        page_size() as u64,
                        protections,
                    ).unwrap();
                }
            }

            Self {
                mmu,
                _backing: backing,
            }
        }

        fn read_vec(&self, vaddr: u64, len: usize) -> Vec<u8> {
            let mut out = vec![MaybeUninit::<u8>::uninit(); len];
            self.mmu.load(vaddr, &mut out).unwrap().to_vec()
        }

        fn flags(&self, page: usize) -> PageFlags {
            self.mmu.pages[page].load_flags()
        }

        fn is_dirty(&self, page: usize) -> bool {
            self.flags(page).contains(PageFlags::INSN_DIRTY)
        }
    }

    #[test]
    fn new_mmu_faults_non_empty_accesses() {
        let mmu = MMU::<u64>::new();

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(mmu.load(BASE, &mut one).is_err());
        assert!(mmu.store(BASE, &[1]).is_err());

        assert!(mmu.load_byte(BASE).is_err());
        assert!(mmu.store_byte(BASE, 1).is_err());

        assert!(mmu.load16_le(BASE).is_err());
        assert!(mmu.load32_le(BASE).is_err());
        assert!(mmu.load64_le(BASE).is_err());

        assert!(mmu.store16_le(BASE, 0x1234).is_err());
        assert!(mmu.store32_le(BASE, 0x1234_5678).is_err());
        assert!(mmu.store64_le(BASE, 0x1234_5678_9abc_def0).is_err());
    }

    #[test]
    fn zero_length_load_and_store_do_not_require_mapping() {
        let mmu = MMU::<u64>::new();

        let mut empty: [MaybeUninit<u8>; 0] = [];

        let loaded = mmu.load(0x1234_5678, &mut empty).unwrap();
        assert!(loaded.is_empty());

        assert!(mmu.store(0x1234_5678, &[]).is_ok());
    }

    #[test]
    fn map_memory_rejects_unaligned_base_size_ptr_and_overflow() {
        let mut backing = PageBacking::new(2);
        let mut mmu = MMU::<u64>::new();

        unsafe {
            assert!(mmu
                .map_memory(
                    1,
                    backing.as_mut_ptr(),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    BASE,
                    backing.get_page(0).as_mut_ptr().add(1),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    BASE,
                    backing.as_mut_ptr(),
                    (page_size() - 1) as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            let overflow_base = u64::MAX - ((page_size() as u64) - 1);

            assert!(mmu
                .map_memory(
                    overflow_base,
                    backing.as_mut_ptr(),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());
        }
    }

    #[test]
    fn nonzero_base_maps_only_requested_page() {
        let mut backing = PageBacking::new(1);
        unsafe {
            backing.as_mut_slice()[0] = 0xaa;
        }

        let mut mmu = MMU::<u64>::new();
        let base = page_addr(2);

        unsafe {
            mmu.map_memory(
                base,
                backing.as_mut_ptr(),
                page_size() as u64,
                MemoryProtections::READ,
            )
                .unwrap();
        }

        assert!(mmu.load_byte(0).is_err());
        assert!(mmu.load_byte(page_addr(1)).is_err());
        assert_eq!(mmu.load_byte(base).unwrap(), 0xaa);
    }

    #[test]
    fn read_only_page_allows_loads_and_rejects_stores() {
        let fixture = Fixture::new(1, MemoryProtections::READ);

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0);

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_ok());

        assert!(fixture.mmu.store_byte(BASE, 1).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
        assert!(fixture.mmu.store32_le(BASE, 0xfeed_beef).is_err());
        assert!(fixture.mmu.store64_le(BASE, 0xfeed_beef_dead_cafe).is_err());
    }

    #[test]
    fn write_only_page_allows_stores_and_rejects_loads() {
        let fixture = Fixture::new(1, MemoryProtections::WRITE);

        assert!(fixture.mmu.store_byte(BASE, 1).is_ok());
        assert!(fixture.mmu.store(BASE, &[1, 2, 3]).is_ok());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_ok());

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.load32_le(BASE).is_err());
        assert!(fixture.mmu.load64_le(BASE).is_err());
    }

    #[test]
    fn execute_only_page_rejects_data_loads_and_stores() {
        let fixture = Fixture::new(1, MemoryProtections::EXECUTE);

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 1).is_err());

        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn empty_protection_mapping_rejects_all_data_accesses() {
        let fixture = Fixture::new(1, MemoryProtections::empty());

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 1).is_err());
    }

    #[test]
    fn byte_load_store_roundtrip() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        for i in 0..64_u8 {
            fixture.mmu.store_byte(i.into(), pattern_byte(i.into())).unwrap();
        }

        for i in 0..64_u8 {
            assert_eq!(
                fixture.mmu.load_byte(i.into()).unwrap(),
                pattern_byte(i.into())
            );
        }
    }

    #[test]
    fn slice_store_load_roundtrip_inside_one_page() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let data = pattern_array::<128>(0);
        let start = 17u64;

        fixture.mmu.store(start, &data).unwrap();

        assert_eq!(fixture.read_vec(start, data.len()), data);
    }

    #[test]
    fn slice_store_load_roundtrip_across_two_pages() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE
        );

        let start = u64::try_from(page_size().strict_sub(5)).unwrap();
        let data = pattern_array::<13>(0);

        fixture.mmu.store(start, &data).unwrap();

        assert_eq!(fixture.read_vec(start, data.len()), data);
    }

    #[test]
    fn slice_store_across_boundary_requires_write_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ | MemoryProtections::WRITE,
            MemoryProtections::READ,
        ]);

        let start = (page_size() - 1) as u64;

        assert!(fixture.mmu.store(start, &[1, 2]).is_err());
    }

    #[test]
    fn slice_load_across_boundary_requires_read_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ | MemoryProtections::WRITE,
            MemoryProtections::WRITE,
        ]);

        let start = (page_size() - 1) as u64;
        let mut out = [MaybeUninit::<u8>::uninit(); 2];

        assert!(fixture.mmu.load(start, &mut out).is_err());
    }

    #[test]
    fn failed_slice_store_across_unmapped_page_does_not_partially_write() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let last = (page_size() - 1) as u64;
        fixture.mmu.store_byte(last, 0xaa).unwrap();

        assert!(fixture.mmu.store(last, &[0xbb, 0xcc]).is_err());

        assert_eq!(fixture.mmu.load_byte(last).unwrap(), 0xaa);
    }

    #[test]
    fn scalar_loads_inside_one_page_are_little_endian() {
        let fixture = Fixture::with_bytes(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let off16 = 11usize;
        let off32 = 19usize;
        let off64 = 29usize;

        assert_eq!(
            fixture.mmu.load16_le(off16 as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(off16))
        );

        assert_eq!(
            fixture.mmu.load32_le(off32 as u64).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(off32))
        );

        assert_eq!(
            fixture.mmu.load64_le(off64 as u64).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(off64))
        );
    }

    #[test]
    fn scalar_stores_inside_one_page_are_little_endian() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        fixture.mmu.store16_le(11, 0xbeef).unwrap();
        assert_eq!(fixture.read_vec(11, 2), 0xbeefu16.to_le_bytes().to_vec());

        fixture.mmu.store32_le(19, 0xaabb_ccdd).unwrap();
        assert_eq!(
            fixture.read_vec(19, 4),
            0xaabb_ccddu32.to_le_bytes().to_vec()
        );

        fixture.mmu.store64_le(29, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(
            fixture.read_vec(29, 8),
            0x0123_4567_89ab_cdefu64.to_le_bytes().to_vec()
        );
    }

    #[test]
    fn scalar_loads_at_last_non_crossing_offsets_succeed() {
        let fixture = Fixture::with_bytes(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let p = page_size();

        assert_eq!(
            fixture.mmu.load16_le((p - 2) as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(p - 2))
        );

        assert_eq!(
            fixture.mmu.load32_le((p - 4) as u64).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(p - 4))
        );

        assert_eq!(
            fixture.mmu.load64_le((p - 8) as u64).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(p - 8))
        );
    }

    #[test]
    fn scalar_loads_crossing_page_boundary_read_expected_bytes() {
        let fixture = Fixture::with_bytes(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let p = page_size();

        assert_eq!(
            fixture.mmu.load16_le((p - 1) as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(p - 1))
        );

        for bytes_in_first_page in 1..4 {
            let start = p - bytes_in_first_page;
            assert_eq!(
                fixture.mmu.load32_le(start as u64).unwrap(),
                u32::from_le_bytes(pattern_array::<4>(start)),
                "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }

        for bytes_in_first_page in 1..8 {
            let start = p - bytes_in_first_page;
            assert_eq!(
                fixture.mmu.load64_le(start as u64).unwrap(),
                u64::from_le_bytes(pattern_array::<8>(start)),
                "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }
    }

    #[test]
    fn scalar_stores_crossing_page_boundary_write_expected_bytes() {
        let fixture = Fixture::new(2, MemoryProtections::READ | MemoryProtections::WRITE);
        let p = page_size();

        let value16 = 0xbeefu16;
        fixture.mmu.store16_le((p - 1) as u64, value16).unwrap();
        assert_eq!(
            fixture.read_vec((p - 1) as u64, 2),
            value16.to_le_bytes().to_vec()
        );

        for bytes_in_first_page in 1..4_u8 {
            let start = p - bytes_in_first_page as usize;
            let value = 0xaabb_ccddu32 ^ bytes_in_first_page as u32;

            fixture.mmu.store32_le(start as u64, value).unwrap();

            assert_eq!(
                fixture.read_vec(start as u64, 4),
                value.to_le_bytes().to_vec(),
                "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }

        for bytes_in_first_page in 1..8 {
            let start = p - bytes_in_first_page;
            let value = 0x0123_4567_89ab_cdefu64 ^ bytes_in_first_page as u64;

            fixture.mmu.store64_le(start as u64, value).unwrap();

            assert_eq!(
                fixture.read_vec(start as u64, 8),
                value.to_le_bytes().to_vec(),
                "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }
    }

    #[test]
    fn crossing_scalar_load_requires_read_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ,
            MemoryProtections::WRITE,
        ]);

        assert!(fixture.mmu.load16_le((page_size() - 1) as u64).is_err());
    }

    #[test]
    fn crossing_scalar_store_requires_write_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::WRITE,
            MemoryProtections::READ,
        ]);

        assert!(fixture
            .mmu
            .store16_le((page_size() - 1) as u64, 0xbeef)
            .is_err());
    }

    #[test]
    fn crossing_scalar_access_to_unmapped_second_page_faults() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let last = (page_size() - 1) as u64;

        fixture.mmu.store_byte(last, 0xaa).unwrap();

        assert!(fixture.mmu.load16_le(last).is_err());
        assert!(fixture.mmu.store16_le(last, 0xbeef).is_err());

        assert_eq!(fixture.mmu.load_byte(last).unwrap(), 0xaa);
    }

    #[test]
    fn store_byte_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_slice_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store(8, &[1, 2, 3, 4]).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_slice_crossing_pages_marks_both_executable_pages_dirty() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));
        assert!(!fixture.is_dirty(1));

        fixture
            .mmu
            .store((page_size() - 2) as u64, &[1, 2, 3, 4])
            .unwrap();

        assert!(fixture.is_dirty(0));
        assert!(fixture.is_dirty(1));
    }

    #[test]
    fn single_page_scalar_store_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store64_le(8, 0x0123_4567_89ab_cdef).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_to_non_executable_page_does_not_mark_dirty() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        fixture.mmu.store16_le(8, 0xbeef).unwrap();
        fixture.mmu.store(16, &[1, 2, 3]).unwrap();

        assert!(!fixture.is_dirty(0));
    }

    // BUG:
    // `for_each_page_chunk` treats `vaddr_end` as an inclusive touched address.
    // Ranges are normally `[start, end)`, so a load of exactly one page from the
    // start of a one-page mapping should not require page 1 to exist.
    #[test]
    fn bug_load_exactly_one_page_should_not_require_next_page() {
        let fixture = Fixture::with_bytes(1, MemoryProtections::READ, pattern_byte);

        let mut out = Box::new_uninit_slice(page_size());

        assert!(fixture.mmu.load(BASE, &mut out).is_ok());
    }

    // BUG:
    // Same exclusive-end bug as above, but through store.
    #[test]
    fn bug_store_exactly_one_page_should_not_require_next_page() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let data = vec![0x5a; page_size()];

        assert!(fixture.mmu.store(BASE, &data).is_ok());
    }

    // BUG:
    // Ending exactly at a page boundary should not check permissions on the next page,
    // because zero bytes are accessed there.
    #[test]
    fn bug_load_ending_at_boundary_should_not_require_read_on_next_page() {
        let fixture = Fixture::with_page_protections_and_bytes(
            &[
                MemoryProtections::READ,
                MemoryProtections::WRITE,
            ],
            pattern_byte,
        );

        let mut out = vec![MaybeUninit::<u8>::uninit(); page_size()];

        assert!(fixture.mmu.load(BASE, &mut out).is_ok());
    }

    // BUG:
    // Cross-page scalar stores write executable pages but never call `set_insn_dirty`
    // on either touched page.
    #[test]
    fn bug_cross_page_scalar_store_should_mark_touched_executable_pages_dirty() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ
                | MemoryProtections::WRITE
                | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));
        assert!(!fixture.is_dirty(1));

        let addr = (page_size() - 1) as u64;

        fixture
            .mmu
            .store16_le(addr, 0xbeef)
            .unwrap();

        assert!(fixture.is_dirty(0));
        assert!(fixture.is_dirty(1));
    }

    fn sparse_fixture_with_hole() -> Fixture {
        let mut backing = PageBacking::new(3);

        unsafe {
            for (i, byte) in backing.as_mut_slice().iter_mut().enumerate() {
                *byte = pattern_byte(i);
            }
        }

        let mut mmu = MMU::<u64>::new();
        let page = page_size() as u64;

        unsafe {
            mmu.map_memory(
                page_addr(0),
                backing.get_page(0).as_mut_ptr(),
                page,
                MemoryProtections::READ | MemoryProtections::WRITE,
            )
                .unwrap();

            // Intentionally skip page 1.

            mmu.map_memory(
                page_addr(2),
                backing.get_page(2).as_mut_ptr(),
                page,
                MemoryProtections::READ | MemoryProtections::WRITE,
            )
                .unwrap();
        }

        Fixture {
            mmu,
            _backing: backing,
        }
    }

    #[test]
    fn unmap_memory_unmaps_one_page() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);

        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_err());

        let mut out = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut out).is_err());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_err());

        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn unmap_memory_unmaps_only_requested_page_range() {
        let mut fixture = Fixture::new(3, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(page_addr(0), 0x10).unwrap();
        fixture.mmu.store_byte(page_addr(1), 0x20).unwrap();
        fixture.mmu.store_byte(page_addr(2), 0x30).unwrap();

        fixture.mmu.unmap_memory(page_addr(1), page).unwrap();

        assert_eq!(fixture.mmu.load_byte(page_addr(0)).unwrap(), 0x10);
        assert!(fixture.mmu.load_byte(page_addr(1)).is_err());
        assert_eq!(fixture.mmu.load_byte(page_addr(2)).unwrap(), 0x30);

        assert!(fixture.mmu.store_byte(page_addr(0), 0x11).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(1), 0x21).is_err());
        assert!(fixture.mmu.store_byte(page_addr(2), 0x31).is_ok());
    }

    #[test]
    fn unmap_memory_can_unmap_multiple_pages() {
        let mut fixture = Fixture::new(4, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(page_addr(1), page * 2).unwrap();

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_err());
        assert!(fixture.mmu.load_byte(page_addr(2)).is_err());
        assert!(fixture.mmu.load_byte(page_addr(3)).is_ok());
    }

    #[test]
    fn unmap_memory_is_idempotent_for_existing_page_entries() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(BASE, page).unwrap();
        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());
    }

    #[test]
    fn unmap_memory_rejects_unaligned_start_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.mmu.unmap_memory(1, page).is_err());

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_unaligned_size_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.mmu.unmap_memory(BASE, page - 1).is_err());

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_range_past_page_table() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture.mmu.unmap_memory(page_addr(1), page).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_ok());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_address_overflow() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        assert!(fixture.mmu.unmap_memory(u64::MAX, 1).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_ok());
    }

    #[test]
    fn unmap_memory_allows_zero_sized_noop_at_start() {
        let mut mmu = MMU::<u64>::new();

        assert!(mmu.unmap_memory(BASE, 0).is_ok());
    }

    #[test]
    fn unmap_memory_then_remap_restores_access() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());

        unsafe {
            fixture
                .mmu
                .map_memory(
                    BASE,
                    fixture._backing.as_mut_ptr(),
                    page,
                    MemoryProtections::READ | MemoryProtections::WRITE,
                )
                .unwrap();
        }

        fixture.mmu.store_byte(BASE, 0xbb).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_can_make_page_read_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .unwrap();

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_err());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn mem_protect_can_make_page_write_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::WRITE)
            .unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.load16_le(BASE).is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_ok());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_ok());
    }

    #[test]
    fn mem_protect_can_make_page_execute_only_for_data_accesses() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::EXECUTE)
            .unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());

        let mut out = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut out).is_err());
        assert!(fixture.mmu.store(BASE, &[0xaa]).is_err());
    }

    #[test]
    fn mem_protect_can_restore_read_write_after_read_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ | MemoryProtections::WRITE)
            .unwrap();

        fixture.mmu.store_byte(BASE, 0xbb).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_only_changes_requested_pages() {
        let mut fixture = Fixture::new(2, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(page_addr(1), page, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(1), 0xbb).is_err());

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_ok());
    }

    #[test]
    fn mem_protect_can_change_multiple_pages() {
        let mut fixture = Fixture::new(3, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(page_addr(0), page * 2, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_err());
        assert!(fixture.mmu.store_byte(page_addr(1), 0xbb).is_err());
        assert!(fixture.mmu.store_byte(page_addr(2), 0xcc).is_ok());

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(2)).is_ok());
    }

    #[test]
    fn mem_protect_rejects_unmapped_page() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());
    }

    #[test]
    fn mem_protect_rejects_sparse_range_and_does_not_partially_update() {
        let mut fixture = sparse_fixture_with_hole();
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(page_addr(0), page * 3, MemoryProtections::READ)
            .is_err());

        // Page 0 and page 2 must still be writable. This catches accidental
        // partial updates if the implementation protects pages as it checks them.
        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(2), 0xbb).is_ok());

        assert_eq!(fixture.mmu.load_byte(page_addr(0)).unwrap(), 0xaa);
        assert_eq!(fixture.mmu.load_byte(page_addr(2)).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_rejects_unaligned_start_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(1, page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_unaligned_size_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(BASE, page - 1, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_range_past_page_table() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(page_addr(1), page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_address_overflow() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        assert!(fixture
            .mmu
            .mem_protect(u64::MAX, 1, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
    }

    #[test]
    fn mem_protect_allows_zero_sized_noop_at_start() {
        let mut mmu = MMU::<u64>::new();
        assert!(mmu.mem_protect(BASE, 0, MemoryProtections::READ).is_ok());
    }
}