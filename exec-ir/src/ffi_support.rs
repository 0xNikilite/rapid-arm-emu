use emu_abi::internal_traits::{GetTlbGeneration, IoMMUByteRawAccess, IoMMURawIntAccess};
use emu_abi::memory::{Page, PageNumber, TLB_MASK, TLB_SIZE, TlbEntry};
use io_mmu::IoMMU;
use std::num::NonZero;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IoMmuStatus {
    Ok = 0,
    Fault = 1,
}

#[inline(always)]
unsafe fn update_tlb(
    tlb: &mut [TlbEntry; TLB_SIZE],
    current_generation: NonZero<u64>,
    page_number: PageNumber,
    page: &Page,
) {
    unsafe {
        let tlb_index = usize::try_from(page_number.0 & TLB_MASK).unwrap_unchecked();

        let tlb_entry = tlb.get_unchecked_mut(tlb_index);
        tlb_entry.update_entry(current_generation, page_number, page);
    }
}

#[inline(always)]
unsafe fn update_tlb_int(
    tlb: &mut [TlbEntry; TLB_SIZE],
    current_generation: NonZero<u64>,
    base_page_number: PageNumber,
    page: &Page,
    second_page: Option<&Page>,
) {
    unsafe { update_tlb(tlb, current_generation, base_page_number, page) }
    if let Some(second_page) = second_page {
        let second_page_number = PageNumber(unsafe { base_page_number.0.unchecked_add(1) });
        unsafe { update_tlb(tlb, current_generation, second_page_number, second_page) }
    }
}

pub unsafe extern "C" fn io_mmu_load_byte(
    io_mmu: *const IoMMU,
    tlb: *mut [TlbEntry; TLB_SIZE],
    addr: u64,
    out: *mut u8,
) -> IoMmuStatus {
    let io_mmu = unsafe { io_mmu.as_ref_unchecked() };
    match io_mmu.load_byte_raw(addr) {
        Ok((page_number, page, byte)) => unsafe {
            let generation = io_mmu.get_generation();
            let tlb = tlb.as_mut_unchecked();
            update_tlb(tlb, generation, page_number, page);
            std::ptr::write(out, byte);
            IoMmuStatus::Ok
        },
        Err(_) => IoMmuStatus::Fault,
    }
}

pub unsafe extern "C" fn io_mmu_store_byte(
    io_mmu: *const IoMMU,
    tlb: *mut [TlbEntry; TLB_SIZE],
    addr: u64,
    value: u8,
) -> IoMmuStatus {
    let io_mmu = unsafe { io_mmu.as_ref_unchecked() };
    match io_mmu.store_byte_raw(addr, value) {
        Ok((page_number, page)) => unsafe {
            let generation = io_mmu.get_generation();
            let tlb = tlb.as_mut_unchecked();
            update_tlb(tlb, generation, page_number, page);
            IoMmuStatus::Ok
        },
        Err(_) => IoMmuStatus::Fault,
    }
}

macro_rules! impl_io_mmu_load_ints {
    ($({ func: $fun_name: ident, ty: $ty: ty })+) => {
        $(
            pub unsafe extern "C" fn $fun_name(
                io_mmu: *const IoMMU,
                tlb: *mut [TlbEntry; TLB_SIZE],
                addr: u64,
                out: *mut $ty,
            ) -> IoMmuStatus {
                let io_mmu = unsafe { io_mmu.as_ref_unchecked() };
                match <IoMMU as IoMMURawIntAccess<$ty>>::load_raw(io_mmu, addr) {
                    Ok((base_page_number, base_page, second_page, value)) => unsafe {
                        let current_generation = io_mmu.get_generation();
                        let tlb = tlb.as_mut_unchecked();

                        update_tlb_int(
                            tlb,
                            current_generation,
                            base_page_number,
                            base_page,
                            second_page
                        );

                        std::ptr::write(out, value);
                        IoMmuStatus::Ok
                    }
                    Err(_) => IoMmuStatus::Fault
                }
            }
        )+
    };
}

impl_io_mmu_load_ints!(
    {
        func: io_mmu_load64_le,
        ty: u64
    }
    {
        func: io_mmu_load32_le,
        ty: u32
    }
    {
        func: io_mmu_load16_le,
        ty: u16
    }
);

macro_rules! impl_io_mmu_store_ints {
    ($({ func: $fun_name: ident, ty: $ty: ty })+) => {
        $(
            pub unsafe extern "C" fn $fun_name(
                io_mmu: *const IoMMU,
                tlb: *mut [TlbEntry; TLB_SIZE],
                addr: u64,
                value: $ty,
            ) -> IoMmuStatus {
                let io_mmu = unsafe { io_mmu.as_ref_unchecked() };
                match <IoMMU as IoMMURawIntAccess<$ty>>::store_raw(io_mmu, addr, value) {
                    Ok((base_page_number, base_page, second_page)) => unsafe {
                        let current_generation = io_mmu.get_generation();
                        let tlb = tlb.as_mut_unchecked();

                        update_tlb_int(
                            tlb,
                            current_generation,
                            base_page_number,
                            base_page,
                            second_page
                        );
                        IoMmuStatus::Ok
                    },
                    Err(_) => IoMmuStatus::Fault
                }
            }
        )+
    };
}

impl_io_mmu_store_ints!(
    {
        func: io_mmu_store64_le,
        ty: u64
    }
    {
        func: io_mmu_store32_le,
        ty: u32
    }
    {
        func: io_mmu_store16_le,
        ty: u16
    }
);
