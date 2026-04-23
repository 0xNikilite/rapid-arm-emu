//! Little-endian tearing load/store primitives.
//!
//! This module provides aligned 8/16/32/64-bit memory operations with the
//! following contract:
//!
//! - Values are interpreted as little-endian.
//! - The caller may or may not have exclusive access to the underlying memory.
//! - Concurrent access may produce torn reads or writes.
//! - Tearing is allowed behavior and is not undefined behavior.
//! - In the absence of contention, a later read must observe exactly the bytes
//!   written by an earlier store.
//!
//! Platform specific behavior:
//! - On x86/x86_64, aligned loads/stores are implemented with inline assembly.
//! - On other targets, operations fall back to bytewise relaxed atomic access.

use std::sync::atomic::{AtomicU8, Ordering};

cfg_select! {
    all(not(miri), any(target_arch = "x86", target_arch = "x86_64")) => {
        #[cfg(not(target_endian = "little"))]
        compile_error!("x86 must be little endian; is there a compiler bug?");


        macro_rules! generate_load_fun_body {
            ($ptr: ident, u64) => {{
                #[cfg(target_arch = "x86_64")]
                {
                    let ret: u64;
                    unsafe {
                        core::arch::asm!(
                            "mov {ret:r}, [{ptr}]",
                            ptr = in(reg) $ptr,
                            ret = lateout(reg) ret,
                            options(pure, readonly, nostack, preserves_flags),
                        );
                    }
                    ret
                }

                #[cfg(target_arch = "x86")]
                {
                    let lo: u32;
                    let hi: u32;
                    unsafe {
                        core::arch::asm!(
                            "mov {lo:e}, [  {ptr}  ]",
                            "mov {hi:e}, [{ptr} + 4]",
                            ptr = in(reg) $ptr,
                            lo = out(reg) lo,
                            hi = out(reg) hi,
                            options(pure, readonly, nostack, preserves_flags),
                        );
                    }
                    ((hi as u64) << 32) | (lo as u64)
                }
            }};
            ($ptr: ident, u32) => {{
                let ret: u32;
                unsafe {
                    core::arch::asm!(
                        "mov {ret:e}, [{ptr}]",
                        ptr = in(reg) $ptr,
                        ret = lateout(reg) ret,
                        options(pure, readonly, nostack, preserves_flags),
                    );
                }
                ret
            }};
            ($ptr: ident, u16) => {{
                let ret: u16;
                unsafe {
                    core::arch::asm!(
                        "movzx {ret:e}, word ptr [{ptr}]",
                        ptr = in(reg) $ptr,
                        ret = lateout(reg) ret,
                        options(pure, readonly, nostack, preserves_flags),
                    );
                }
                ret
            }};
        }

        macro_rules! generate_store_fun_body {
            ($ptr: ident, $value: ident, u64) => {{
                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                        core::arch::asm!(
                            "mov [{ptr}], {val:r}",
                            ptr = in(reg) $ptr,
                            val = in(reg) $value,
                            options(nostack, preserves_flags),
                        );
                    }
                }

                #[cfg(target_arch = "x86")]
                {
                    let lo: u32 = $value as u32;
                    let hi: u32 = ($value >> 32) as u32;
                    unsafe {
                        core::arch::asm!(
                            "mov [  {ptr}  ], {lo:e}",
                            "mov [{ptr} + 4], {hi:e}",
                            ptr = in(reg) $ptr,
                            lo = in(reg) lo,
                            hi = in(reg) hi,
                            options(nostack, preserves_flags),
                        );
                    }
                }
            }};
            ($ptr: ident, $value: ident, u32) => {{
                unsafe {
                    core::arch::asm!(
                        "mov [{ptr}], {val:e}",
                        ptr = in(reg) $ptr,
                        val = in(reg) $value,
                        options(nostack, preserves_flags),
                    );
                }
            }};
            ($ptr: ident, $value: ident, u16) => {{
                unsafe {
                    core::arch::asm!(
                        "mov word ptr [{ptr}], {val:x}",
                        ptr = in(reg) $ptr,
                        val = in(reg) $value,
                        options(nostack, preserves_flags),
                    );
                }
            }};
        }
    }

    _ => {
        macro_rules! generate_load_fun_body {
            ($ptr: ident, $ty: ident) => {{
                let mut bytes = [const { core::mem::MaybeUninit::<u8>::uninit() }; size_of::<$ty>()];
                for (i, slot) in bytes.iter_mut().enumerate() {
                    unsafe {
                        let value = (*$ptr.add(i)).load(Ordering::Relaxed);
                        slot.write(value);
                    }
                }

                let bytes: [u8; size_of::<$ty>()] = unsafe { core::mem::transmute(bytes) };

                <$ty>::from_le_bytes(bytes)
            }};
        }

        macro_rules! generate_store_fun_body {
            ($ptr: ident, $value: ident, $ty: ident) => {{
                let bytes = $value.to_le_bytes();
                for (i, byte) in bytes.iter().enumerate() {
                    unsafe {
                        let byte = *byte;
                        (*$ptr.add(i)).store(byte, Ordering::Relaxed)
                    }
                }
            }};
        }
    }
}

macro_rules! generate_load_store_fun {
    ($($aligned_load_name: ident, $aligned_store_name: ident, $ty: ident;)*) => {
        $(
            /// Loads an aligned little-endian `u64`.
            ///
            /// # Safety
            ///
            #[doc = concat!(" - `ptr` must be valid to read `size_of::<", stringify!($ty), ">()` bytes.")]
            #[doc = concat!(" - `ptr` must be properly to `size_of::<", stringify!($ty), ">()` bytes.")]
            /// - The pointed-to storage must be compatible with bytewise atomic access.
            /// - Concurrent access is permitted, and the result may be torn.
            #[inline(always)]
            pub unsafe fn $aligned_load_name(ptr: *const AtomicU8) -> $ty {
                unsafe {
                    core::hint::assert_unchecked(ptr.addr().is_multiple_of(size_of::<$ty>()))
                }
                generate_load_fun_body!(ptr, $ty)
            }

            #[doc = concat!("Stores an aligned little-endian `", stringify!($ty), "`.")]
            ///
            /// # Safety
            ///
            #[doc = concat!(" - `ptr` must be valid to write `size_of::<", stringify!($ty), ">()` bytes.")]
            #[doc = concat!(" - `ptr` must be properly to `size_of::<", stringify!($ty), ">()` bytes.")]
            /// - The pointed-to storage must be compatible with bytewise atomic access.
            /// - Concurrent access is permitted, and the write may tear.
            #[inline(always)]
            pub unsafe fn $aligned_store_name(ptr: *const AtomicU8, value: $ty) {
                unsafe {
                    core::hint::assert_unchecked(ptr.addr().is_multiple_of(size_of::<$ty>()))
                }

                generate_store_fun_body!(ptr, value, $ty)
            }
        )*
    };
}


generate_load_store_fun! {
    load_le_64_aligned, store_le_64_aligned, u64;
    load_le_32_aligned, store_le_32_aligned, u32;
    load_le_16_aligned, store_le_16_aligned, u16;
}

/// Loads a single byte.
///
/// # Safety
///
/// `ptr` must be valid to read one `AtomicU8`.
#[inline(always)]
pub unsafe fn load_le_8_aligned(ptr: *const AtomicU8) -> u8 {
    unsafe { (*ptr).load(Ordering::Relaxed) }
}

/// Stores a single byte.
///
/// # Safety
///
/// `ptr` must be valid to write one `AtomicU8`.
#[inline(always)]
pub unsafe fn store_le_8_aligned(ptr: *const AtomicU8, value: u8) {
    unsafe { (*ptr).store(value, Ordering::Relaxed) }
}
