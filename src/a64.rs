use std::ptr::NonNull;
use crate::{exclusive_monitor, vaddr};
use crate::exclusive_monitor::ExclusiveMonitor;
use crate::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};

pub type VAddr = u64;

pub const PAGE_SIZE: VAddr = 4096;

#[allow(
    clippy::cast_possible_truncation,
    reason = "this ethod ensures no truncation happens"
)]
const fn u64_to_usize(int: u64) -> Option<usize> {
    // FIXME feature(const_convert)
    
    match usize::BITS >= u64::BITS {
        true => Some(int as usize),
        false => {
            // widening cast
            let max = usize::MAX as u64;
            if int > max {
                return None
            }
            Some(PAGE_SIZE as usize)
        },
    }
}


unsafe impl vaddr::sealed::VAddr for u64 {
    const NULL: Self = 0;
    const PAGE_SIZE: usize = u64_to_usize(PAGE_SIZE).unwrap();

    fn reservation_index(self) -> usize {
        let mut x = self;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        
        let bucket_count: u16 = exclusive_monitor::BUCKET_COUNT;
        
        #[allow(
            clippy::cast_possible_truncation,
            reason = "bucket count is u16, so x mod u16 fitsi in u16"
        )]
        let index = (x % u64::from(bucket_count)) as u16;
        
        usize::from(index)
    }

    fn add_addr(self, other: Self) -> Option<Self> {
        self.checked_add(other)
    }

    fn add_offset(self, other: usize) -> Option<Self> {
        self.checked_add(u64::try_from(other).ok()?)
    }

    fn is_page_aligned(self) -> bool {
        self.is_multiple_of(PAGE_SIZE)
    }

    fn try_to_usize(self) -> Option<usize> {
        u64_to_usize(self)
    }

    fn div_rem_page_size(self) -> (Self, usize) {
        let div = self / PAGE_SIZE;
        // PAGE_SIZE fits in a usize, so x mod PAGE_SIZE must fit in usize
        let offset = (self % PAGE_SIZE) as usize;
        (div, offset)
    }

    unsafe fn div_page_size_unchecked(self) -> Self {
        let (div, rem) = self.div_rem_page_size();
        unsafe { core::hint::assert_unchecked(rem == 0) }
        div
    }
}

#[repr(C, align(16))]
struct Vector(u128);

const _: () = assert!(align_of::<Vector>() == 16);

struct ExecutingData {
    sp: u64,
    pc: u64,
    x_registers: [u64; 31],
    pstate: u32,
    fpsr: u32,
    fpcr: u32,
    vectors: [Vector; 32],
    
}

impl ExecutingData {
    fn clear_instruction_cache(&mut self) {
        todo!("invalidate instruction cache")
    }
}

pub struct Arm64CpuCore {
    exclusive_monitor: NonNull<ExclusiveMonitor<VAddr>>,
    halt_reason: AtomicHaltReason,
    executing: parking_lot::Mutex<ExecutingData>,
}

impl Arm64CpuCore {
    // FIXME broken docs
    /// Creates a new CPU core backed by guest memory at `base_ptr`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following:
    ///
    /// - `base_ptr` points to a valid allocation of at least `memory_size` bytes.
    /// - `memory_size` must be a multiple of `PAGE_SIZE`
    /// - That allocation remains alive and must not be freed for at least as long as
    ///   this `Arm64CpuCore` exists.
    /// - Any access through `base_ptr`, whether by this CPU core or by external code,
    ///   must obey Rust's aliasing and synchronization rules.
    /// - If other code reads or writes the pointed-to memory while this CPU may execute,
    ///   the caller must ensure the CPU is not concurrently executing instructions that
    ///   may access that memory.
    ///
    /// In other words, the backing memory must outlive the CPU core, and the caller is
    /// responsible for preventing concurrent unsynchronized access between the CPU and
    /// any external user of that memory.
    pub unsafe fn new(
        base_ptr: *mut u8,
        memory_size: VAddr,
        exclusive_monitor: *const ExclusiveMonitor<VAddr>,
    ) -> Self {
        let _ = (base_ptr, memory_size, exclusive_monitor);
        todo!()
    }

    #[track_caller]
    fn execute(
        &self,
        fun: impl FnMut(&mut ExecutingData) -> HaltReasonInner
    ) -> HaltReason {
        let Some(mut lock) = self.executing.try_lock() else {
            panic!("the CPU is already executing")
        };

        let data: &mut ExecutingData = &mut lock;
        let mut fun = fun;
        loop {
            let halt_reason = fun(data);

            if halt_reason.contains(HaltReasonInner::InvalidateInstructionCache) {
                data.clear_instruction_cache();
                // if we only halted because we had InvalidateInstructionCache
                if (halt_reason ^ HaltReasonInner::InvalidateInstructionCache).is_empty() {
                    continue
                }
            }

            break HaltReason::from_inner(halt_reason)
        }
    }

    /// Runs the emulated CPU.
    /// Cannot be recursively called.
    pub fn run(&self) -> HaltReason {
        todo!()
    }

    /// Step the emulated CPU for one instruction.
    /// Cannot be recursively called.
    pub fn step(&self) -> HaltReason {
        todo!()
    }

    pub fn halt(&self, reason: HaltReason) {
        self.halt_reason.add_reasons(reason.into_inner())
    }
}
