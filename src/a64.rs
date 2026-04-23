use crate::{cpu_fabric, vaddr};
use crate::cpu_fabric::CpuFabric;
use crate::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
use crate::mmu::{MemoryFault, MMU};

pub type VAddr = u64;

pub const PAGE_SIZE: VAddr = 4096;

// FIXME feature(const_convert)
#[allow(
    clippy::cast_possible_truncation,
    reason = "this function ensures no truncation happens"
)]
const fn u64_to_usize(int: u64) -> Option<usize> {
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
    const PAGE_SIZE_SELF: Self = PAGE_SIZE;
    
    type InsnWord = u32;

    fn reservation_index(self) -> usize {
        let mut x = self;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        
        let bucket_count: u16 = cpu_fabric::BUCKET_COUNT;
        
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

    fn is_multiple_of(self, rhs: Self) -> bool {
        self.is_multiple_of(rhs)
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
    
    fn fetch_insn_word(self, mmu: &MMU<Self>) -> Result<Self::InsnWord, MemoryFault> {
        let (page, offset) = mmu.aligned_static_acces::<4>(self)?;
        // Safety: `self` is properly aligned since `aligned_static_acces` succeded
        unsafe { page.fetch32(offset) }
    }
}

#[derive(Copy, Clone)]
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
    mmu: MMU<VAddr>,
    fabric: CpuFabric<VAddr>,
    halt_reason: AtomicHaltReason,
    executing: parking_lot::Mutex<ExecutingData>,
}

impl Arm64CpuCore {
    pub fn new(mmu: MMU<VAddr>, fabric: CpuFabric<VAddr>) -> Self {
        Self {
            mmu,
            fabric,
            halt_reason: AtomicHaltReason::new(HaltReasonInner::empty()),
            executing: parking_lot::Mutex::new(const {
                ExecutingData {
                    sp: 0,
                    pc: 0,
                    x_registers: [0; 31],
                    pstate: 0,
                    fpsr: 0,
                    fpcr: 0,
                    vectors: [Vector(0); 32],
                }
            })
        }
    }

    pub fn mmu(&self) -> &MMU<VAddr> {
        &self.mmu
    }

    pub fn mmu_mut(&mut self) -> &mut MMU<VAddr> {
        &mut self.mmu
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
