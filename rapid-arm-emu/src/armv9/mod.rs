use parking_lot::Mutex;
use crate::armv9::jit::CodeCache;
use crate::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
use crate::mmu::IoMMU;

pub(crate) mod jit;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(16))]
pub struct Vector(pub u128);

const _: () = assert!(align_of::<Vector>() == 16 && size_of::<Vector>() == 16);

#[repr(C)]
pub(crate) struct ProcessorState {
    sp: u64,
    pc: u64,
    x_registers: [u64; 31],
    pstate: u32,
    fpsr: u32,
    fpcr: u32,
    vectors: [Vector; 32],
}

impl ProcessorState {
    pub fn initial() -> Self {
        Self {
            sp: 0,
            pc: 0,
            x_registers: [0; 31],
            pstate: 0,
            fpsr: 0,
            fpcr: 0,
            vectors: [Vector(0); 32],
        }
    }
}

struct ExecutingData {
    processor_state: ProcessorState,
    code_cache: CodeCache,
}

impl ExecutingData {
    fn resume(&mut self, cpu: &Armv9CpuCore) -> HaltReasonInner {
        self.code_cache.run(&mut self.processor_state, cpu)
    }

    fn step(&mut self, cpu: &Armv9CpuCore) -> HaltReasonInner {
        cpu.halt_reason.add_reasons(HaltReasonInner::Step);
        self.resume(cpu)
    }


    fn invalidate_instruction_cache(&mut self, cpu: &Armv9CpuCore) {
        for dirty_range in cpu.mmu.drain_dirty_icache() {
            self.code_cache.invalidate_cache(dirty_range)
        }
    }
}

pub struct Armv9CpuCore {
    mmu: IoMMU,
    halt_reason: AtomicHaltReason,
    executing: Mutex<ExecutingData>,
}

impl Armv9CpuCore {
    pub fn new(mmu: IoMMU) -> Self {
        Self {
            mmu,
            halt_reason: AtomicHaltReason::new(HaltReasonInner::empty()),
            executing: Mutex::new(ExecutingData {
                processor_state: ProcessorState::initial(),
                code_cache: CodeCache::new()
            })
        }
    }

    pub fn mmu(&self) -> &IoMMU {
        &self.mmu
    }

    pub fn mmu_mut(&mut self) -> &mut IoMMU {
        &mut self.mmu
    }

    #[track_caller]
    fn execute(
        &self,
        step: bool,
        fun: impl FnMut(&mut ExecutingData) -> HaltReasonInner
    ) -> Option<HaltReason> {
        let Some(mut lock) = self.executing.try_lock() else {
            panic!("the CPU is already executing")
        };

        let data: &mut ExecutingData = &mut lock;
        let mut fun = fun;
        loop {
            let halt_reason = fun(data);
            
            debug_assert!(!halt_reason.is_empty());

            if halt_reason.contains(HaltReasonInner::InvalidateInsnCache) {
                data.invalidate_instruction_cache(self);
                // if we only halted because we had InvalidateInstructionCache
                if (halt_reason ^ HaltReasonInner::InvalidateInsnCache).is_empty() {
                    continue
                }
            }
            
            let reason = HaltReason::from_inner(halt_reason);
            if reason.is_some() || step {
                break reason
            }
        }
    }

    /// Runs the emulated CPU.
    /// Cannot be recursively called.
    pub fn resume(&self) -> HaltReason {
        let step = false;
        self.execute(step, |data| data.resume(self))
            .expect("execute should never return None, since it is not in step mode")
    }

    /// Step the emulated CPU for one instruction.
    /// Cannot be recursively called.
    pub fn step(&self) -> Option<HaltReason> {
        let step = true;
        self.execute(step, |data| data.step(self))
    }

    pub fn halt(&self, reason: HaltReason) {
        self.halt_reason.add_reasons(reason.into_inner())
    }
}
