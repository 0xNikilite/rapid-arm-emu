use std::collections::HashMap;
use std::num::NonZero;
use std::ops::Range;
use crate::armv9::{Armv9CpuCore, ProcessorState};
use crate::halt_reason::HaltReasonInner;
use crate::io_mmu::HostPointer;

const MAX_INSTRUCTIONS_PER_BLOCK: NonZero<u32> = NonZero::new(16 * 1024).unwrap();


// this might seem wierd, but when compiling a basic block,
// we might start from one place, and go back
// like:
//               top:
//               nop
//               nop
// jumps here -> add x, y;
//               jump top
pub(crate) struct CodeBlock {
    /// Inclusive virtual address range touched while decoding.
    /// \[start, end\].
    /// note that the start isn't guarenteed to be the entrypoint of the function
    addr: Range<HostPointer>,

    machine_code_handle: unsafe extern "C" fn(),
}

impl CodeBlock {
    fn execute(
        &self,
        _state: &mut ProcessorState,
        _cpu: &Armv9CpuCore
    ) -> HaltReasonInner {
        todo!()
    }
}

pub(crate) struct CodeCache {
    cache: HashMap<HostPointer, CodeBlock>
}

impl CodeCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new()
        }
    }

    pub fn run(&mut self, state: &mut ProcessorState, cpu: &Armv9CpuCore) -> HaltReasonInner {
        todo!()
    }

    pub fn invalidate_cache(&mut self, range: Range<HostPointer>) {
        self.cache.retain(move |_entrypoint, block| {
            let collides = range.start < block.addr.end && block.addr.start < range.end;
            !collides
        })
    }
}
