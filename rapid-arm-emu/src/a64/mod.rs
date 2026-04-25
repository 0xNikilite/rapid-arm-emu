use std::ops::Range;
use crate::ir::BasicBlockIr;
use crate::mmu::HostPointer;

mod instruction_decoder;

pub(crate) fn build_ir() -> (Range<HostPointer>, BasicBlockIr) {
    todo!()
}