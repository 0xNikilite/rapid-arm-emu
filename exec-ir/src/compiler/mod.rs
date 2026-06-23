use crate::compiler::cranelift_backend::CraneliftCompiler;
use crate::compiler::gcc_backend::GccJit;
use crate::compiler::llvm_backend::LLVMJit;
use crate::compiler::sync_cell::SyncCell;
use crate::{ExecIr, ExecStateExtra};
use emu_abi::exec_state::ExecState;
use emu_abi::halt_reason::AtomicHaltReason;
use emu_abi::memory::{IoMMUIdentifierRef, Tlb};
use io_mmu::IoMMU;
use io_mmu::icache::ICache;
use std::any::Any;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

mod cranelift_backend;
mod gcc_backend;
mod llvm_backend;
mod sync_cell;

type ExecBlockFFI = unsafe extern "C" fn(
    exec_state: &mut ExecState,
    exec_state_extra: &mut ExecStateExtra,
    tlb_entries: &mut Tlb,
    io_mmu_ident: IoMMUIdentifierRef<'_>,
    halt_reason_ptr: &AtomicU32,
    io_mmu: &IoMMU<dyn ICache>,
) -> u32;

const _: () = assert!(size_of::<&IoMMU<dyn ICache>>() == size_of::<usize>());

#[derive(Clone)]
pub struct CompiledExecChunk {
    ffi: ExecBlockFFI,

    // Keeps the JIT resources alive for at least as long as the fn pointer.
    // If this is dropped while `ffi` may still be called, we get very very bad UB
    _resources: Arc<SyncCell<dyn Any + Send>>,
}

impl CompiledExecChunk {
    fn new_with_recources(ffi: ExecBlockFFI, resources: impl Any + Send) -> Self {
        Self {
            ffi,
            _resources: Arc::new(SyncCell::new(resources)),
        }
    }

    #[inline]
    pub fn call<T: ?Sized + ICache>(
        &self,
        exec_state: &mut ExecState,
        extra: &mut ExecStateExtra,
        tlb: &mut Tlb,
        halt_reason: &AtomicHaltReason,
        io_mmu: &IoMMU<T>,
    ) -> u32 {
        cfg_select! {
            debug_assertions => {
                if let Some((fault, op)) = extra.mem_fault_metadata.take_memory_fault() {
                    panic!("called compiled chunk with a pending memory fault {fault} from {op:?}");
                }
            }
            _ => {
                let had_mem_fault = extra.mem_fault_metadata.was_real_memory_trap;
                if had_mem_fault {
                    #[cold]
                    #[inline(never)]
                    #[track_caller]
                    fn handle_prending_fault() -> ! {
                        emu_abi::abort::panic_abort!(
                            "called compiled chunk with a pending memory fault"
                        )
                    }

                    handle_prending_fault()
                }
            }
        }

        let halt_reason = halt_reason.as_ffi();

        let (io_mmu_ident, io_mmu) = unsafe { io_mmu.as_ffi() };
        unsafe { (self.ffi)(exec_state, extra, tlb, io_mmu_ident, halt_reason, &io_mmu) }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum CompileTier {
    // the mystical Tier0, when there is an even faster backend
    Tier1,
    Tier2,

    GccJit,
    LLVM,
}

struct CompileBlockOptions {
    function_name: String,
    show_disasm: bool,
}

pub struct ExecIrCompiler {
    next_function_id: AtomicUsize,
    cranelift_compiler: OnceLock<CraneliftCompiler>,
    gccjit: OnceLock<GccJit>,
    llvm: OnceLock<LLVMJit>,
    show_disasm: bool,
}

impl Default for ExecIrCompiler {
    fn default() -> Self {
        Self {
            next_function_id: AtomicUsize::new(0),
            cranelift_compiler: OnceLock::new(),
            gccjit: OnceLock::new(),
            llvm: OnceLock::new(),
            show_disasm: false,
        }
    }
}

impl ExecIrCompiler {
    pub fn with_show_disassmbly(mut self) -> Self {
        self.show_disasm = true;
        self
    }

    pub fn compile(&self, exec_ir: &ExecIr, tier: CompileTier) -> CompiledExecChunk {
        self.try_compile(exec_ir, tier)
            .unwrap_or_else(|err| panic!("failed to compile ExecIr: {err}"))
    }

    fn cranelift_compiler(&self) -> &CraneliftCompiler {
        self.cranelift_compiler
            .get_or_init(|| CraneliftCompiler::new().unwrap())
    }

    fn gccjit(&self) -> &GccJit {
        self.gccjit.get_or_init(|| GccJit::new().unwrap())
    }

    fn llvm(&self) -> &LLVMJit {
        self.llvm.get_or_init(|| LLVMJit::new().unwrap())
    }

    pub fn try_compile(
        &self,
        exec_ir: &ExecIr,
        tier: CompileTier,
    ) -> anyhow::Result<CompiledExecChunk> {
        let function_name = {
            let id = self.next_function_id.fetch_add(1, Ordering::Relaxed);
            format!("exec_chunk_{id}")
        };

        let options = CompileBlockOptions {
            function_name,
            show_disasm: self.show_disasm,
        };

        match tier {
            CompileTier::Tier1 => {
                let optimized = false;
                self.cranelift_compiler()
                    .try_compile(options, exec_ir, optimized)
            }
            CompileTier::Tier2 => {
                let optimized = true;
                self.cranelift_compiler()
                    .try_compile(options, exec_ir, optimized)
            }
            CompileTier::GccJit => self.gccjit().try_compile(options, exec_ir),
            CompileTier::LLVM => self.llvm().try_compile(options, exec_ir),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ExecIrCompiler>()
    };

    const _: () = {
        fn _test_compiles_sized_generic<T: ?Sized + ICache>(
            chunk: &CompiledExecChunk,
            exec_state: &mut ExecState,
            extra: &mut ExecStateExtra,
            tlb: &mut Tlb,
            halt_reason: &AtomicHaltReason,
            io_mmu: &IoMMU<T>,
        ) -> u32 {
            chunk.call(exec_state, extra, tlb, halt_reason, io_mmu)
        }
    };
}
