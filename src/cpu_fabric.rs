//! CPU fabric state shared by emulated CPU contexts.
//!
//! The `cpu_fabric` module owns state that is conceptually shared between CPUs, cores.
//! Today, that shared state is the exclusive monitor which is used to model
//! load-exclusive/store-exclusive style atomic sequences.
//!
//! In the future, `CpuFabric` may grow to include other CPU-adjacent shared
//! resources, such as interrupt routing, cache-coherency metadata, shared
//! timing state, or other cross-core coordination structures. For now, it is a
//! small wrapper around the exclusive monitor so cloned CPU contexts can observe
//! and invalidate the same reservation state.


use std::mem::MaybeUninit;
use std::sync::Arc;
use crossbeam_utils::CachePadded;
use crate::sync::Mutex;
use crate::vaddr;
use crate::vaddr::VAddr;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Version(u64);

pub(crate) struct ReservationToken<A: VAddr> {
    version: Version,
    value: <A as vaddr::sealed::VAddr>::ExclusiveMonitorLoadValue,
}


pub(crate) struct ReservationSlot<VAddr> {
    address: VAddr,
    version: Version,
}

pub(crate) const BUCKET_COUNT: u16 = 257;

pub(crate) struct ExclusiveMonitor<VAddr> {
    reservations: [CachePadded<Mutex<ReservationSlot<VAddr>>>; BUCKET_COUNT as usize],
}

impl<A: VAddr> ExclusiveMonitor<A> {

    /// #
    pub fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            let ptr = this.as_mut_ptr();
            for i in 0..BUCKET_COUNT {
                std::ptr::write(
                    &raw mut (*ptr).reservations[usize::from(i)],
                    CachePadded::new(Mutex::new(ReservationSlot {
                        address: A::NULL,
                        version: Version(0),
                    }))
                )
            }

            this.assume_init_mut()
        }
    }

    pub fn new_arc() -> Arc<Self> {
        let mut uninit = Arc::new_uninit();
        Self::init(Arc::get_mut(&mut uninit).unwrap());
        unsafe { uninit.assume_init() }
    }

    #[must_use]
    pub(crate) fn ldrex(
        &self,
        addr: A,
        load_op: impl FnOnce() -> A::ExclusiveMonitorLoadValue,
    ) -> ReservationToken<A> {
        let reserve_idx = addr.reservation_index();
        let mut lock = self.reservations[reserve_idx].lock();

        lock.address = addr;
        let version = lock.version;

        let value = load_op();
        ReservationToken {
            version,
            value,
        }
    }

    pub(crate) fn strex<T>(
        &self,
        addr: A,
        tok: ReservationToken<A>,
        store_op: impl FnOnce(A::ExclusiveMonitorLoadValue) -> Result<T, ()>,
    ) -> Result<T, ()> {
        let reserve_idx = addr.reservation_index();
        let mut lock = self.reservations[reserve_idx].lock();

        if lock.address != addr || lock.version != tok.version {
            return Err(());
        }

        // Wrapping is acceptable here: token reuse would require 2^64 successful
        // invalidations of the same reservation slot before an old token could match again.
        // and there aren't any better alternatives
        lock.version.0 = lock.version.0.wrapping_add(1);

        store_op(tok.value)
    }
}

#[repr(transparent)]
pub struct CpuFabric<VAddr>(Arc<ExclusiveMonitor<VAddr>>);

impl<A: VAddr> CpuFabric<A> {
    pub fn new() -> Self {
        Self(ExclusiveMonitor::new_arc())
    }
}

impl<A> Clone for CpuFabric<A> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    fn clone_from(&mut self, source: &Self) {
        if !Arc::ptr_eq(&self.0, &source.0) {
            *self = source.clone();
        }
    }
}

impl<A: VAddr> Default for CpuFabric<A> {
    fn default() -> Self {
        Self::new()
    }
}

const _: () = {
    fn _assert_sync<A: VAddr>() {
        fn is_sync<T: Sync>() {}

        is_sync::<CpuFabric<A>>()
    }

    fn _assert_send<A: VAddr>() {
        fn is_send<T: Send>() {}

        is_send::<CpuFabric<A>>()
    }
};


#[cfg(test)]
mod tests {
    use super::*;

    use loom::sync::{Arc, Mutex, Condvar};
    use crate::a64::sealed::ExclusiveMonitorLoad;

    struct BarrierState {
        count: usize,
        generation_id: usize,
    }

    // port of std::sync::Barrier
    struct Barrier {
        num_threads: usize,
        state: Mutex<BarrierState>,
        cond: Condvar,
    }

    impl Barrier {
        fn new(n: usize) -> Self {
            Self {
                num_threads: n,
                state: Mutex::new(BarrierState {
                    count: 0,
                    generation_id: 0,
                }),
                cond: Condvar::new(),
            }
        }

        fn wait(&self) {
            let mut lock = self.state.lock().unwrap();
            let local_gen = lock.generation_id;
            lock.count = lock.count.strict_add(1);
            if lock.count < self.num_threads {
                while local_gen == lock.generation_id {
                    lock = self.cond.wait(lock).unwrap();
                }
            } else {
                lock.count = 0;
                lock.generation_id = lock.generation_id.wrapping_add(1);
                self.cond.notify_all();
            }
        }
    }

    #[test]
    fn test_exclusive_monitor() {
        if cfg!(miri) {
            return;
        }

        loom::model(move || {
            let monitor = Arc::from_std(ExclusiveMonitor::<u64>::new_arc());
            let memory = Arc::new(Mutex::new(0_u32));
            let barrier = Arc::new(Barrier::new(2));

            let thread_run = || {
                let memory = Arc::clone(&memory);
                let monitor = Arc::clone(&monitor);
                let barrier = Arc::clone(&barrier);
                loom::thread::spawn(move || {
                    let addr = 0x10000DEAD00BEEF;

                    let token = monitor.ldrex(addr, || ExclusiveMonitorLoad::U8(0));
                    barrier.wait();
                    let _ = monitor.strex(addr, token, |val| {
                        match val {
                            ExclusiveMonitorLoad::U8(0) => {
                                *memory.try_lock().unwrap() += 1;
                                Ok(())
                            }
                            _ => Err(())
                        }
                    });
                })
            };

            let jh1 = thread_run();
            let jh2 = thread_run();

            jh1.join().unwrap();
            jh2.join().unwrap();

            assert_eq!(*memory.try_lock().unwrap(), 1);
        });
    }
}