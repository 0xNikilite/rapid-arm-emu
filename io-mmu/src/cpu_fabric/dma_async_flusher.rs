use crate::dma::DmaDevice;
use crate::page_table::SharedDmaPage;
use emu_abi::abort::AbortGuard;
use emu_abi::abort::panic_abort;
use emu_abi::convert::{u64_to_usize, usize_to_u64};
use emu_abi::memory::{PageNumber, PagePointer};
use parking_lot::{Condvar, Mutex};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use thin_vec::{ThinVec, thin_vec};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct DmaDeviceId(NonNull<()>);

impl DmaDeviceId {
    pub fn of(dma: &Arc<dyn DmaDevice>) -> Self {
        Self(unsafe { NonNull::new_unchecked(Arc::as_ptr(dma).cast::<()>().cast_mut()) })
    }
}

unsafe impl Send for DmaDeviceId {}
unsafe impl Sync for DmaDeviceId {}

pub(crate) trait DmaFlusherCallbacks: Send {
    fn on_success(self: Box<Self>);

    fn on_failure(self: Box<Self>, error: &anyhow::Error);
}

pub(crate) struct DmaFnCallback<F>(F);

impl<F: FnOnce(Result<(), &anyhow::Error>) + Send> DmaFnCallback<F> {
    pub const fn new(callback: F) -> Self {
        Self(callback)
    }
}

impl<F: FnOnce(Result<(), &anyhow::Error>) + Send> DmaFlusherCallbacks for DmaFnCallback<F> {
    fn on_success(self: Box<Self>) {
        let cb = (*self).0;
        cb(Ok(()))
    }

    fn on_failure(self: Box<Self>, error: &anyhow::Error) {
        let cb = (*self).0;
        cb(Err(error))
    }
}

struct QueueEntry {
    owned_page: SharedDmaPage,
    device: Arc<dyn DmaDevice>,
    dev_page_offset: PageNumber,
    callbacks: ThinVec<Box<dyn DmaFlusherCallbacks>>,
}

struct EnqueueEntry {
    owned_page: SharedDmaPage,
    device: Arc<dyn DmaDevice>,
    dev_page_offset: PageNumber,
    callback: Option<Box<dyn DmaFlusherCallbacks>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct QueueDedupEntry {
    host_ptr: PagePointer,
    dev_id: DmaDeviceId,
    dev_page_offset: PageNumber,
}

struct Queue {
    queue: VecDeque<QueueEntry>,
    already_in_queue: HashMap<QueueDedupEntry, u64>,
    // Even at 1 billion entries per second, this u64 takes ~585 years to overflow.
    // not a practical concern on any platform.
    base: u64,
}

impl Queue {
    pub fn dequeue(&mut self) -> Option<(QueueEntry, PagePointer)> {
        assert_eq!(self.queue.len(), self.already_in_queue.len());

        let entry = self.queue.pop_front()?;

        let abort = AbortGuard(());

        let page = entry
            .owned_page
            .get()
            .expect("enqueued a page that wasn't faulted");

        let ptr = page.page_pointer();

        let dev_id = DmaDeviceId::of(&entry.device);

        let removed = self.already_in_queue.remove(&QueueDedupEntry {
            host_ptr: ptr,
            dev_id,
            dev_page_offset: entry.dev_page_offset,
        });

        // must be the last entry (the one just popped)
        assert_eq!(removed, Some(self.base));

        self.base = match self.queue.is_empty() {
            true => 0,
            false => self.base.strict_add(1),
        };

        abort.disarm();

        Some((entry, ptr))
    }

    pub fn enqueue(
        &mut self,
        enqueue: EnqueueEntry,
        queue_limit: usize,
    ) -> Result<(), EnqueueEntry> {
        assert_eq!(self.queue.len(), self.already_in_queue.len());

        let mem_page = enqueue.owned_page.get().expect("can't enqueue empty page");
        let ptr = mem_page.page_pointer();
        let dev_id = DmaDeviceId::of(&enqueue.device);

        let dedup_entry = QueueDedupEntry {
            host_ptr: ptr,
            dev_id,
            dev_page_offset: enqueue.dev_page_offset,
        };

        match self.already_in_queue.entry(dedup_entry) {
            Entry::Occupied(index) => {
                if let Some(callback) = enqueue.callback {
                    let abort = AbortGuard(());
                    let index = u64_to_usize((*index.get()).strict_sub(self.base)).unwrap();
                    let entry = &mut self.queue[index];
                    abort.disarm();

                    entry.callbacks.push(callback);
                }

                Ok(())
            }

            Entry::Vacant(entry) => {
                if self.queue.len() >= queue_limit {
                    return Err(enqueue);
                }

                let abort = AbortGuard(());
                let pushed_val_idx = usize_to_u64(self.queue.len()).unwrap();
                let abs_index = self.base.strict_add(pushed_val_idx);
                self.queue.push_back(QueueEntry {
                    owned_page: enqueue.owned_page,
                    device: enqueue.device,
                    dev_page_offset: enqueue.dev_page_offset,
                    callbacks: match enqueue.callback {
                        Some(cb) => thin_vec![cb],
                        None => thin_vec![],
                    },
                });

                entry.insert(abs_index);
                abort.disarm();

                Ok(())
            }
        }
    }
}

struct Inner {
    queue_limit: usize,
    queue: Mutex<Queue>,
    closed: AtomicBool,
    enqueued: Condvar,
    dequeued: Condvar,
}

#[cold]
#[inline(never)]
fn worker_thread_died() -> ! {
    panic_abort!("async flusher thread died unexpectedly")
}

impl Inner {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        // ensure there is no lost wakeup by first locking the queue
        let lock = self.queue.lock();
        self.enqueued.notify_all();
        self.enqueued.notify_one();
        self.dequeued.notify_all();
        self.dequeued.notify_one();
        drop(lock)
    }

    fn enqueue(
        &self,
        page: SharedDmaPage,
        device: Arc<dyn DmaDevice>,
        page_offset: PageNumber,
        callback: Option<Box<dyn DmaFlusherCallbacks>>,
    ) {
        if page.get().is_none() {
            return;
        }

        let mut entry = EnqueueEntry {
            owned_page: page,
            device,
            dev_page_offset: page_offset,
            callback,
        };

        let mut lock = self.queue.lock();
        loop {
            if self.closed.load(Ordering::Acquire) {
                worker_thread_died()
            }

            match lock.enqueue(entry, self.queue_limit) {
                Ok(()) => break,
                Err(requeue) => entry = requeue,
            }

            self.dequeued.wait(&mut lock);
        }

        drop(lock);

        self.enqueued.notify_one();
    }

    fn dequeue(&self) -> Option<(QueueEntry, PagePointer)> {
        // this avoids re locking the queue when loading closed
        // do note the lock gets released in the line `wait(lock)`
        let mut lock = None;
        loop {
            if self.closed.load(Ordering::Acquire) {
                return None;
            }

            let lock = lock.get_or_insert_with(|| self.queue.lock());

            if let Some(entry) = lock.dequeue() {
                self.dequeued.notify_one();
                return Some(entry);
            }

            self.enqueued.wait(lock);
        }
    }
}

struct DmaFlusherWorker(Arc<Inner>);

impl Drop for DmaFlusherWorker {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl DmaFlusherWorker {
    fn run_worker(self) {
        pub fn catch_unwind<F: FnOnce() -> R, R>(f: F) -> std::thread::Result<R> {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        }

        while let Some(entry) = self.0.dequeue() {
            let (entry, page_ptr) = entry;

            let QueueEntry {
                owned_page,
                device,
                dev_page_offset: page_offset,
                callbacks,
            } = entry;

            let res = catch_unwind(move || unsafe {
                device.fault_out(page_offset, page_ptr.as_non_null_ptr())
            });

            // explicitly keep owned_page alive only until flush completes
            drop(owned_page);

            let res = res.unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "DMA device panicked whilst processing request"
                ))
            });

            let _panic = catch_unwind(move || {
                let callbacks = callbacks.into_iter();

                match res {
                    Ok(()) => callbacks.for_each(|cb| {
                        let _panic = catch_unwind(|| cb.on_success());
                    }),
                    Err(ref error) => callbacks.for_each(|cb| {
                        let _panic = catch_unwind(|| cb.on_failure(error));
                    }),
                }
            });
        }
    }
}

// if this ever becomes a bottleneck, use a threadpool instead
pub(crate) struct DmaAsyncFlusher {
    // make the handle to inner weak
    // so that we avoid any ref-cycles that may happen from storing `CpuFabric`
    // inside a callback; and also all we really need inner for is to queue things
    // we don't ever need any additional data from it
    inner: Weak<Inner>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for DmaAsyncFlusher {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.close();
        }

        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        }
    }
}

impl DmaAsyncFlusher {
    pub fn new(queue_limit: usize) -> Self {
        let inner = Arc::new(Inner {
            queue_limit,
            queue: Mutex::new(Queue {
                queue: VecDeque::new(),
                already_in_queue: HashMap::new(),
                base: 0,
            }),
            closed: AtomicBool::new(false),
            enqueued: Condvar::new(),
            dequeued: Condvar::new(),
        });

        // construct self first
        // this way if the thread construction panics inner still closes
        let mut this = Self {
            inner: Arc::downgrade(&inner),
            thread: None,
        };

        let worker = DmaFlusherWorker(inner);

        let thread = std::thread::spawn(move || worker.run_worker());

        this.thread = Some(thread);

        this
    }

    pub fn enqueue(
        &self,
        page: &SharedDmaPage,
        device: &Arc<dyn DmaDevice>,
        page_offset: PageNumber,
    ) {
        self.inner
            .upgrade()
            .unwrap_or_else(|| worker_thread_died())
            .enqueue(page.clone(), Arc::clone(device), page_offset, None)
    }

    pub fn enqueue_with_cb(
        &self,
        page: &SharedDmaPage,
        device: &Arc<dyn DmaDevice>,
        page_offset: PageNumber,
        callback: Box<dyn DmaFlusherCallbacks>,
    ) {
        let cb = Some(callback);
        self.inner
            .upgrade()
            .unwrap_or_else(|| worker_thread_died())
            .enqueue(page.clone(), Arc::clone(device), page_offset, cb)
    }
}

impl Default for DmaAsyncFlusher {
    fn default() -> Self {
        let queue_limit = 1024;
        Self::new(queue_limit)
    }
}
