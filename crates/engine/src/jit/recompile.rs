mod scheduler;

pub(crate) use scheduler::{Priority, Ticket};

use crate::FxHashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::{EngineOptions, error::Error};

use super::compiler::{CompilationTier, Compiler, RecompileJob, Unpublished};

pub(crate) const REQUESTED: u8 = 1;
pub(crate) const READY: u8 = 2;
pub(crate) const DEFAULT_COMPILER_THREADS: usize = 4;

#[derive(Clone, Copy)]
pub(crate) struct Log(Option<Instant>);

impl Log {
    pub(crate) fn new(enabled: bool) -> Self {
        Self(enabled.then(Instant::now))
    }

    pub(crate) fn enabled(self) -> bool {
        self.0.is_some()
    }

    pub(crate) fn event(self, message: std::fmt::Arguments<'_>) {
        if let Some(origin) = self.0 {
            eprintln!(
                "[js jit recompile +{:.3}ms] {message}",
                origin.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Variant {
        function: u64,
        variant: super::entries::VariantId,
    },
    Loop(usize),
}

impl Key {
    pub(crate) fn ordinary(function: u64) -> Self {
        Self::Variant {
            function,
            variant: super::entries::VariantId::ordinary(function),
        }
    }
}

pub(crate) struct Completion {
    pub(crate) key: Key,
    pub(crate) ticket: Ticket,
    pub(crate) result: Result<Option<Unpublished>, Error>,
}

struct NotifyOnExit {
    state: Arc<AtomicU8>,
    shared: Arc<Shared>,
}

impl Drop for NotifyOnExit {
    fn drop(&mut self) {
        // A surviving worker keeps the result channel open even if an attempt was lost.
        self.shared.stop();
        self.state.fetch_or(READY, Ordering::Release);
    }
}

struct Work {
    queue: scheduler::Queue<Key, RecompileJob>,
    stopped: bool,
}

struct Shared {
    work: Mutex<Work>,
    available: Condvar,
}

impl Shared {
    fn stop(&self) {
        self.work
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .stopped = true;
        self.available.notify_all();
    }
}

pub(crate) struct Service {
    shared: Arc<Shared>,
    receiver: Receiver<Completion>,
    buffered: Option<Completion>,
    threads: Vec<JoinHandle<()>>,
    log: Log,
    pub(crate) in_flight: FxHashMap<Key, Ticket>,
}

impl Service {
    pub(crate) fn new(
        options: EngineOptions,
        direct: usize,
        state: Arc<AtomicU8>,
        log: Log,
    ) -> Result<Self, Error> {
        let shared = Arc::new(Shared {
            work: Mutex::new(Work {
                queue: scheduler::Queue::new(),
                stopped: false,
            }),
            available: Condvar::new(),
        });
        let (results, receiver) = mpsc::channel();
        let mut service = Self {
            shared,
            receiver,
            buffered: None,
            threads: Vec::new(),
            log,
            in_flight: FxHashMap::default(),
        };
        for index in 0..options.compiler_threads {
            service.threads.push(spawn_worker(
                service.shared.clone(),
                results.clone(),
                options.clone(),
                direct,
                state.clone(),
                log,
                Worker::Background(index),
            )?);
        }
        service.threads.push(spawn_worker(
            service.shared.clone(),
            results,
            options,
            direct,
            state,
            log,
            Worker::Lazy,
        )?);
        Ok(service)
    }

    pub(crate) fn submit(
        &mut self,
        mut job: RecompileJob,
        priority: Priority,
        dependencies: &[Ticket],
    ) -> Result<Ticket, Error> {
        let code = u64::from(job.code.as_u32());
        let key = job.key();
        let mut work = self
            .shared
            .work
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if work.stopped {
            return Err(Error::Compile("compilation service stopped".into()));
        }
        if self.log.enabled() {
            job.queued_at = Some(Instant::now());
            self.log.event(format_args!(
                "queued code={code} function={} priority={priority:?} tier={:?} outstanding={}",
                job.label,
                job.tier,
                self.in_flight.len() + usize::from(!self.in_flight.contains_key(&key))
            ));
        }
        let tier = job.tier;
        let ticket = work.queue.submit(key, priority, job, dependencies);
        if tier == CompilationTier::Baseline
            && let Some(job) = work.queue.pending_mut(key)
        {
            job.tier = tier;
        }
        self.in_flight.insert(key, ticket);
        drop(work);
        self.shared.available.notify_all();
        Ok(ticket)
    }

    pub(crate) fn receive(&mut self) -> Result<Option<Completion>, Error> {
        if let Some(completion) = self.buffered.take() {
            return Ok(Some(completion));
        }
        match self.receiver.try_recv() {
            Ok(completion) => Ok(Some(completion)),
            Err(TryRecvError::Empty)
                if !self
                    .shared
                    .work
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .stopped =>
            {
                Ok(None)
            }
            Err(_) if self.in_flight.is_empty() => Ok(None),
            Err(_) => {
                self.shared.stop();
                self.join();
                while self.receiver.try_recv().is_ok() {}
                self.in_flight.clear();
                Err(Error::Compile("compilation workers stopped".into()))
            }
        }
    }

    pub(crate) fn complete(&mut self, ticket: Ticket, key: Key) {
        assert_eq!(self.in_flight.remove(&key), Some(ticket));
        self.shared
            .work
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .queue
            .complete(ticket, key);
        self.shared.available.notify_all();
    }

    pub(crate) fn wait_for_completion(&mut self) -> Result<(), Error> {
        if self.buffered.is_some() {
            return Ok(());
        }
        match self.receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(completion) => self.buffered = Some(completion),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(Error::Compile("compilation workers stopped".into()));
            }
        }
        Ok(())
    }

    fn join(&mut self) {
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                eprintln!("js jit: compilation worker panicked");
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Worker {
    Background(usize),
    Lazy,
}

fn spawn_worker(
    shared: Arc<Shared>,
    results: Sender<Completion>,
    options: EngineOptions,
    direct: usize,
    state: Arc<AtomicU8>,
    log: Log,
    worker: Worker,
) -> Result<JoinHandle<()>, Error> {
    let name = match worker {
        Worker::Background(index) => index.to_string(),
        Worker::Lazy => "lazy".into(),
    };
    thread::Builder::new()
        .name(format!("js-jit-{name}"))
        .spawn(move || {
            let _notify = NotifyOnExit {
                state: state.clone(),
                shared: shared.clone(),
            };
            log.event(format_args!("worker {name} started"));
            let mut compilers = FxHashMap::default();
            loop {
                let next = {
                    let mut work = shared.work.lock().unwrap();
                    loop {
                        if work.stopped {
                            break None;
                        }
                        let next = match worker {
                            Worker::Background(_) => work.queue.pop(),
                            Worker::Lazy => work.queue.pop_lazy(),
                        };
                        if let Some(job) = next {
                            break Some(job);
                        }
                        work = shared.available.wait(work).unwrap();
                    }
                };
                let Some((ticket, key, job)) = next else {
                    break;
                };
                let code = u64::from(job.code.as_u32());
                if let Some(queued) = job.queued_at {
                    log.event(format_args!(
                        "dequeued code={code} function={} worker={name} queue_ms={:.3}",
                        job.label,
                        queued.elapsed().as_secs_f64() * 1000.0
                    ));
                }
                if let std::collections::hash_map::Entry::Vacant(e) = compilers.entry(job.tier) {
                    match Compiler::new(
                        options.clone(),
                        job.tier,
                        direct as *const usize,
                        Arc::as_ptr(&state) as usize,
                        log,
                    ) {
                        Ok(created) => {
                            e.insert(created);
                        }
                        Err(error) => {
                            let _ = results.send(Completion {
                                ticket,
                                key,
                                result: Err(error),
                            });
                            state.fetch_or(READY, Ordering::Release);
                            break;
                        }
                    }
                }
                let compiler = compilers.get_mut(&job.tier).unwrap();
                let result = compiler.recompile(job).map(|replacement| {
                    replacement.map(|replacement| compiler.take_unpublished(replacement))
                });
                if results
                    .send(Completion {
                        ticket,
                        key,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
                state.fetch_or(READY, Ordering::Release);
            }
            log.event(format_args!("worker {name} stopped"));
        })
        .map_err(|error| Error::Compile(error.to_string()))
}

impl Drop for Service {
    fn drop(&mut self) {
        self.log.event(format_args!(
            "shutdown outstanding={}",
            self.in_flight.len()
        ));
        self.shared.stop();
        self.threads.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_exit_cancels_outstanding_work_with_surviving_workers() {
        let state = Arc::new(AtomicU8::new(0));
        let mut service = Service::new(
            EngineOptions {
                compiler_threads: 3,
                ..Default::default()
            },
            0,
            state.clone(),
            Log::new(false),
        )
        .unwrap();
        assert_eq!(service.threads.len(), 4);
        let mut queue = scheduler::Queue::new();
        let key = Key::ordinary(1);
        let ticket = queue.submit(key, Priority::Optimize, (), &[]);
        service.in_flight.insert(key, ticket);
        let shared = service.shared.clone();
        let failed = thread::spawn(move || {
            let _notify = NotifyOnExit { state, shared };
            panic!("worker failed");
        });
        assert!(failed.join().is_err());
        assert!(service.receive().is_err());
        assert!(service.in_flight.is_empty());
        assert!(service.threads.is_empty());
        assert!(service.receive().unwrap().is_none());
    }
}
