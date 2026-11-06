mod scheduler;

pub(crate) use scheduler::{Priority, Ticket};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::{EngineOptions, error::Error};

use super::compiler::{Compiler, RecompileJob, Unpublished};

pub(crate) const REQUESTED: u8 = 1;
pub(crate) const READY: u8 = 2;
pub(crate) const DEFAULT_COMPILER_THREADS: usize = 2;

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
    Function(u64),
    Loop(usize),
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
    threads: Vec<JoinHandle<()>>,
    log: Log,
    pub(crate) in_flight: HashMap<Key, Ticket>,
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
            threads: Vec::new(),
            log,
            in_flight: HashMap::new(),
        };
        for index in 0..options.compiler_threads {
            service.threads.push(spawn_worker(
                service.shared.clone(),
                results.clone(),
                options,
                direct,
                state.clone(),
                log,
                index,
            )?);
        }
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
                "queued code={code} function={} priority={priority:?} outstanding={}",
                job.label,
                self.in_flight.len() + usize::from(!self.in_flight.contains_key(&key))
            ));
        }
        let ticket = work.queue.submit(key, priority, job, dependencies);
        self.in_flight.insert(key, ticket);
        drop(work);
        self.shared.available.notify_one();
        Ok(ticket)
    }

    pub(crate) fn receive(&mut self) -> Result<Option<Completion>, Error> {
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

    pub(crate) fn cancel_pending(&mut self, key: Key) -> bool {
        let ticket = self
            .shared
            .work
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .queue
            .cancel_pending(key);
        if let Some(ticket) = ticket {
            assert_eq!(self.in_flight.remove(&key), Some(ticket));
            self.shared.available.notify_all();
            return true;
        }
        false
    }

    fn join(&mut self) {
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                eprintln!("js jit: compilation worker panicked");
            }
        }
    }
}

fn spawn_worker(
    shared: Arc<Shared>,
    results: Sender<Completion>,
    options: EngineOptions,
    direct: usize,
    state: Arc<AtomicU8>,
    log: Log,
    index: usize,
) -> Result<JoinHandle<()>, Error> {
    thread::Builder::new()
        .name(format!("js-jit-{index}"))
        .spawn(move || {
            let _notify = NotifyOnExit {
                state: state.clone(),
                shared: shared.clone(),
            };
            log.event(format_args!("worker {index} started"));
            let mut compiler = None;
            loop {
                let next = {
                    let mut work = shared.work.lock().unwrap();
                    loop {
                        if work.stopped {
                            break None;
                        }
                        if let Some(job) = work.queue.pop() {
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
                        "dequeued code={code} function={} queue_ms={:.3}",
                        job.label,
                        queued.elapsed().as_secs_f64() * 1000.0
                    ));
                }
                if compiler.is_none() {
                    match Compiler::new(
                        options,
                        direct as *const usize,
                        Arc::as_ptr(&state) as usize,
                        log,
                    ) {
                        Ok(created) => compiler = Some(created),
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
                let compiler = compiler.as_mut().unwrap();
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
            log.event(format_args!("worker {index} stopped"));
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
        self.join();
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
        assert_eq!(service.threads.len(), 3);
        let mut queue = scheduler::Queue::new();
        let key = Key::Function(1);
        let ticket = queue.submit(key, Priority::Normal, (), &[]);
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
