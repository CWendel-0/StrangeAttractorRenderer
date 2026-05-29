use std::thread::{self, JoinHandle};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use crate::attractor::{Attractor, Point};

const BATCH: usize = 65_536;

pub enum SimCommand {
    /// Replace parameters (same length as attractor's param list).
    SetParams(Vec<f32>),
    /// Reset the integrator state without changing parameters.
    Reset,
    /// Shut down the worker thread.
    Stop,
}

pub struct SimWorker {
    pub rx: Receiver<Vec<Point>>,
    cmd_tx: Sender<SimCommand>,
    handle: Option<JoinHandle<()>>,
}

impl SimWorker {
    /// Spawns a background thread running `attractor` and returns the handle.
    pub fn spawn<A: Attractor>(mut attractor: A) -> Self {
        let (pt_tx, pt_rx) = bounded::<Vec<Point>>(8);
        let (cmd_tx, cmd_rx) = bounded::<SimCommand>(16);

        let handle = thread::spawn(move || {
            let mut buf: Vec<Point> = Vec::with_capacity(BATCH);

            'outer: loop {
                // Drain pending commands without blocking.
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        SimCommand::SetParams(p) => attractor.reset(&p),
                        SimCommand::Reset        => attractor.reset(&attractor.params().to_vec()),
                        SimCommand::Stop         => break 'outer,
                    }
                }

                // Generate a full batch of points.
                buf.clear();
                for _ in 0..BATCH {
                    match attractor.step() {
                        Some(pt) => buf.push(pt),
                        None => {
                            // Orbit escaped or diverged — restart from default IC.
                            // Discard pre-divergence points so the batch is pure
                            // post-reset data; the break sends whatever partial count
                            // was accumulated (possibly zero).
                            attractor.reset(&attractor.params().to_vec());
                            buf.clear();
                            break;
                        }
                    }
                }

                // Move the filled batch into the channel.  On success the receiver
                // owns the allocation; on failure we recycle it rather than dropping
                // and immediately re-allocating.
                let outgoing = std::mem::replace(&mut buf, Vec::new());
                match pt_tx.try_send(outgoing) {
                    Ok(())                              => buf.reserve(BATCH),
                    Err(TrySendError::Full(returned))   => { buf = returned; buf.clear(); }
                    Err(TrySendError::Disconnected(_))  => break 'outer,
                }
            }
        });

        SimWorker { rx: pt_rx, cmd_tx, handle: Some(handle) }
    }

    /// Send a command to the worker.  Returns `true` if delivered, `false` if
    /// the channel was full.  `Stop` delivery is guaranteed by the `Drop` impl.
    pub fn send(&self, cmd: SimCommand) -> bool {
        self.cmd_tx.try_send(cmd).is_ok()
    }
}

impl Drop for SimWorker {
    fn drop(&mut self) {
        // Blocking send guarantees Stop is delivered even if the channel was full.
        let _ = self.cmd_tx.send(SimCommand::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
